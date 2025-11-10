# SPDX-License-Identifier: MIT
# src/event_feed_app/alerts/detector.py
"""
Alert detection logic for identifying significant guidance changes and other events.
"""
from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from event_feed_app.events.registry import get_plugin
from .store import AlertStore

logger = logging.getLogger(__name__)


class GuidanceAlertDetector:
    """
    Detects alert-worthy guidance changes from press releases and earnings reports.
    
    Uses the GuidanceChangePlugin to identify significant guidance updates and
    generates structured alert objects for delivery to users.
    """
    
    def __init__(
        self,
        min_significance: float = 0.7,
        store: Optional[AlertStore] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the guidance alert detector.
        
        Args:
            min_significance: Minimum significance score (0-1) to trigger an alert
            store: AlertStore instance for persisting state (optional)
            config: Additional configuration (optional)
        """
        self.guidance_plugin = get_plugin("guidance_change")
        self.min_significance = min_significance
        self.store = store or AlertStore()
        self.config = config or {}
        
        # Configure the plugin if config provided
        if self.config.get("guidance_plugin_config"):
            # Type: ignore because EventPlugin protocol doesn't define configure
            # but GuidanceChangePlugin has it
            self.guidance_plugin.configure(self.config["guidance_plugin_config"])  # type: ignore
    
    def detect_alerts(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process documents and return alert-worthy guidance changes.
        
        Args:
            docs: List of normalized documents (from Silver layer or orchestrator output)
                  Expected fields: press_release_id, title_clean, full_text_clean, 
                  release_date, company_name (optional)
        
        Returns:
            List of alert objects ready for delivery
        """
        alerts = []
        gated_count = 0
        passed_gate_count = 0
        candidates_total = 0
        
        for doc in docs:
            try:
                normalized_doc = self._normalize_doc_for_plugin(doc)
                if self.guidance_plugin.gate(normalized_doc):
                    passed_gate_count += 1
                    candidates = list(self.guidance_plugin.detect(normalized_doc))
                    candidates_total += len(candidates)
                else:
                    gated_count += 1
                
                doc_alerts = self._process_document(doc)
                alerts.extend(doc_alerts)
            except Exception as e:
                logger.error(
                    f"Error processing document {doc.get('press_release_id', 'unknown')}: {e}",
                    exc_info=True
                )
        
        logger.info(f"Gate stats: {passed_gate_count} passed, {gated_count} gated out of {len(docs)} total")
        logger.info(f"Candidates: {candidates_total} total from {passed_gate_count} docs")
        logger.info(f"Detected {len(alerts)} alerts from {len(docs)} documents")
        return alerts
    
    def _process_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single document and return 0+ alerts."""
        alerts = []
        
        # Normalize document format for plugin
        normalized_doc = self._normalize_doc_for_plugin(doc)
        
        # Debug: Log first doc to verify fields
        if not hasattr(self, '_logged_first_doc'):
            self._logged_first_doc = True
            logger.info(f"[DEBUG] First doc keys: {list(doc.keys())[:15]}")
            logger.info(f"[DEBUG] Normalized doc: title={normalized_doc.get('title', '')[:50]}, "
                       f"body_len={len(normalized_doc.get('body', ''))}, "
                       f"company_name={normalized_doc.get('company_name', 'MISSING')}")
        
        # NOTE: Gate is commented out in plugin2.py (intentionally too restrictive for alerts)
        # We skip gating and rely on detect() method's internal logic for precision
        
        # Detect candidates
        candidates = list(self.guidance_plugin.detect(normalized_doc))
        
        if candidates and not hasattr(self, '_logged_first_candidate'):
            self._logged_first_candidate = True
            logger.info(f"[DEBUG] First candidate keys: {list(candidates[0].keys())}")
            logger.info(f"[DEBUG] First candidate: metric={candidates[0].get('metric')}, "
                       f"period={candidates[0].get('period')}, "
                       f"rule_type={candidates[0].get('_rule_type')}")
        
        logger.debug(f"Found {len(candidates)} guidance candidates in {doc.get('press_release_id')}")
        
        for cand in candidates:
            # Normalize candidate
            norm_cand = self.guidance_plugin.normalize(cand)
            
            # Inject doc context fields needed by plugin methods
            norm_cand["_doc_title"] = normalized_doc.get("title", "")
            norm_cand["_doc_body"] = normalized_doc.get("body", "")
            norm_cand["company_id"] = normalized_doc.get("company_id", "")
            norm_cand["company_name"] = normalized_doc.get("company_name", "")
            
            # Compare with prior state (if any)
            prior_key = self.guidance_plugin.prior_key(norm_cand)
            prior = self._load_prior_state(prior_key)
            comparison = self.guidance_plugin.compare(norm_cand, prior)
            
            # Score significance
            # Type: ignore because EventPlugin protocol doesn't define cfg
            # but GuidanceChangePlugin has it
            plugin_cfg = getattr(self.guidance_plugin, 'cfg', {})
            is_sig, score, features = self.guidance_plugin.decide_and_score(
                norm_cand, comparison, plugin_cfg
            )
            
            logger.debug(
                f"Candidate scored: is_sig={is_sig}, score={score:.3f}, "
                f"threshold={self.min_significance}"
            )
            
            # Emit alert if significant enough OR if config requests alerting all guidance commentary
            if (is_sig and score >= self.min_significance) or self.config.get("alert_all_guidance", False):
                # Build alert object
                alert = self._build_alert(doc, normalized_doc, norm_cand, comparison, score, features)
                alerts.append(alert)
                
                # Update prior state for future comparisons
                self._save_prior_state(prior_key, norm_cand)
                
                logger.info(
                    f"Alert generated: {alert['alert_id']} for {alert.get('company_name', 'unknown')} "
                    f"(score={score:.3f})"
                )
        
        return alerts
    
    def _normalize_doc_for_plugin(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize document to format expected by GuidanceChangePlugin.
        
        Plugin expects: title, body, company_name (optional), period_doc (optional)
        Gate also checks: l1_class, rule_pred_cid, rule_label for earnings keywords
        """
        # Generate company_id if missing (use company_name as fallback)
        company_id = doc.get("company_id") or doc.get("ticker") or doc.get("company_name") or "unknown"
        company_name = doc.get("company_name") or "Unknown Company"
        
        # Map category to l1_class for gate compatibility
        l1_class = doc.get("l1_class") or doc.get("category") or doc.get("rule_label") or ""
        
        return {
            "title": doc.get("title_clean") or doc.get("title") or "",
            "body": doc.get("full_text_clean") or doc.get("content") or doc.get("full_text") or "",
            "company_id": company_id,
            "company_name": company_name,
            "l1_class": l1_class,  # For gate() earnings label check
            "period_doc": doc.get("period") or "",
            "press_release_id": doc.get("press_release_id") or doc.get("id") or "",
            "release_date": doc.get("release_date") or doc.get("timestamp") or "",
            "source_url": doc.get("source_url") or doc.get("url") or doc.get("link") or "",
        }
    
    def _build_alert(
        self,
        orig_doc: Dict[str, Any],
        normalized_doc: Dict[str, Any],
        candidate: Dict[str, Any],
        comparison: Dict[str, Any],
        score: float,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build structured alert object."""
        
        # Generate unique alert ID
        alert_id = self._generate_alert_id(orig_doc, candidate)
        
        # Extract company information
        company_name = (
            orig_doc.get("company_name") or 
            normalized_doc.get("company_name") or 
            candidate.get("company_name") or 
            "Unknown Company"
        )
        
        # Build human-readable summary
        summary = self._generate_summary(candidate, comparison)
        
        # Extract metrics with changes
        metrics = self._extract_metrics(candidate, comparison)
        
        return {
            "alert_id": alert_id,
            "event_id": orig_doc.get("press_release_id") or orig_doc.get("id"),
            "alert_type": "guidance_change",
            "company_name": company_name,
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "significance_score": round(score, 3),
            "summary": summary,
            "metrics": metrics,
            "metadata": {
                "press_release_url": orig_doc.get("source_url") or orig_doc.get("url") or orig_doc.get("link"),
                "release_date": orig_doc.get("release_date") or orig_doc.get("timestamp"),
                "direction": comparison.get("direction"),
                "change_magnitude": comparison.get("magnitude"),
                "period": candidate.get("period") or normalized_doc.get("period_doc"),
                "features": features,  # Debug info
            }
        }
    
    def _generate_summary(self, candidate: Dict[str, Any], comparison: Dict[str, Any]) -> str:
        """Generate human-readable summary of guidance change."""
        parts = []
        
        # Extract direction
        direction = comparison.get("direction", "updated")
        if direction == "up":
            action = "raised"
        elif direction == "down":
            action = "lowered"
        else:
            action = "updated"
        
        # Extract metrics
        metrics = candidate.get("metrics") or []
        if metrics:
            metric_names = [m.get("metric", "guidance") for m in metrics[:2]]  # First 2
            metric_str = ", ".join(metric_names)
            parts.append(f"{metric_str.title()} {action}")
        else:
            parts.append(f"Guidance {action}")
        
        # Add period if available
        period = candidate.get("period")
        if period and period not in ("UNKNOWN", "PERIOD-UNKNOWN"):
            parts.append(f"for {period}")
        
        return " ".join(parts)
    
    def _extract_metrics(
        self, 
        candidate: Dict[str, Any], 
        comparison: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract structured metrics from candidate."""
        metrics = []
        
        for metric_data in candidate.get("metrics") or []:
            metric_entry = {
                "metric": metric_data.get("metric", "unknown"),
                "value": metric_data.get("value"),
                "range": metric_data.get("range"),
                "unit": metric_data.get("unit"),
                "direction": metric_data.get("direction"),
                "basis": metric_data.get("basis"),  # organic, cc_fx, etc.
            }
            
            # Add comparison data if available
            if comparison.get("changes"):
                for change in comparison["changes"]:
                    if change.get("metric") == metric_entry["metric"]:
                        metric_entry["prior_value"] = change.get("prior_value")
                        metric_entry["change_pct"] = change.get("change_pct")
            
            metrics.append(metric_entry)
        
        return metrics
    
    def _generate_alert_id(self, doc: Dict[str, Any], candidate: Dict[str, Any]) -> str:
        """Generate unique alert ID."""
        parts = [
            doc.get("press_release_id") or doc.get("id") or "",
            candidate.get("company_name") or doc.get("company_name") or "",
            candidate.get("period") or "",
            str(datetime.now(timezone.utc).timestamp()),
        ]
        content = "|".join(str(p) for p in parts)
        return hashlib.blake2s(content.encode("utf-8"), digest_size=16).hexdigest()
    
    def _load_prior_state(self, prior_key: tuple) -> Optional[Dict[str, Any]]:
        """Load prior state for comparison (e.g., from database or cache)."""
        if not prior_key or not self.store:
            return None
        
        try:
            return self.store.get_prior_state(prior_key)
        except Exception as e:
            logger.warning(f"Failed to load prior state for {prior_key}: {e}")
            return None
    
    def _save_prior_state(self, prior_key: tuple, candidate: Dict[str, Any]):
        """Save candidate as new prior state for future comparisons."""
        if not prior_key or not self.store:
            return
        
        try:
            self.store.save_prior_state(prior_key, candidate)
        except Exception as e:
            logger.error(f"Failed to save prior state for {prior_key}: {e}")
