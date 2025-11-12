# SPDX-License-Identifier: MIT
# src/event_feed_app/alerts/detector.py
"""
Alert detection logic for identifying significant guidance changes and other events.
"""
from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Sequence
import os
from urllib.parse import quote_plus

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

    def _resolve_press_release_url(self, orig_doc: Dict[str, Any], normalized_doc: Dict[str, Any]) -> str:
        """Best-effort URL resolution with configurable fallback.
        Priority:
          1) normalized_doc.source_url (if non-empty)
          2) orig_doc.source_url | url | link (if non-empty)
          3) config.document_base_url + "/" + press_release_id
          4) env DOCUMENT_BASE_URL + "/" + press_release_id
          5) Synthetic review URL using press_release_id (always present)
        
        Note: Source documents from Gmail have empty source_url, so fallback is needed.
        """
        # Check for existing URL (must be non-empty)
        url = (
            normalized_doc.get("source_url")
            or orig_doc.get("source_url")
            or orig_doc.get("url")
            or orig_doc.get("link")
            or ""
        ).strip()
        
        if url:
            return url
        
        # Get document identifier
        doc_id = orig_doc.get("press_release_id") or orig_doc.get("id") or ""
        
        # Try configured base URL
        base = (self.config.get("document_base_url")
                or os.getenv("DOCUMENT_BASE_URL")
                or "").strip()
        
        if base and doc_id:
            return f"{base.rstrip('/')}/{doc_id}"
        
        # Final synthetic fallback: configurable viewer URL
        # Options:
        #   1. Local network: http://192.168.1.100:8501 (set via VIEWER_BASE_URL)
        #   2. Ngrok tunnel: https://abc123.ngrok.io (temporary public URL)
        #   3. Cloud deployment: https://your-app.run.app (permanent)
        # Start viewer: streamlit run scripts/view_pr_web.py --server.address 0.0.0.0
        if doc_id:
            viewer_base = os.getenv("VIEWER_BASE_URL", "http://localhost:8501")
            return f"{viewer_base.rstrip('/')}/?id={doc_id}"
        
        # Last resort: provide manual lookup instruction
        return f"View with: python scripts/view_press_releases.py --id {doc_id or 'UNKNOWN'}"
    
    def detect_alerts(self, docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process documents and return alert-worthy guidance changes.
        
        Args:
            docs: Sequence of normalized documents (from Silver layer or orchestrator output)
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
        """
        Process a single document and return alerts.
        
        NOTE: This now aggregates all candidates from the same document into ONE alert
        to prevent spam. Each alert contains multiple guidance_items.
        """
        # Normalize for plugin
        normalized_doc = self._normalize_doc_for_plugin(doc)
        
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
        
        # Collect all significant guidance items for this document
        guidance_items = []
        max_score = 0.0
        
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
            
            # Collect significant guidance items
            if (is_sig and score >= self.min_significance) or self.config.get("alert_all_guidance", False):
                guidance_items.append({
                    "candidate": norm_cand,
                    "comparison": comparison,
                    "score": score,
                    "features": features
                })
                max_score = max(max_score, score)
                
                # Update prior state for future comparisons
                self._save_prior_state(prior_key, norm_cand)
        
        # If we have any significant guidance, create ONE aggregated alert for this document
        if guidance_items:
            alert = self._build_aggregated_alert(doc, normalized_doc, guidance_items, max_score)
            logger.info(
                f"Alert generated: {alert['alert_id']} for {alert.get('company_name', 'unknown')} "
                f"with {len(guidance_items)} guidance items (max_score={max_score:.3f})"
            )
            return [alert]
        
        return []
    
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
                # Robust URL resolution (with fallback to config/env base URL)
                "press_release_url": self._resolve_press_release_url(orig_doc, normalized_doc),
                "release_date": orig_doc.get("release_date") or orig_doc.get("timestamp"),
                "direction": comparison.get("direction"),
                "change_magnitude": comparison.get("magnitude"),
                "period": candidate.get("period") or normalized_doc.get("period_doc"),
                "features": features,  # Debug info
                # Helpful context when URL is missing
                "title": normalized_doc.get("title", ""),
                "body_snippet": self._extract_snippet(orig_doc, normalized_doc, max_chars=350),
            }
        }
    
    def _build_aggregated_alert(
        self,
        orig_doc: Dict[str, Any],
        normalized_doc: Dict[str, Any],
        guidance_items: List[Dict[str, Any]],
        max_score: float
    ) -> Dict[str, Any]:
        """
        Build a single alert object aggregating multiple guidance candidates.
        
        Args:
            orig_doc: Original document dict
            normalized_doc: Normalized document for plugin
            guidance_items: List of dicts with keys: candidate, comparison, score, features
            max_score: Highest score among all candidates
        
        Returns:
            Single alert object with multiple guidance items
        """
        # Generate unique alert ID based on document
        alert_id = hashlib.sha256(
            f"{orig_doc.get('press_release_id')}_{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Extract company information (from first item, should be same for all)
        company_name = (
            orig_doc.get("company_name") or 
            normalized_doc.get("company_name") or 
            guidance_items[0]["candidate"].get("company_name") or 
            "Unknown Company"
        )
        
        # Build aggregated metrics list
        all_metrics = []
        all_summaries = []
        
        # Build detailed guidance items for viewer
        detailed_guidance_items = []
        
        for item in guidance_items:
            candidate = item["candidate"]
            comparison = item["comparison"]
            
            # Generate summary for this item
            summary = self._generate_summary(candidate, comparison)
            all_summaries.append(summary)
            
            # Extract metrics
            metrics = self._extract_metrics(candidate, comparison)
            all_metrics.extend(metrics)
            
            # Build detailed item for viewer display
            detailed_item = {
                "metric": candidate.get("metric") or "unknown",
                "direction": comparison.get("direction") or "unknown",
                "period": candidate.get("period") or "UNKNOWN",
                "value_str": self._format_value_change(candidate, comparison),
                "confidence": item.get("score", 0.0),
                "text_snippet": candidate.get("snippet") or candidate.get("text") or ""
            }
            detailed_guidance_items.append(detailed_item)
        
        # Create combined summary
        if len(guidance_items) == 1:
            combined_summary = all_summaries[0]
        else:
            combined_summary = f"{len(guidance_items)} guidance updates: " + "; ".join(all_summaries[:3])
            if len(all_summaries) > 3:
                combined_summary += f" ... and {len(all_summaries) - 3} more"
        
        return {
            "alert_id": alert_id,
            "event_id": orig_doc.get("press_release_id") or orig_doc.get("id"),
            "alert_type": "guidance_change",
            "company_name": company_name,
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "significance_score": round(max_score, 3),
            "summary": combined_summary,
            "metrics": all_metrics,  # All metrics from all candidates
            "guidance_count": len(guidance_items),
            "guidance_items": detailed_guidance_items,  # Structured items for viewer
            "metadata": {
                # Robust URL resolution (with fallback to config/env base URL)
                "press_release_url": self._resolve_press_release_url(orig_doc, normalized_doc),
                "release_date": orig_doc.get("release_date") or orig_doc.get("timestamp"),
                "period": guidance_items[0]["candidate"].get("period") or normalized_doc.get("period_doc"),
                # Full content for viewer (so we don't need to query silver data)
                "title": normalized_doc.get("title", ""),
                "body": normalized_doc.get("body", ""),  # Full text instead of snippet
                "source": orig_doc.get("source", ""),
                "category": orig_doc.get("l1_class") or orig_doc.get("category", ""),
            }
        }

    def _extract_snippet(self, orig_doc: Dict[str, Any], normalized_doc: Dict[str, Any], max_chars: int = 350) -> str:
        """Extract a short, readable snippet from the document body.
        Prefers cleaned text. Returns at most max_chars, trimmed at a word boundary.
        """
        body = (
            normalized_doc.get("body")
            or orig_doc.get("full_text_clean")
            or orig_doc.get("full_text")
            or ""
        )
        if not isinstance(body, str) or not body.strip():
            return ""
        text = body.strip()
        # Use first non-empty paragraph
        paras = [p.strip() for p in text.splitlines() if p.strip()]
        if not paras:
            return ""
        first = paras[0]
        # Collapse excessive spaces
        first = " ".join(first.split())
        if len(first) <= max_chars:
            return first
        # Trim to nearest word boundary within max_chars
        cut = first[:max_chars]
        # try to cut at the last period or space
        for sep in [". ", ".", " "]:
            idx = cut.rfind(sep)
            if idx >= max(40, int(0.6 * max_chars)):
                return cut[:idx+1].rstrip()
        return cut.rstrip() + "…"

    
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
    
    def _format_value_change(self, candidate: Dict[str, Any], comparison: Dict[str, Any]) -> str:
        """Format a human-readable value change string from candidate/comparison."""
        metrics = candidate.get("metrics") or []
        if not metrics:
            return ""
        
        # Use first metric for simplicity
        metric_data = metrics[0]
        value = metric_data.get("value")
        value_range = metric_data.get("range")
        unit = metric_data.get("unit", "")
        
        # Check for prior value in comparison
        prior_value = None
        change_pct = None
        if comparison.get("changes"):
            for change in comparison["changes"]:
                if change.get("metric") == metric_data.get("metric"):
                    prior_value = change.get("prior_value")
                    change_pct = change.get("change_pct")
                    break
        
        # Format: "prior → new (change%)" or just "new" or "range"
        parts = []
        if prior_value is not None:
            parts.append(f"{prior_value}{unit}")
            parts.append("→")
        
        if value_range:
            parts.append(f"{value_range[0]}-{value_range[1]}{unit}")
        elif value is not None:
            parts.append(f"{value}{unit}")
        
        if change_pct is not None:
            parts.append(f"({change_pct:+.0f}%)")
        
        return " ".join(parts) if parts else "N/A"
    
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
