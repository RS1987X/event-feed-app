# SPDX-License-Identifier: MIT
# src/event_feed_app/signals/schema.py
"""
Parquet schema definitions for signal storage.

These schemas define the canonical structure for storing detected events
in a queryable, columnar format optimized for analytics and time-series queries.
"""

import pyarrow as pa
from typing import Dict, Any, List
from datetime import datetime


class GuidanceEventSchema:
    """
    Schema for individual guidance change events.
    
    One row per detected event (metric/period combination).
    Optimized for time-series queries and cross-company comparisons.
    """
    
    @staticmethod
    def get_schema() -> pa.Schema:
        """Return PyArrow schema for guidance events."""
        return pa.schema([
            # Identifiers
            ("event_id", pa.string()),                    # Unique: "{pr_id}_{metric}_{period}_{idx}"
            ("press_release_id", pa.string()),            # Links to silver layer
            ("alert_id", pa.string()),                    # Links to aggregated alert
            ("company_id", pa.string()),                  # Normalized company identifier
            ("company_name", pa.string()),                # Display name
            ("detected_at", pa.timestamp("us", tz="UTC")), # Detection timestamp
            
            # Parametrization (core signal data)
            ("metric", pa.string()),                      # revenue, ebitda, margin, eps, etc.
            ("metric_kind", pa.string()),                 # level, growth, margin
            ("period", pa.string()),                      # FY2025, Q3-2025, FY2025-FY2027
            ("value_low", pa.float64()),                  # Lower bound of range
            ("value_high", pa.float64()),                 # Upper bound of range
            ("unit", pa.string()),                        # ccy, pct, text
            ("currency", pa.string()),                    # USD, EUR, SEK, null
            ("basis", pa.string()),                       # reported, organic, cc_fx
            ("direction_hint", pa.string()),              # up, down, flat, null
            
            # Comparison with prior guidance (if exists)
            ("prior_value_low", pa.float64()),            # Previous guidance lower bound
            ("prior_value_high", pa.float64()),           # Previous guidance upper bound
            ("prior_mid", pa.float64()),                  # Midpoint of prior range
            ("delta_pp", pa.float64()),                   # Percentage point change (for pct metrics)
            ("delta_pct", pa.float64()),                  # Percent change (for ccy metrics)
            ("range_tightened", pa.bool_()),              # Range became narrower
            ("change_label", pa.string()),                # raise, lower, narrow, widen, initiate, maintain
            
            # Detection metadata
            ("confidence", pa.float64()),                 # Significance score 0-1
            ("rule_type", pa.string()),                   # guidance_change, guidance_initiate, etc.
            ("rule_name", pa.string()),                   # Specific proximity/numeric rule matched
            ("trigger_source", pa.string()),              # What triggered detection
            ("text_snippet", pa.string()),                # Context excerpt (200 chars max)
            
            # Minimal pointers (not full text!)
            ("press_release_date", pa.date32()),          # Publication date
            ("source_url", pa.string()),                  # Link to original PR
            ("source_type", pa.string()),                 # issuer_pr, exchange, regulator, etc.
        ])
    
    @staticmethod
    def event_to_dict(event: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a plugin event + context to schema-compliant dict.
        
        Args:
            event: Event dict from plugin.detect()
            context: Additional context (press_release_id, company info, comparison, etc.)
        
        Returns:
            Dict matching GuidanceEventSchema
        """
        comparison = context.get("comparison", {})
        
        return {
            # Identifiers
            "event_id": context.get("event_id"),
            "press_release_id": context.get("press_release_id"),
            "alert_id": context.get("alert_id"),
            "company_id": context.get("company_id"),
            "company_name": context.get("company_name"),
            "detected_at": datetime.fromisoformat(context.get("detected_at", datetime.utcnow().isoformat())),
            
            # Parametrization
            "metric": event.get("metric"),
            "metric_kind": event.get("metric_kind"),
            "period": event.get("period"),
            "value_low": event.get("value_low"),
            "value_high": event.get("value_high"),
            "unit": event.get("unit"),
            "currency": event.get("currency"),
            "basis": event.get("basis"),
            "direction_hint": event.get("direction_hint"),
            
            # Comparison
            "prior_value_low": comparison.get("old_low"),
            "prior_value_high": comparison.get("old_high"),
            "prior_mid": comparison.get("old_mid"),
            "delta_pp": comparison.get("delta_pp"),
            "delta_pct": comparison.get("delta_pct"),
            "range_tightened": comparison.get("range_tightened"),
            "change_label": comparison.get("change_label"),
            
            # Metadata
            "confidence": context.get("confidence", 0.0),
            "rule_type": event.get("_rule_type"),
            "rule_name": event.get("_rule_name"),
            "trigger_source": event.get("_trigger_source"),
            "text_snippet": context.get("text_snippet", "")[:200],  # Limit to 200 chars
            
            # Pointers
            "press_release_date": context.get("press_release_date"),
            "source_url": context.get("source_url"),
            "source_type": context.get("source_type"),
        }


class AggregatedAlertSchema:
    """
    Schema for aggregated alerts (one per press release).
    
    Summarizes all events detected in a single document.
    Useful for alert delivery and document-level queries.
    """
    
    @staticmethod
    def get_schema() -> pa.Schema:
        """Return PyArrow schema for aggregated alerts."""
        return pa.schema([
            # Alert identifiers
            ("alert_id", pa.string()),                    # Deterministic hash of press_release_id
            ("press_release_id", pa.string()),            # Links to silver layer
            ("company_id", pa.string()),                  # Primary company
            ("company_name", pa.string()),                # Display name
            ("detected_at", pa.timestamp("us", tz="UTC")),
            
            # Summary
            ("event_count", pa.int32()),                  # Number of individual events
            ("max_confidence", pa.float64()),             # Highest confidence score
            ("summary", pa.string()),                     # Human-readable summary
            ("primary_period", pa.string()),              # Most common period
            
            # Aggregated metrics (JSON array of guidance items)
            ("guidance_items_json", pa.string()),         # JSON array of simplified items
            
            # Pointers
            ("press_release_date", pa.date32()),
            ("source_url", pa.string()),
        ])
    
    @staticmethod
    def alert_to_dict(alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an aggregated alert to schema-compliant dict.
        
        Args:
            alert: Alert dict from detector._build_aggregated_alert()
        
        Returns:
            Dict matching AggregatedAlertSchema
        """
        import json
        
        # Simplify guidance_items for storage (keep only essential fields)
        simplified_items = [
            {
                "metric": item.get("metric"),
                "period": item.get("period"),
                "direction": item.get("direction"),
                "value_str": item.get("value_str"),
                "confidence": item.get("confidence"),
            }
            for item in alert.get("guidance_items", [])
        ]
        
        return {
            "alert_id": alert.get("alert_id"),
            "press_release_id": alert.get("event_id"),  # Note: event_id is actually press_release_id
            "company_id": alert.get("company_id"),  # May need to extract from metadata
            "company_name": alert.get("company_name"),
            "detected_at": datetime.fromisoformat(alert.get("detected_at")),
            
            "event_count": alert.get("guidance_count", 0),
            "max_confidence": alert.get("significance_score", 0.0),
            "summary": alert.get("summary"),
            "primary_period": alert.get("metadata", {}).get("period"),
            
            "guidance_items_json": json.dumps(simplified_items),
            
            "press_release_date": alert.get("metadata", {}).get("release_date"),
            "source_url": alert.get("metadata", {}).get("press_release_url"),
        }


# Partition strategies for efficient querying
PARTITION_SPECS = {
    "events": {
        "by_date": ["detected_date"],  # Partitioned by date(detected_at)
        "by_company": ["company_id", "detected_date"],  # For company-specific queries
        "by_period": ["period", "detected_date"],  # For cross-company period analysis
    },
    "aggregated": {
        "by_date": ["detected_date"],
    }
}
