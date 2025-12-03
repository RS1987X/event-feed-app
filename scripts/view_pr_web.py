#!/usr/bin/env python3
"""
Simple web viewer for press releases by ID with alert feedback.

Usage:
    # Start the server
    streamlit run scripts/view_pr_web.py
    
    # Then open in browser:
    http://localhost:8501/?id=1980c93f9f819a6d&alert_id=alert_abc123
    
This provides a clickable URL for alert messages when source_url is missing.
Users can also flag whether the alert was correct or incorrect.
"""
import os
import sys
from pathlib import Path
from typing import Any
import streamlit as st

# Add project to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from load_env import load_env  # type: ignore
load_env()

from event_feed_app.config import Settings  # type: ignore
from event_feed_app.utils.io import load_from_gcs  # type: ignore
from event_feed_app.alerts.feedback_store import FeedbackStore  # type: ignore
from event_feed_app.alerts.alert_payload_store import AlertPayloadStore  # type: ignore

st.set_page_config(page_title="Press Release Viewer", layout="wide")

# Build stamp helps verify deployed version
from datetime import datetime, timezone
BUILD_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# Lightweight CSS for cleaner detection formatting
st.markdown(
    """
    <style>
    .det-card {border:1px solid #2a2a2a;background:#111;border-radius:12px;padding:14px;margin:8px 0;}
    .det-title {font-weight:600;color:#c9c9c9;margin-bottom:8px;}
    .badges {display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 10px;}
    .badge {background:#1a1a1a;border:1px solid #333;border-radius:999px;padding:4px 10px;font-size:0.95rem;color:#ddd;}
    .badge .label {color:#9a9a9a;margin-right:6px;}
    .badge.up {border-color:#2e7d32;color:#9be07e;}
    .badge.down {border-color:#c62828;color:#ff9e9e;}
    .badge.unchanged {border-color:#607d8b;color:#cfd8dc;}
    .badge.unknown {border-color:#8e8e8e;color:#e0e0e0;}
    .value-box {background:#0f1b2d;border:1px solid #1e3a5f;color:#cfe7ff;border-radius:8px;padding:10px 12px;margin:6px 0 10px;}
    .value-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:6px 0 10px;}
    .vbox {background:#0f1b2d;border:1px solid #1e3a5f;color:#cfe7ff;border-radius:8px;padding:10px 12px;}
    .vbox .label {color:#97b7d7;font-size:0.85rem;display:block;margin-bottom:4px;}
    .vbox .val {font-size:1.1rem;font-weight:600;}
    .excerpt {background:#0f0f0f;border-radius:8px;padding:12px;white-space:pre-wrap;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;color:#d7d7d7;border:1px solid #222;}
    .unknown-icon {color:#ff5252;font-weight:700;margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper: parse structured value from value_str (supports ranges and prior→current)
def _parse_value_str(value_str: str):
    import re
    if not value_str:
        return {"present": False}
    s = value_str.strip()
    # Extract optional trailing change like "(+10%)" or "(+1.5pp)"
    change_text = None
    m_change = re.search(r"\(([^)]+)\)\s*$", s)
    if m_change:
        change_text = m_change.group(1).strip()
        s = s[: m_change.start()].strip()

    def parse_part(part: str):
        part = part.strip()
        # Range "a-b" or "a – b" or "a to b"
        m_range = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(%?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)\s*(%?)\s*$", part)
        if m_range:
            low, suf1, high, suf2 = m_range.groups()
            unit = suf1 or suf2
            return {"is_range": True, "low": low, "high": high, "unit": unit}
        # Single value
        m_single = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(%?)\s*$", part)
        if m_single:
            val, unit = m_single.groups()
            return {"is_range": False, "value": val, "unit": unit}
        # Fallback: return raw text
        return {"is_range": False, "value": part, "unit": ""}

    if "→" in s:
        prior_part, current_part = [p.strip() for p in s.split("→", 1)]
        prior = parse_part(prior_part)
        current = parse_part(current_part)
        return {"present": True, "has_prior": True, "prior": prior, "current": current, "change": change_text}

    parsed = parse_part(s)
    return {"present": True, "has_prior": False, "current": parsed, "change": change_text}

# Get parameters from URL
params = st.query_params
press_release_id = params.get("id", "")
alert_id = params.get("alert_id", "")
signal_type = params.get("signal_type", "guidance_change")

if not press_release_id or not alert_id:
    st.title("📄 Press Release Viewer")
    st.info("To view a press release, add `?id=<press_release_id>&alert_id=<alert_id>&signal_type=<type>` to the URL")
    st.markdown("**Example:** `http://localhost:8501/?id=1980c93f9f819a6d&alert_id=abc123&signal_type=guidance_change`")
    st.markdown("---")
    st.markdown("This viewer is used by the alert system to display press releases with feedback forms.")
    st.stop()

# Load alert payload (contains PR content) - MUCH faster than loading from silver data
@st.cache_data(ttl=3600)
def load_alert_payload(alert_id: str, signal_type: str):
    """Load alert payload which includes the full PR content."""
    store = AlertPayloadStore()
    return store.get_alert_payload(alert_id, signal_type)

# Check if this is a false negative review (alert_id starts with "fn_")
is_false_negative_review = alert_id.startswith("fn_")

with st.spinner("Loading press release..."):
    # Load from alert payload store (works for both positives and negatives now)
    alert = load_alert_payload(alert_id, signal_type)
    
    if alert and is_false_negative_review:
        st.info("📝 **False Negative Review**: This PR was NOT flagged. Review to see if guidance was missed.")

if not alert:
    st.error(f"❌ Press release not found: `{press_release_id}`")
    st.info("The press release may be too old or the ID is incorrect.")
    st.stop()

# Extract fields from alert payload metadata
metadata = alert.get("metadata", {})
title = metadata.get("title", "Untitled")
company = alert.get("company_name", "Unknown Company")
release_date = metadata.get("release_date", "Unknown Date")
body = metadata.get("body", "")  # Full content stored in alert payload
source_url = metadata.get("press_release_url", "")
category = metadata.get("category", "")
source = metadata.get("source", "")

# Display document
st.title(f"📄 {title}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Company", company)
with col2:
    st.metric("Release Date", release_date)
with col3:
    if category:
        st.metric("Category", category)

# Prominent link to original press release
if source_url:
    st.info(f"🔗 **View Original Press Release:** [{source_url}]({source_url})")
else:
    st.warning("⚠️ Original source URL not available")

st.markdown("---")

# Display body with proper formatting
if body:
    st.markdown("### Content")
    # Gather detected text snippets for highlighting (case-insensitive)
    detected_snippets = []
    try:
        for gi in alert.get("guidance_items", []):
            ts = gi.get("text_snippet")
            if ts and isinstance(ts, str):
                detected_snippets.append(ts.strip())
    except Exception:
        pass

    # Deduplicate snippets and sort longest-first to avoid nested replacements
    detected_snippets = sorted(list({s for s in detected_snippets if s}), key=len, reverse=True)

    # Split into paragraphs for better readability
    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]

    def highlight_para(text: str) -> str:
        # Apply HTML <mark> highlighting for each detected snippet
        highlighted = text
        for s in detected_snippets:
            try:
                # Escape regex special chars from snippet
                import re
                pattern = re.escape(s)
                # Replace all case-insensitive occurrences with a bold, bright yellow highlight
                highlighted = re.sub(
                    pattern,
                    lambda m: f"<mark style='background-color:#ffeb3b;padding:2px 4px;border-radius:3px;font-weight:600;font-size:1.05em;border:2px solid #fdd835'>{m.group(0)}</mark>",
                    highlighted,
                    flags=re.IGNORECASE,
                )
            except Exception:
                # Fallback: leave paragraph unchanged if any error
                pass
        return highlighted

    for para in paragraphs:
        # Use unsafe_allow_html to render <mark> tags
        st.markdown(highlight_para(para), unsafe_allow_html=True)
        st.markdown("")  # Add spacing
else:
    st.warning("No content available for this press release.")

# Feedback section
st.markdown("---")

st.subheader("🔍 What the System Detected")

# Display detected guidance items from alert
guidance_items = alert.get("guidance_items", [])

# Initialize session state for item feedback
if "item_feedback" not in st.session_state:
    st.session_state.item_feedback = {}
if "missing_items" not in st.session_state:
    st.session_state.missing_items = []

if guidance_items:
    for idx, item in enumerate(guidance_items):
        item_key = f"item_{idx}"

        # Header uses uppercase metric for consistency
        item_type = (item.get("item_type") or "").lower()
        header_label = (item.get("metric") or "Unknown").upper()
        if item_type == "guidance_generic":
            header_label = "GUIDANCE"
        with st.expander(f"Detection #{idx+1}: {header_label}", expanded=True):
            # Card wrapper
            st.markdown("<div class='det-card'>", unsafe_allow_html=True)

            # Badges row
            direction = (item.get("direction") or "unknown").lower()
            period = item.get("period", "N/A")
            metric = item.get("metric", "N/A")
            segment = item.get("segment") or ""

            # Direction class and icon
            dir_class = direction if direction in {"up","down","unchanged"} else "unknown"
            icon = {"up": "📈", "down": "📉", "unchanged": "➡️"}.get(dir_class, "<span class='unknown-icon'>?</span>")

            # Render badges, suppress Metric badge when it's the generic 'guidance'
            badges_html = [
                f"<div class='badge {dir_class}'><span class='label'>Direction</span>{icon} {direction}</div>",
                f"<div class='badge'><span class='label'>Period</span>{period}</div>",
            ]
            if segment:
                badges_html.append(f"<div class='badge'><span class='label'>Segment</span>{segment}</div>")
            if item_type != "guidance_generic" and (metric or "").lower() != "guidance":
                badges_html.insert(0, f"<div class='badge'><span class='label'>Metric</span>{metric}</div>")
            st.markdown(
                "<div class='badges'>" + "".join(badges_html) + "</div>",
                unsafe_allow_html=True,
            )

            # Per-item correctness checkbox (kept interactive)
            item_correct = st.checkbox("✓ Correct", value=True, key=f"correct_{item_key}")

            # Structured value rendering
            # For generic 'guidance' metric, avoid showing structured value boxes
            pv = _parse_value_str(item.get("value_str", "")) if item_type != "guidance_generic" and (metric or "").lower() != "guidance" else {"present": False}
            if pv.get("present"):
                if pv.get("has_prior"):
                    # Show prior and current side-by-side
                    st.markdown("<div class='value-grid'>", unsafe_allow_html=True)
                    pr = pv["prior"]
                    cr = pv["current"]
                    # Prior box
                    if pr.get("is_range"):
                        prior_val = f"{pr['low']}–{pr['high']}{pr.get('unit','')}"
                    else:
                        prior_val = f"{pr.get('value','')}{pr.get('unit','')}"
                    st.markdown(f"<div class='vbox'><span class='label'>Prior</span><div class='val'>{prior_val}</div></div>", unsafe_allow_html=True)
                    # Current box
                    if cr.get("is_range"):
                        current_val = f"{cr['low']}–{cr['high']}{cr.get('unit','')}"
                    else:
                        current_val = f"{cr.get('value','')}{cr.get('unit','')}"
                    st.markdown(f"<div class='vbox'><span class='label'>Current</span><div class='val'>{current_val}</div></div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    if pv.get("change"):
                        st.caption(f"Change: {pv['change']}")
                else:
                    cr = pv["current"]
                    if cr.get("is_range"):
                        st.markdown("<div class='value-grid'>", unsafe_allow_html=True)
                        st.markdown(f"<div class='vbox'><span class='label'>Low</span><div class='val'>{cr['low']}{cr.get('unit','')}</div></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='vbox'><span class='label'>High</span><div class='val'>{cr['high']}{cr.get('unit','')}</div></div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='value-box'><strong>Value:</strong> {cr.get('value','')}{cr.get('unit','')}</div>", unsafe_allow_html=True)
                    if pv.get("change"):
                        st.caption(f"Change: {pv['change']}")

            # Text excerpt
            if item.get("text_snippet"):
                st.markdown("**Text excerpt:**")
                st.markdown(f"<div class='excerpt'>{item['text_snippet']}</div>", unsafe_allow_html=True)

            # Corrections UI if marked incorrect
            if not item_correct:
                st.warning("**Corrections needed:**")
                
                # Item type selector
                corrected_item_type = st.radio(
                    "Item Type",
                    ["financial_metric", "guidance_generic"],
                    index=0 if item_type == "financial_metric" else 1,
                    key=f"corr_item_type_{item_key}",
                    horizontal=True,
                    help="Financial metric: specific metric with numbers. Generic guidance: vague outlook without numbers."
                )
                
                corr_col1, corr_col2 = st.columns(2)

                with corr_col1:
                    # Only show metric dropdown for financial_metric type
                    if corrected_item_type == "financial_metric":
                        # Import FIN_METRICS to extract all unique metric names
                        try:
                            from event_feed_app.events.guidance_change.plugin import FIN_METRICS
                            # Extract unique metric names (first element of each tuple)
                            fin_metric_set = sorted(set(m[0] for m in FIN_METRICS.values()))
                        except:
                            fin_metric_set = []
                        
                        # Build metric options: FIN_METRICS + common additions + fallback
                        metric_options = fin_metric_set + ["eps", "capex", "cash_flow", "other"]
                        # Deduplicate while preserving order
                        seen = set()
                        metric_options = [m for m in metric_options if not (m in seen or seen.add(m))]
                        
                        default_metric = item.get("metric", "other")
                        if default_metric and default_metric not in metric_options:
                            metric_options.insert(0, default_metric)
                        
                        corrected_metric = st.selectbox(
                            "Correct Metric",
                            metric_options,
                            index=metric_options.index(default_metric) if default_metric in metric_options else 0,
                            key=f"corr_metric_{item_key}"
                        )
                    else:
                        corrected_metric = "guidance"
                        st.info("Generic guidance has no specific metric")
                    
                    corrected_segment = st.text_input(
                        "Segment / Business Unit",
                        value=segment,
                        key=f"corr_segment_{item_key}"
                    )
                    corrected_period = st.text_input(
                        "Correct Period",
                        value=item.get("period", ""),
                        key=f"corr_period_{item_key}"
                    )

                with corr_col2:
                    direction_options = ["raise", "lower", "maintain", "initiate", "withdraw", "unknown"]
                    default_direction = item.get("direction", "raise")
                    if default_direction not in direction_options:
                        direction_options.append(default_direction)
                    corrected_direction = st.selectbox(
                        "Correct Direction",
                        direction_options,
                        index=direction_options.index(default_direction),
                        key=f"corr_direction_{item_key}"
                    )
                    # Structured value inputs only for financial_metric type
                    if corrected_item_type == "financial_metric":
                        parsed_now = _parse_value_str(item.get("value_str", ""))
                        if parsed_now.get("present") and not parsed_now.get("has_prior") and parsed_now.get("current", {}).get("is_range"):
                            corrected_low = st.text_input(
                                "Correct Low",
                                value=str(parsed_now["current"].get("low", "")),
                                key=f"corr_value_low_{item_key}"
                            )
                            corrected_high = st.text_input(
                                "Correct High",
                                value=str(parsed_now["current"].get("high", "")),
                                key=f"corr_value_high_{item_key}"
                            )
                            corrected_value = f"{corrected_low}-{corrected_high}{parsed_now['current'].get('unit','')}"
                        else:
                            corrected_value = st.text_input(
                                "Correct Value",
                                value=item.get("value_str", ""),
                                key=f"corr_value_{item_key}"
                            )
                    else:
                        corrected_value = ""
                        st.info("Generic guidance has no numeric value")

                st.session_state.item_feedback[item_key] = {
                    "item_index": idx,
                    "is_correct": False,
                    "detected": item,
                    "corrections": {
                        "item_type": corrected_item_type,
                        "metric": corrected_metric,
                        "direction": corrected_direction,
                        "period": corrected_period,
                        "value_str": corrected_value,
                        "segment": corrected_segment,
                    },
                }
            else:
                st.session_state.item_feedback[item_key] = {
                    "item_index": idx,
                    "is_correct": True,
                    "detected": item,
                }

            if item.get("confidence"):
                st.caption(f"Confidence: {item['confidence']*100:.0f}%")

            # Close card wrapper
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No detailed guidance items available in alert payload")

# Section for adding missing items
st.markdown("---")
st.subheader("➕ Missing Detections")
st.caption("If the system missed any guidance items, add them here")

# Button to add new missing item
if st.button("+ Add Missing Item"):
    st.session_state.missing_items.append({
        "metric": "revenue",
        "direction": "raise",
        "period": "",
        "value_str": "",
        "location_hint": ""
    })

# Display forms for each missing item
for missing_idx, missing_item in enumerate(st.session_state.missing_items):
    with st.expander(f"Missing Item #{missing_idx + 1}", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            metric = st.selectbox(
                "Metric",
                ["revenue", "eps", "ebit", "ebitda", "margin", "capex", "cash_flow", "guidance", "other"],
                index=["revenue", "eps", "ebit", "ebitda", "margin", "capex", "cash_flow", "guidance", "other"].index(missing_item.get("metric", "revenue")),
                key=f"missing_metric_{missing_idx}"
            )
            period = st.text_input(
                "Period (e.g., Q4_2025, FY_2025)",
                value=missing_item.get("period", ""),
                key=f"missing_period_{missing_idx}"
            )
        
        with col2:
            direction = st.selectbox(
                "Direction",
                ["raise", "lower", "maintain", "initiate", "withdraw"],
                index=["raise", "lower", "maintain", "initiate", "withdraw"].index(missing_item.get("direction", "raise")),
                key=f"missing_direction_{missing_idx}"
            )
            value_str = st.text_input(
                "Value (e.g., $100M → $120M)",
                value=missing_item.get("value_str", ""),
                key=f"missing_value_{missing_idx}"
            )
        
        location_hint = st.text_input(
            "Location in text (optional)",
            value=missing_item.get("location_hint", ""),
            placeholder="e.g., 'paragraph 3', 'near end of document'",
            key=f"missing_location_{missing_idx}"
        )
        
        # Update the missing item in session state
        st.session_state.missing_items[missing_idx] = {
            "metric": metric,
            "direction": direction,
            "period": period,
            "value_str": value_str,
            "location_hint": location_hint
        }
        
        # Remove button
        if st.button("🗑️ Remove", key=f"remove_missing_{missing_idx}"):
            st.session_state.missing_items.pop(missing_idx)
            st.rerun()

# Show overall alert metadata
with st.expander("📊 Alert Metadata"):
    st.json({
        "alert_id": alert.get("alert_id"),
        "company_name": alert.get("company_name"),
        "significance_score": alert.get("significance_score"),
        "detected_at": alert.get("detected_at"),
        "guidance_count": len(guidance_items)
    })

st.markdown("---")
st.subheader("🎯 Was This Detection Correct?")

# Initialize FeedbackStore
feedback_store = FeedbackStore()

# Check for existing feedback
existing_feedback = feedback_store.get_feedback_for_signal(
    signal_type=signal_type,
    alert_id=alert_id
)

if existing_feedback:
    st.success(f"✅ You already submitted feedback for this alert")
    for fb in existing_feedback:
        feedback_label = "✅ Correct" if fb["is_correct"] else "❌ Incorrect"
        st.info(f"{feedback_label} - {fb['feedback_type']} - {fb['submitted_at']}")
        if fb.get("notes"):
            st.caption(f"Note: {fb['notes']}")
        if fb.get("metadata"):
            with st.expander("📋 Feedback Details"):
                st.json(fb["metadata"])
else:
    # Quick feedback buttons
    st.markdown("**Quick Rating:**")
    col1, col2 = st.columns(2)
    
    # Use session state to track if detailed form should be shown
    if "show_detailed_form" not in st.session_state:
        st.session_state.show_detailed_form = False
    
    # Adjust button labels and feedback semantics based on review type
    if is_false_negative_review:
        # For negative reviews: user is checking if system SHOULD have flagged it
        correct_label = "✅ Correct (No Guidance)"
        incorrect_label = "❌ Missed Guidance"
        correct_help = "System was right NOT to flag this PR"
        incorrect_help = "System missed guidance that was present"
    else:
        # For positive alerts: user is checking if system WAS RIGHT to flag it
        correct_label = "✅ Correct Alert"
        incorrect_label = "❌ Incorrect Alert"
        correct_help = "System correctly detected guidance"
        incorrect_help = "System incorrectly flagged this PR"
    
    with col1:
        if st.button(correct_label, use_container_width=True, type="primary", help=correct_help):
            user_id = params.get("user_id", "anonymous")
            
            # Build extraction feedback metadata
            extraction_feedback = {
                "items": list(st.session_state.item_feedback.values()),
                "missing_items": st.session_state.missing_items,
                "overall_correct": all(item.get("is_correct", True) for item in st.session_state.item_feedback.values()) and len(st.session_state.missing_items) == 0
            }
            
            try:
                feedback_id = feedback_store.save_feedback(
                    signal_type=signal_type,
                    alert_id=alert_id,
                    press_release_id=press_release_id,
                    user_id=user_id,
                    is_correct=True,
                    guidance_metadata={"extraction_feedback": extraction_feedback}
                )
                st.success(f"✅ Thank you! Feedback saved: {feedback_id}")
                # Clear session state
                st.session_state.item_feedback = {}
                st.session_state.missing_items = []
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save feedback: {e}")
    
    with col2:
        if st.button(incorrect_label, use_container_width=True, help=incorrect_help):
            st.session_state.show_detailed_form = True
    
    # Detailed feedback form (shown when "Incorrect" is clicked)
    if st.session_state.show_detailed_form:
            st.markdown("---")
            
            if is_false_negative_review:
                st.markdown("**📝 What guidance did the system miss?**")
            else:
                st.markdown("**📝 Please provide details about the incorrect alert:**")
            
            with st.form("detailed_feedback_form"):
                # Issue type - different options for negative reviews vs positive alerts
                if is_false_negative_review:
                    issue_type = st.selectbox(
                        "Type of missed guidance:",
                        [
                            "false_negative",
                            "missed_revenue",
                            "missed_eps",
                            "missed_margin",
                            "missed_other_metric"
                        ],
                        help="What type of guidance was missed?"
                    )
                else:
                    issue_type = st.selectbox(
                        "What was wrong?",
                        [
                            "false_positive",
                            "wrong_metric",
                            "wrong_direction",
                            "wrong_period",
                            "wrong_magnitude",
                            "missed_context"
                        ],
                        help="Select the type of error"
                    )
                
                # For guidance_change specific fields
                correct_metric = []
                correct_direction = ""
                correct_period = ""
                guidance_type = ""
                
                if signal_type == "guidance_change":
                    st.markdown("**Guidance Details (optional):**")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        correct_metric = st.multiselect(
                            "Correct metric(s)",
                            ["revenue", "eps", "ebit", "ebitda", "margin", "capex", "cash_flow", "other"],
                            help="What metrics were actually mentioned?"
                        )
                    
                    with col_b:
                        correct_direction = st.selectbox(
                            "Correct direction",
                            ["", "up", "down", "unchanged", "mixed"],
                            help="What was the actual guidance direction?"
                        )
                    
                    correct_period = st.text_input(
                        "Correct period",
                        placeholder="e.g., Q4_2025, FY_2025",
                        help="What period was the guidance for?"
                    )
                    
                    guidance_type = st.selectbox(
                        "Guidance type",
                        ["", "initiate", "update", "raise", "lower", "reaffirm", "withdraw"],
                        help="What type of guidance change was it?"
                    )
                
                # General notes
                notes = st.text_area(
                    "Additional notes",
                    placeholder="Any other details that would help improve the signal detection...",
                    height=100
                )
                
                submitted = st.form_submit_button("Submit Feedback", use_container_width=True)
                
                if submitted:
                    user_id = params.get("user_id", "anonymous")
                    
                    # Build metadata including extraction feedback
                    extraction_feedback = {
                        "items": list(st.session_state.item_feedback.values()),
                        "missing_items": st.session_state.missing_items,
                        "overall_correct": all(item.get("is_correct", True) for item in st.session_state.item_feedback.values()) and len(st.session_state.missing_items) == 0
                    }
                    
                    guidance_metadata: dict[str, Any] = {
                        "issue_type": issue_type,
                        "extraction_feedback": extraction_feedback
                    }
                    
                    if signal_type == "guidance_change":
                        if correct_metric:
                            guidance_metadata["correct_metrics"] = correct_metric
                        if correct_direction:
                            guidance_metadata["correct_direction"] = correct_direction
                        if correct_period:
                            guidance_metadata["correct_period"] = correct_period
                        if guidance_type:
                            guidance_metadata["guidance_type"] = guidance_type
                    
                    try:
                        feedback_id = feedback_store.save_feedback(
                            signal_type=signal_type,
                            alert_id=alert_id,
                            press_release_id=press_release_id,
                            user_id=user_id,
                            is_correct=False,
                            guidance_metadata=guidance_metadata,
                            notes=notes
                        )
                        st.success(f"✅ Thank you for the detailed feedback! Saved: {feedback_id}")
                        # Clear session state
                        st.session_state.show_detailed_form = False
                        st.session_state.item_feedback = {}
                        st.session_state.missing_items = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to save feedback: {e}")

# Metadata expander
with st.expander("📊 Document Metadata"):
    st.json({
        "press_release_id": press_release_id,
        "alert_id": alert_id if alert_id else "N/A",
        "company_name": company,
        "release_date": release_date,
        "source_url": source_url,
        "category": category,
        "title_length": len(title),
        "body_length": len(body),
    })

# Footer with copy ID button
st.markdown("---")
st.caption(f"Press Release ID: `{press_release_id}` · Viewer build: {BUILD_STAMP}")
