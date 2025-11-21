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

with st.spinner("Loading alert details..."):
    alert = load_alert_payload(alert_id, signal_type)

if not alert:
    st.error(f"❌ Alert not found: `{alert_id}`")
    st.info("The alert may have expired or the ID is incorrect.")
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
    # Split into paragraphs for better readability
    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    for para in paragraphs:
        st.markdown(para)
        st.markdown("")  # Add spacing
else:
    st.warning("No content available for this press release.")

# Feedback section
st.markdown("---")

st.subheader("🔍 What the System Detected")

# Display detected guidance items from alert
guidance_items = alert.get("guidance_items", [])

if guidance_items:
    for idx, item in enumerate(guidance_items, 1):
        with st.expander(f"Detection #{idx}: {item.get('metric', 'Unknown').upper()}", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Metric", item.get("metric", "N/A"))
            with col_b:
                direction = item.get("direction", "N/A")
                direction_emoji = {"up": "📈", "down": "📉", "unchanged": "➡️"}.get(direction, "❓")
                st.metric("Direction", f"{direction_emoji} {direction}")
            with col_c:
                st.metric("Period", item.get("period", "N/A"))
            
            if item.get("value_str"):
                st.info(f"**Value:** {item['value_str']}")
            
            if item.get("text_snippet"):
                st.markdown(f"**Text excerpt:**")
                st.code(item["text_snippet"], language=None)
            
            if item.get("confidence"):
                st.caption(f"Confidence: {item['confidence']*100:.0f}%")
else:
    st.info("No detailed guidance items available in alert payload")

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
    
    with col1:
        if st.button("✅ Correct Alert", use_container_width=True, type="primary"):
            user_id = params.get("user_id", "anonymous")
            
            try:
                feedback_id = feedback_store.save_feedback(
                    signal_type=signal_type,
                    alert_id=alert_id,
                    press_release_id=press_release_id,
                    user_id=user_id,
                    is_correct=True
                )
                st.success(f"✅ Thank you! Feedback saved: {feedback_id}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save feedback: {e}")
    
    with col2:
        if st.button("❌ Incorrect Alert", use_container_width=True):
            st.session_state.show_detailed_form = True
    
    # Detailed feedback form (shown when "Incorrect" is clicked)
    if st.session_state.show_detailed_form:
            st.markdown("---")
            st.markdown("**📝 Please provide details about the incorrect alert:**")
            
            with st.form("detailed_feedback_form"):
                # Issue type
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
                    
                    # Build metadata
                    guidance_metadata: dict[str, Any] = {
                        "issue_type": issue_type
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
                        st.session_state.show_detailed_form = False
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
st.caption(f"Press Release ID: `{press_release_id}`")
