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
alert_id = params.get("alert_id", "")  # Optional: for feedback tracking

if not press_release_id:
    st.title("📄 Press Release Viewer")
    st.info("To view a press release, add `?id=<press_release_id>` to the URL")
    st.markdown("**Example:** `http://localhost:8501/?id=1980c93f9f819a6d`")
    st.markdown("---")
    st.markdown("This viewer is used by the alert system when press releases don't have a source URL.")
    st.stop()

# Load data
@st.cache_data(ttl=3600)
def load_silver_data():
    cfg = Settings()
    return load_from_gcs(cfg.gcs_silver_root)

with st.spinner("Loading press release data..."):
    df = load_silver_data()

# Filter to requested ID
doc = df[df["press_release_id"] == press_release_id]

if doc.empty:
    st.error(f"❌ Press release not found: `{press_release_id}`")
    st.info("Check that the ID is correct and the data has been synced from GCS.")
    st.stop()

# Extract document fields
row = doc.iloc[0]
title = row.get("title_clean") or row.get("title") or "Untitled"
company = row.get("company_name") or "Unknown Company"
release_date = row.get("release_date") or "Unknown Date"
body = row.get("full_text_clean") or row.get("full_text") or ""
source_url = row.get("source_url") or row.get("url") or row.get("link") or ""
category = row.get("l1_class") or row.get("category") or ""

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

if source_url:
    st.markdown(f"**Original Source:** [{source_url}]({source_url})")

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

# Feedback section (only if alert_id is provided)
if alert_id:
    st.markdown("---")
    
    # Load alert payload to show what system detected
    payload_store = AlertPayloadStore()
    signal_type = params.get("signal_type", "guidance_change")
    
    alert_payload = payload_store.get_alert_payload(alert_id, signal_type)
    
    if alert_payload:
        st.subheader("🔍 What the System Detected")
        
        # Display detected guidance items
        guidance_items = alert_payload.get("guidance_items", [])
        
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
                "alert_id": alert_payload.get("alert_id"),
                "company_name": alert_payload.get("company_name"),
                "significance_score": alert_payload.get("significance_score"),
                "detected_at": alert_payload.get("detected_at"),
                "guidance_count": len(guidance_items)
            })
    else:
        st.warning("⚠️ Alert detection details not available (payload not found in GCS)")
    
    st.markdown("---")
    st.subheader("🎯 Was This Detection Correct?")
    
    # Initialize FeedbackStore
    feedback_store = FeedbackStore()
    signal_type = params.get("signal_type", "guidance_change")
    
    # Check for existing feedback
    existing_feedback = feedback_store.get_feedback_for_signal(
        signal_type=signal_type,
        signal_id=alert_id
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
                        signal_id=alert_id,
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
                    guidance_metadata = {
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
                            signal_id=alert_id,
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
