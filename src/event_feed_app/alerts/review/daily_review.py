#!/usr/bin/env python3
"""
Daily false negative review via Telegram.

Sends 10 random PRs that were NOT flagged as guidance for manual review.
User can reply with feedback if any contain guidance that was missed.
"""
import pandas as pd
from datetime import datetime, timedelta
import random
import logging
import os
import requests

from event_feed_app.alerts.store import AlertStore
from event_feed_app.utils.gcs_io import load_parquet_df_date_range

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_telegram_message(message: str, parse_mode: str = "Markdown"):
    """Send message via Telegram Bot API."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Telegram message sent successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def send_daily_false_negative_review(
    sample_size: int = 10,
    days_back: int = 1,
    focus_category: str = None
):
    """
    Send daily sample of non-flagged PRs to Telegram for review.
    
    Args:
        sample_size: Number of PRs to send
        days_back: How many days back to sample from
        focus_category: Category to focus on (optional, but categories aren't populated regularly)
    """
    logger.info("Starting daily false negative review...")
    
    try:
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Loading PRs from {start_date} to {end_date}")
        
        # Load recent PRs from GCS using the same utility as alert runner
        uri = "gs://event-feed-app-data/silver_normalized/table=press_releases/"
        df = load_parquet_df_date_range(
            uri=uri,
            start_date=str(start_date),
            end_date=str(end_date),
            date_column="release_date",
            columns=["press_release_id", "company_name", "title", "full_text", "release_date", "category", "source_url", "source"]
        )
        
        logger.info(f"Loaded {len(df)} total PRs")
        
        # Optionally filter by category (but categories are rarely populated)
        if focus_category:
            category_df = df[df['category'] == focus_category]
            logger.info(f"Filtered to {len(category_df)} {focus_category} PRs")
            if len(category_df) == 0:
                logger.warning(f"No PRs found with category '{focus_category}', using all PRs instead")
                category_df = df
        else:
            category_df = df
            logger.info(f"Using all PRs (no category filter)")
        
        # Get PRs that WERE flagged (from alerts DB)
        try:
            store = AlertStore()
            
            import sqlite3
            with sqlite3.connect(store.db_path) as conn:
                flagged_df = pd.read_sql(
                    """
                    SELECT DISTINCT press_release_id 
                    FROM alerts 
                    WHERE alert_type='guidance_change'
                    AND detected_at >= datetime('now', ? || ' days')
                    """,
                    conn,
                    params=(-days_back,)
                )
            flagged_ids = set(flagged_df['press_release_id'])
            logger.info(f"Found {len(flagged_ids)} PRs that were flagged in last {days_back} days")
        except Exception as e:
            logger.warning(f"Could not load flagged PRs: {e}")
            flagged_ids = set()
        
        # Get NON-flagged PRs
        non_flagged = category_df[~category_df['press_release_id'].isin(flagged_ids)]
        logger.info(f"Non-flagged PRs: {len(non_flagged)}")
        
        # Exclude PRs that already have feedback
        try:
            from event_feed_app.alerts.feedback_store import FeedbackStore
            feedback_store = FeedbackStore()
            
            # Get all feedback for guidance_change
            all_feedback = feedback_store.get_feedback_for_signal(signal_type="guidance_change")
            
            # Extract press_release_ids that have feedback
            # (both from false negative reviews and positive alerts)
            reviewed_ids = set()
            for fb in all_feedback:
                pr_id = fb.get("press_release_id")
                if pr_id:
                    reviewed_ids.add(pr_id)
            
            logger.info(f"Found {len(reviewed_ids)} PRs with existing feedback")
            
            # Filter out already reviewed PRs
            non_flagged = non_flagged[~non_flagged['press_release_id'].isin(reviewed_ids)]
            logger.info(f"Non-flagged PRs without feedback: {len(non_flagged)}")
            
        except Exception as e:
            logger.warning(f"Could not load existing feedback: {e}")
        
        if len(non_flagged) == 0:
            logger.info("No non-flagged PRs found for review (all already reviewed)")
            return
        
        # Sample randomly
        sample = non_flagged.sample(n=min(sample_size, len(non_flagged)), random_state=random.randint(0, 10000))
        
        # Send each PR individually so users can reply to specific messages
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not bot_token or not chat_id:
            logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
            return
        
        # Get the viewer URL base (same env var as positive alerts)
        viewer_url_base = os.getenv("VIEWER_BASE_URL", "http://localhost:8501")
        
        # Send header first
        header = f"📊 *Daily False Negative Review* - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        header += f"❓ *Did we miss any guidance?*\n"
        header += f"Review {len(sample)} PRs that were NOT flagged for guidance.\n"
        header += f"Click each link to view the full PR and provide feedback.\n"
        send_telegram_message(header, parse_mode="Markdown")
        
        # Send each PR as a link to the viewer (same as positive alerts)
        import requests
        from urllib.parse import urlencode
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        def _escape_md(text: str) -> str:
            """Escape special characters for Telegram Markdown."""
            if not isinstance(text, str):
                return ""
            return (
                text.replace("_", "\\_")
                    .replace("*", "\\*")
                    .replace("`", "\\`")
                    .replace("[", "\\[")
            )
        
        from event_feed_app.alerts.alert_payload_store import AlertPayloadStore
        payload_store = AlertPayloadStore()
        
        for idx, row in sample.iterrows():
            pr_id = row['press_release_id']
            company = row.get('company_name', 'Unknown')
            title = row['title'][:100]  # Truncate for readability
            date = row['release_date']
            
            # Create a fake alert_id for the viewer (we use fn_ prefix for false negative reviews)
            alert_id = f"fn_{pr_id}"
            
            # --- Save alert payload for this negative review ---
            alert_payload = {
                "alert_id": alert_id,
                "alert_type": "guidance_change",
                "press_release_id": pr_id,
                "company_name": company,
                "detected_at": datetime.now().isoformat(),
                "guidance_items": [],  # No detections for negatives
                "metadata": {
                    "title": row.get("title", ""),
                    "release_date": date,
                    "body": row.get("full_text", ""),
                    "press_release_url": row.get("source_url", ""),
                    "category": row.get("category", ""),
                    "source": row.get("source", "")
                }
            }
            try:
                payload_store.save_alert_payload(alert_payload)
                logger.info(f"Saved alert payload for negative review: {alert_id}")
            except Exception as e:
                logger.warning(f"Failed to save alert payload for {alert_id}: {e}")
            
            # Build viewer URL (same format as positive alerts)
            query_params = {
                "id": pr_id,
                "alert_id": alert_id,
                "signal_type": "guidance_change"
            }
            query_string = urlencode(query_params)
            viewer_url = f"{viewer_url_base}?{query_string}"
            safe_url = _escape_md(viewer_url)
            
            # Send message with plain URL (Telegram auto-converts to clickable link)
            # Same format as positive alerts
            pr_msg = f"*{company}* ({date})\n{title}\n\n"
            pr_msg += f"📱 View & Give Feedback:\n{safe_url}"
            
            payload = {
                "chat_id": chat_id,
                "text": pr_msg,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            response = requests.post(url, json=payload)
            
            if response.ok:
                logger.info(f"Sent PR {pr_id} with viewer URL: {viewer_url}")
        
        # Send footer
        footer = (
            "\n━━━━━━━━━━━━━━━━━━━━\n"
            "💡 Click each link above to:\n"
            "• Read the full press release\n"
            "• Mark as correct (not guidance) or incorrect (missed guidance)\n"
            "• Add optional notes about what was missed\n\n"
            "Your feedback trains the model!"
        )
        send_telegram_message(footer, parse_mode="Markdown")
        logger.info(f"Sent {len(sample)} PRs for review via PR viewer")
        
        # Save list of reviewed PRs for tracking (don't need message_ids anymore)
        # The feedback happens in the PR viewer now, not via Telegram replies
        output_file = f"data/labeling/daily_review_{datetime.now().strftime('%Y%m%d')}.json"
        os.makedirs("data/labeling", exist_ok=True)
        import json
        with open(output_file, "w") as f:
            json.dump({
                "date": datetime.now().isoformat(),
                "category": focus_category,
                "review_type": "false_negative_check",
                "count": len(sample),
                "press_releases": [
                    {
                        "press_release_id": row["press_release_id"],
                        "company": row.get("company_name", "Unknown"),
                        "title": row["title"],
                        "date": row["release_date"]
                    }
                    for _, row in sample.iterrows()
                ]
            }, f, indent=2)
        logger.info(f"Saved review tracking to {output_file}")
        logger.info("Note: User feedback will be saved to GCS via PR viewer (same as positive feedback)")
        
    except Exception as e:
        logger.error(f"Error in daily review: {e}", exc_info=True)
        error_msg = f"⚠️ Daily false negative review failed:\n`{str(e)}`"
        try:
            send_telegram_message(error_msg, parse_mode="Markdown")
        except:
            pass


def main():
    """Entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Send daily false negative review via Telegram")
    parser.add_argument("--size", type=int, default=10, help="Number of PRs to sample")
    parser.add_argument("--days", type=int, default=10, help="Days back to sample from")
    parser.add_argument("--category", type=str, default=None, help="Category to focus on (optional, rarely populated)")
    
    args = parser.parse_args()
    
    send_daily_false_negative_review(
        sample_size=args.size,
        days_back=args.days,
        focus_category=args.category
    )


if __name__ == "__main__":
    main()
