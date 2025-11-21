#!/usr/bin/env python3
"""
Telegram bot webhook handler for false negative feedback.

Processes user replies to daily review messages and saves feedback to GCS
using the same FeedbackStore as positive feedback.

Setup:
1. Set TELEGRAM_BOT_TOKEN in .env
2. Run as webhook: python -m event_feed_app.alerts.review.telegram_handler
3. Or run polling mode for testing: python -m event_feed_app.alerts.review.telegram_handler --poll

User commands:
- Reply to PR message with 👍 or /correct = NOT guidance (true negative)
- Reply to PR message with 👎 or /missed = IS guidance (false negative)
- Optional: Add text after command for notes
  Example: "/missed They raised Q4 revenue outlook to $500M"
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from event_feed_app.alerts.feedback_store import FeedbackStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_review_tracking(date_str: str = None) -> Dict[str, Any]:
    """
    Load the review tracking file to map message_id -> press_release_id.
    
    Args:
        date_str: Date in YYYYMMDD format, defaults to today
        
    Returns:
        Dict with message tracking data
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y%m%d")
    
    tracking_file = Path(f"data/labeling/daily_review_{date_str}.json")
    
    if not tracking_file.exists():
        logger.warning(f"No tracking file found: {tracking_file}")
        return {}
    
    with open(tracking_file, "r") as f:
        return json.load(f)


def process_feedback_reply(message: Dict[str, Any]) -> Optional[str]:
    """
    Process a user reply to a daily review message.
    
    Args:
        message: Telegram message object
        
    Returns:
        Response message to send back to user, or None if not a feedback reply
    """
    # Check if this is a reply to a bot message
    if "reply_to_message" not in message:
        return None
    
    reply_to = message["reply_to_message"]
    replied_message_id = reply_to["message_id"]
    
    # Get user's text
    text = message.get("text", "").strip()
    
    # Parse command and notes
    is_missed = False
    is_correct = False
    notes = ""
    
    if text.startswith("👎") or text.lower().startswith("/missed"):
        is_missed = True
        # Extract notes (everything after the command)
        if text.startswith("👎"):
            notes = text[1:].strip()
        else:
            parts = text.split(maxsplit=1)
            notes = parts[1] if len(parts) > 1 else ""
    elif text.startswith("👍") or text.lower().startswith("/correct"):
        is_correct = True
        if text.startswith("👍"):
            notes = text[1:].strip()
        else:
            parts = text.split(maxsplit=1)
            notes = parts[1] if len(parts) > 1 else ""
    else:
        # Not a feedback command
        return None
    
    # Load tracking data to find which PR this message corresponds to
    tracking = load_review_tracking()
    
    if not tracking or "messages" not in tracking:
        return "❌ Could not find review tracking data. Feedback not saved."
    
    # Find the PR for this message_id
    pr_data = None
    for msg in tracking["messages"]:
        if msg["message_id"] == replied_message_id:
            pr_data = msg
            break
    
    if not pr_data:
        return "❌ Could not find PR data for this message. Maybe review is too old?"
    
    # Save feedback using FeedbackStore (same as positive feedback)
    try:
        feedback_store = FeedbackStore()
        
        # For false negatives, we don't have an alert_id (it was never created)
        # Use press_release_id as the identifier
        alert_id = f"fn_{pr_data['press_release_id']}"
        
        if is_missed:
            # User says this IS guidance (false negative)
            feedback_id = feedback_store.save_feedback(
                signal_type="guidance_change",
                alert_id=alert_id,
                press_release_id=pr_data["press_release_id"],
                user_id=message["from"]["id"],
                is_correct=False,  # The system was WRONG (missed it)
                guidance_metadata={"issue_type": "false_negative", "review_type": "daily_fn_review"},
                notes=notes or "False negative - should have been flagged"
            )
            response = f"✅ Feedback saved: {feedback_id}\n"
            response += f"📌 Marked as FALSE NEGATIVE (should have been flagged)\n"
            response += f"Company: {pr_data['company']}"
            if notes:
                response += f"\nNote: {notes}"
            return response
        
        elif is_correct:
            # User says this is NOT guidance (true negative)
            feedback_id = feedback_store.save_feedback(
                signal_type="guidance_change",
                alert_id=alert_id,
                press_release_id=pr_data["press_release_id"],
                user_id=message["from"]["id"],
                is_correct=True,  # The system was RIGHT (correctly ignored)
                guidance_metadata={"issue_type": "true_negative", "review_type": "daily_fn_review"},
                notes=notes or "True negative - correctly not flagged"
            )
            response = f"✅ Feedback saved: {feedback_id}\n"
            response += f"📌 Marked as TRUE NEGATIVE (correctly ignored)\n"
            response += f"Company: {pr_data['company']}"
            if notes:
                response += f"\nNote: {notes}"
            return response
        
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}", exc_info=True)
        return f"❌ Error saving feedback: {str(e)}"
    
    return None


def send_telegram_message(chat_id: str, text: str, reply_to_message_id: int = None):
    """Send a message via Telegram Bot API."""
    import requests
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    
    if reply_to_message_id:
        payload["reply_to_message_id"] = reply_to_message_id
    
    response = requests.post(url, json=payload)
    if not response.ok:
        logger.error(f"Failed to send message: {response.text}")


def handle_update(update: Dict[str, Any]):
    """Process a single Telegram update."""
    if "message" not in update:
        return
    
    message = update["message"]
    chat_id = message["chat"]["id"]
    
    # Process feedback reply
    response_text = process_feedback_reply(message)
    
    if response_text:
        send_telegram_message(
            chat_id=str(chat_id),
            text=response_text,
            reply_to_message_id=message["message_id"]
        )


def run_polling():
    """Run bot in polling mode (for testing)."""
    import requests
    import time
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    offset = 0
    
    logger.info("Starting Telegram bot in polling mode...")
    logger.info("Send test messages to the bot!")
    
    while True:
        try:
            response = requests.get(url, params={"offset": offset, "timeout": 30})
            
            if not response.ok:
                logger.error(f"Error getting updates: {response.text}")
                time.sleep(5)
                continue
            
            data = response.json()
            
            if not data["ok"]:
                logger.error(f"Telegram API error: {data}")
                time.sleep(5)
                continue
            
            for update in data["result"]:
                handle_update(update)
                offset = update["update_id"] + 1
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Stopping bot...")
            break
        except Exception as e:
            logger.error(f"Error in polling loop: {e}", exc_info=True)
            time.sleep(5)


def run_webhook():
    """Run bot as webhook (for production)."""
    from flask import Flask, request
    
    app = Flask(__name__)
    
    @app.route("/webhook", methods=["POST"])
    def webhook():
        """Handle incoming Telegram webhook."""
        update = request.get_json()
        handle_update(update)
        return "OK"
    
    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return "OK"
    
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting Telegram webhook server on port {port}")
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
def main():
    """Entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Telegram feedback handler")
    parser.add_argument("--poll", action="store_true", help="Run in polling mode instead of webhook")
    
    args = parser.parse_args()
    
    if args.poll:
        run_polling()
    else:
        run_webhook()


if __name__ == "__main__":
    main()
