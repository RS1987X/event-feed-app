#!/usr/bin/env python3
"""
Daily false negative review via Telegram.

Sends 10 random PRs that were NOT flagged as guidance for manual review.
User can reply with feedback if any contain guidance that was missed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import gcsfs
from datetime import datetime, timedelta
import random
import logging
import os
import requests

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
    focus_category: str = "earnings"
):
    """
    Send daily sample of non-flagged PRs to Telegram for review.
    
    Args:
        sample_size: Number of PRs to send
        days_back: How many days back to sample from
        focus_category: Category to focus on (earnings most likely to have guidance)
    """
    logger.info("Starting daily false negative review...")
    
    try:
        fs = gcsfs.GCSFileSystem()
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Loading PRs from {start_date} to {end_date}")
        
        # Load recent PRs
        gcs_path = "gs://event-feed-app-data/silver_normalized/table=press_releases/**/*.parquet"
        df = pd.read_parquet(
            gcs_path,
            filesystem=fs
        )
        
        # Filter by date
        df = df[
            (pd.to_datetime(df['release_date']) >= pd.Timestamp(start_date)) &
            (pd.to_datetime(df['release_date']) <= pd.Timestamp(end_date))
        ]
        
        logger.info(f"Loaded {len(df)} total PRs")
        
        # Filter to focus category
        if focus_category:
            category_df = df[df['category'] == focus_category]
            logger.info(f"Filtered to {len(category_df)} {focus_category} PRs")
        else:
            category_df = df
        
        # Get PRs that WERE flagged (from alerts DB)
        try:
            from event_feed_app.alerts.store import AlertStore
            store = AlertStore()
            
            import sqlite3
            with sqlite3.connect(store.db_path) as conn:
                flagged_df = pd.read_sql(
                    """
                    SELECT DISTINCT press_release_id 
                    FROM alerts 
                    WHERE signal_type='guidance_change'
                    AND detected_at >= datetime('now', '-2 days')
                    """,
                    conn
                )
            flagged_ids = set(flagged_df['press_release_id'])
            logger.info(f"Found {len(flagged_ids)} PRs that were flagged")
        except Exception as e:
            logger.warning(f"Could not load flagged PRs: {e}")
            flagged_ids = set()
        
        # Get NON-flagged PRs
        non_flagged = category_df[~category_df['press_release_id'].isin(flagged_ids)]
        logger.info(f"Non-flagged PRs: {len(non_flagged)}")
        
        if len(non_flagged) == 0:
            logger.info("No non-flagged PRs found for review")
            return
        
        # Sample randomly
        sample = non_flagged.sample(n=min(sample_size, len(non_flagged)), random_state=random.randint(0, 10000))
        
        # Format message
        message = f"📊 **Daily False Negative Review**\n"
        message += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        message += f"Category: {focus_category}\n"
        message += f"PRs reviewed: {len(sample)}\n\n"
        message += "❓ **Did we miss any guidance?**\n"
        message += "Review these PRs that were NOT flagged:\n\n"
        
        for idx, row in sample.iterrows():
            pr_id = row['press_release_id']
            company = row.get('company_name', 'Unknown')
            title = row['title'][:100]
            date = row['release_date']
            url = row.get('source_url', '')
            
            message += f"**{idx + 1}. {company}** ({date})\n"
            message += f"_{title}_\n"
            message += f"ID: `{pr_id}`\n"
            if url:
                message += f"[Read full PR]({url})\n"
            message += "\n"
        
        message += "━━━━━━━━━━━━━━━━━━━━\n"
        message += "💡 **To report a miss:**\n"
        message += "Reply with: `/missed <PR_ID> <reason>`\n"
        message += "Example: `/missed abc123 Contains Q4 revenue guidance`\n"
        
        # Send to Telegram
        send_telegram_message(message, parse_mode="Markdown")
        logger.info(f"Sent {len(sample)} PRs for review")
        
        # Save to local file for tracking
        output_file = f"data/labeling/daily_review_{datetime.now().strftime('%Y%m%d')}.csv"
        sample[['press_release_id', 'company_name', 'release_date', 'title', 'category', 'source_url']].to_csv(
            output_file, index=False
        )
        logger.info(f"Saved review list to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in daily review: {e}", exc_info=True)
        error_msg = f"⚠️ Daily false negative review failed:\n`{str(e)}`"
        try:
            send_telegram_message(error_msg, parse_mode="Markdown")
        except:
            pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Send daily false negative review via Telegram")
    parser.add_argument("--size", type=int, default=10, help="Number of PRs to sample")
    parser.add_argument("--days", type=int, default=1, help="Days back to sample from")
    parser.add_argument("--category", type=str, default="earnings", help="Category to focus on")
    
    args = parser.parse_args()
    
    send_daily_false_negative_review(
        sample_size=args.size,
        days_back=args.days,
        focus_category=args.category
    )
