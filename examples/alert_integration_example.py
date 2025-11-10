# SPDX-License-Identifier: MIT
# examples/alert_integration_example.py
"""
Example demonstrating how to integrate alert system with the processing pipeline.
"""
from pathlib import Path
import sys
import os

# Add src and scripts to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

# Load environment variables from .env file
from load_env import load_env
load_env()

from event_feed_app.alerts.orchestration import AlertOrchestrator
from event_feed_app.alerts.store import AlertStore


def example_user_setup():
    """Set up example users with different preferences."""
    print("=" * 80)
    print("Setting Up Example Users")
    print("=" * 80)
    
    store = AlertStore()
    
    # User 1: Email only
    store.save_user_preferences(
        user_id="analyst@company.com",
        preferences={
            "channels": ["email"],
            "email": "analyst@company.com",
            "min_significance": 0.6,
        }
    )
    print("✓ User 1: analyst@company.com (email, threshold=0.6)")
    
    # User 2: Telegram only
    store.save_user_preferences(
        user_id="ceo@company.com",
        preferences={
            "channels": ["telegram"],
            "telegram_chat_id": "123456789",  # Get from telegram_get_chat_id.py
            "min_significance": 0.8,
        }
    )
    print("✓ User 2: ceo@company.com (telegram, threshold=0.8)")
    
    # User 3: Both channels
    store.save_user_preferences(
        user_id="manager@company.com",
        preferences={
            "channels": ["email", "telegram"],
            "email": "manager@company.com",
            "telegram_chat_id": "987654321",
            "min_significance": 0.5,
        }
    )
    print("✓ User 3: manager@company.com (email+telegram, threshold=0.5)")
    print()


def example_batch_processing():
    """
    Example: Process a batch of documents and generate alerts.
    
    This would typically be called after the main orchestrator pipeline
    has classified documents.
    """
    print("=" * 80)
    print("Example 1: Batch Alert Processing")
    print("=" * 80)
    
    # Initialize alert system with Telegram support
    # Credentials are loaded from .env file automatically
    config = {
        "smtp": {
            "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
            "port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME"),
            "password": os.getenv("SMTP_PASSWORD"),
            "from_address": os.getenv("SMTP_FROM_ADDRESS", "alerts@yourcompany.com"),
        },
        "telegram": {
            "bot_token": os.getenv("TELEGRAM_BOT_TOKEN")  # From .env file
        }
    }
    
    orchestrator = AlertOrchestrator(
        config=config,
        enable_document_dedup=False,  # Option A: trust plugin dedup
    )
    
    # Example documents (these would come from your pipeline)
    sample_docs = [
        {
            "press_release_id": "pr_001",
            "title_clean": "ACME Corp Raises Revenue Guidance for FY2025",
            "full_text_clean": """
                ACME Corporation (NYSE: ACME) today announced that it is raising its 
                revenue guidance for fiscal year 2025 to $500-550 million, up from 
                the previous guidance of $450-480 million. The company also reaffirmed 
                its operating margin guidance of approximately 20%.
                
                "We are pleased with our strong performance in Q3 and are confident 
                in achieving these updated targets," said CEO Jane Smith.
            """,
            "release_date": "2025-11-10",
            "company_name": "ACME Corporation",
            "source_url": "https://example.com/acme-guidance-update",
        },
        {
            "press_release_id": "pr_002",
            "title_clean": "XYZ Ltd Q3 2025 Financial Results",
            "full_text_clean": """
                XYZ Ltd reported Q3 2025 revenue of EUR 120 million, up 15% year-over-year.
                Net income was EUR 12 million. The company maintains its full-year 
                revenue guidance of EUR 480-500 million.
            """,
            "release_date": "2025-11-10",
            "company_name": "XYZ Ltd",
            "source_url": "https://example.com/xyz-q3-results",
        }
    ]
    
    # Process documents and generate alerts
    stats = orchestrator.process_documents(sample_docs)
    
    print(f"\nProcessing Results:")
    print(f"  Documents processed: {stats['docs_processed']}")
    print(f"  Alerts detected: {stats['alerts_detected']}")
    print(f"  Duplicates filtered: {stats['alerts_deduplicated']}")
    print(f"  Alerts delivered: {stats['alerts_delivered']}")
    print(f"  Delivery failures: {stats['delivery_failures']}")


def example_telegram_info():
    """Show Telegram setup information."""
    print("\n" + "=" * 80)
    print("Example 2: Telegram Setup (FREE!)")
    print("=" * 80)
    
    print("""
Telegram is FREE and easy to set up:

1. Create a bot:
   - Open Telegram, search for @BotFather
   - Send: /newbot
   - Follow prompts to name your bot
   - BotFather gives you a token like: 123456789:ABC-DEF...
   
2. Get user chat IDs:
   - Users send /start to your bot
   - Run: python scripts/telegram_get_chat_id.py YOUR_BOT_TOKEN
   - Script shows all chat IDs
   
3. Configure:
   telegram_config = {
       "bot_token": os.getenv("TELEGRAM_BOT_TOKEN")
   }
   
4. Add to user preferences:
   store.save_user_preferences(
       user_id="user@example.com",
       preferences={
           "channels": ["telegram"],
           "telegram_chat_id": "123456789",
       }
   )

Cost: FREE, unlimited messages!

See docs/TELEGRAM_SETUP.md for full instructions.
""")


if __name__ == "__main__":
    # Run examples
    print("\nRunning Alert System Integration Examples\n")
    
    # Note: These examples won't actually send emails/Telegram unless configured
    # and user preferences are set up with valid email addresses/chat IDs
    
    example_user_setup()
    example_batch_processing()
    example_telegram_info()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Set up Telegram bot (see docs/TELEGRAM_SETUP.md)")
    print("2. Configure SMTP settings (use environment variables!)")
    print("3. Set up user preferences using scripts/alert_cli.py")
    print("4. Integrate alert processing into your pipeline")
    print("\nFor more information, see EARNINGS_GUIDANCE_ALERT_STRATEGY.md")
