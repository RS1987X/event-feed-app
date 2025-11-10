# Telegram Alert Setup Guide

Telegram alerts are **FREE** and **easy** to set up! No cost, instant delivery, works on all platforms.

## Why Telegram?

✅ **Free**: No cost for bot API usage  
✅ **Fast**: Instant push notifications  
✅ **Cross-platform**: Works on mobile, desktop, web  
✅ **Rich formatting**: Supports markdown, emojis, links  
✅ **Reliable**: High uptime, proven delivery  

## Setup Steps (5 minutes)

### 1. Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` command
3. Follow prompts to name your bot (e.g., "MyCompany Guidance Alerts")
4. BotFather gives you a **bot token** like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`
5. **Save this token** - you'll need it for configuration

### 2. Get User Chat IDs

Each user needs to get their `chat_id`:

**Option A: Use a helper script** (recommended)

```python
# scripts/telegram_get_chat_id.py
import requests
import sys

bot_token = sys.argv[1]  # Pass your bot token
url = f"https://api.telegram.org/bot{bot_token}/getUpdates"

print("Send /start to your bot on Telegram, then run this script...")
response = requests.get(url).json()

for update in response.get("result", []):
    chat = update.get("message", {}).get("chat", {})
    chat_id = chat.get("id")
    username = chat.get("username", "N/A")
    print(f"Chat ID: {chat_id}, Username: @{username}")
```

Run:
```bash
python scripts/telegram_get_chat_id.py YOUR_BOT_TOKEN
```

**Option B: Use curl**

1. Have user send `/start` to your bot
2. Run:
   ```bash
   curl https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
3. Look for `"chat":{"id":123456789,...}` in the response

### 3. Configure the Alert System

```python
from src.event_feed_app.alerts.delivery import AlertDelivery
from src.event_feed_app.alerts.store import AlertStore

# Initialize with Telegram config
telegram_config = {
    "bot_token": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"  # From BotFather
}

delivery = AlertDelivery(
    telegram_config=telegram_config,
    smtp_config={...}  # Optional email config
)
```

### 4. Add Users with Telegram Preferences

```python
store = AlertStore()

# Add user with Telegram channel enabled
store.save_user_preferences(
    user_id="john@example.com",
    preferences={
        "channels": ["telegram", "email"],  # Enable both
        "telegram_chat_id": "123456789",    # From step 2
        "min_significance": 0.5,
    }
)
```

### 5. Test It!

```python
# Send a test alert
test_alert = {
    "alert_id": "test-123",
    "company_name": "ACME Corp",
    "summary": "Q4 2024 revenue guidance raised",
    "significance_score": 0.85,
    "metrics": [
        {
            "metric": "revenue",
            "direction": "up",
            "new_value": "1.2B",
            "prior_value": "1.0B",
        }
    ],
    "metadata": {
        "period": "Q4-2024",
        "press_release_url": "https://example.com/pr",
    }
}

delivery.deliver(test_alert)
```

You should see a nicely formatted message on Telegram! 🎉

## Message Format

Telegram messages use Markdown formatting:

```
🔔 *ACME Corp - Guidance Alert*

📈 *Revenue*: $1.2B ↑20% (was $1.0B)

📅 Period: `Q4-2024`
⭐ Significance: *85/100*

[📄 View Press Release](https://example.com/pr)
```

## Cost Analysis

**Telegram Bot API**: FREE ✅
- No limits on message volume for personal/small business use
- No subscription fees
- No per-message costs

**Comparison**:
- Twilio SMS: ~$0.0075/message (not free)
- SendGrid Email: Free tier limited to 100/day
- Telegram: Unlimited, free forever

## Security Notes

- **Bot token**: Keep it secret! Don't commit to git
- **Chat IDs**: Not sensitive, but keep private
- **Store credentials**: Use environment variables or secrets management

```bash
# .env file (add to .gitignore!)
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
```

```python
import os
telegram_config = {
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN")
}
```

## Advanced: Webhook Setup (Optional)

For high-volume bots, set up webhooks instead of polling:

```python
import requests

bot_token = "YOUR_BOT_TOKEN"
webhook_url = "https://your-domain.com/telegram-webhook"

requests.post(
    f"https://api.telegram.org/bot{bot_token}/setWebhook",
    json={"url": webhook_url}
)
```

But for guidance alerts (low volume), getUpdates is fine!

## Troubleshooting

**"Unauthorized" error**
- Check bot token is correct
- Make sure no extra spaces in token

**Messages not delivered**
- Verify user has sent `/start` to bot
- Check chat_id is correct (numeric)
- Ensure bot is not blocked by user

**"Chat not found"**
- User must send `/start` to bot first
- Chat ID must be numeric (not username)

## References

- [Telegram Bot API Docs](https://core.telegram.org/bots/api)
- [BotFather Commands](https://core.telegram.org/bots#6-botfather)
- [Markdown Formatting](https://core.telegram.org/bots/api#markdown-style)
