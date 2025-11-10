# Credentials Setup Guide

## Where to Store Sensitive Credentials

**⚠️ NEVER commit tokens, passwords, or API keys to git!**

This project uses environment variables stored in a `.env` file for all sensitive credentials.

## Quick Setup (2 minutes)

### 1. Copy the template

```bash
cd /home/ichard/projects/event-feed-app
cp .env.example .env
```

### 2. Edit `.env` with your credentials

```bash
nano .env  # or use your favorite editor
```

### 3. Fill in your values

```bash
# Telegram (5 minutes to set up)
TELEGRAM_BOT_TOKEN=123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11

# Email (optional, if you want email alerts too)
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-specific-password
SMTP_FROM_ADDRESS=alerts@yourcompany.com
```

### 4. Verify `.env` is gitignored

```bash
grep -q "^\.env$" .gitignore && echo "✓ Safe - .env is gitignored" || echo "⚠️  WARNING: Add .env to .gitignore!"
```

## Getting Credentials

### Telegram Bot Token (FREE, 5 minutes)

1. **Open Telegram** and search for `@BotFather`

2. **Create a bot**:
   ```
   Send: /newbot
   Choose a name: "My Company Alerts"
   Choose a username: "mycompany_alerts_bot"
   ```

3. **Copy the token**:
   ```
   BotFather responds with:
   "Use this token to access the HTTP API:
   123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
   ```

4. **Add to .env**:
   ```bash
   TELEGRAM_BOT_TOKEN=123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
   ```

5. **Done!** That's it - Telegram is ready to use.

### Gmail SMTP (Optional, for email alerts)

1. **Enable 2-Step Verification** on your Google account:
   - Go to: https://myaccount.google.com/security
   - Turn on 2-Step Verification

2. **Create an App Password**:
   - Go to: https://myaccount.google.com/apppasswords
   - Select app: "Mail"
   - Select device: "Other" → enter "Alert System"
   - Click "Generate"
   - Copy the 16-character password

3. **Add to .env**:
   ```bash
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=abcd efgh ijkl mnop  # The app password
   SMTP_FROM_ADDRESS=alerts@yourcompany.com
   ```

## How the Code Uses .env

The alert system automatically loads credentials from `.env`:

```python
# In your code, just use os.getenv()
import os

config = {
    "telegram": {
        "bot_token": os.getenv("TELEGRAM_BOT_TOKEN")
    },
    "smtp": {
        "username": os.getenv("SMTP_USERNAME"),
        "password": os.getenv("SMTP_PASSWORD"),
    }
}
```

The examples automatically load `.env` using `scripts/load_env.py`.

## Alternative: System Environment Variables

Instead of `.env`, you can also set system-wide environment variables:

### Linux/Mac:
```bash
# Add to ~/.bashrc or ~/.zshrc
export TELEGRAM_BOT_TOKEN="123456789:ABC-DEF..."
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
```

### Production (Docker/Cloud):
```yaml
# docker-compose.yml
services:
  alert-system:
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - SMTP_USERNAME=${SMTP_USERNAME}
      - SMTP_PASSWORD=${SMTP_PASSWORD}
```

## Security Best Practices

### ✅ DO:
- Use `.env` files for local development
- Use environment variables in production
- Use app-specific passwords (not your main password)
- Keep `.env` in `.gitignore`
- Rotate tokens if exposed
- Use different tokens for dev/staging/prod

### ❌ DON'T:
- Commit `.env` to git
- Share tokens in Slack/email
- Use your main Gmail password
- Store tokens in code files
- Push tokens to public repos

## Checking Your Setup

Run this to verify your credentials are loaded:

```bash
cd /home/ichard/projects/event-feed-app
python3 scripts/load_env.py
```

Output should show:
```
✓ Loaded 5 environment variables from .env

Loaded variables:
  TELEGRAM_BOT_TOKEN=***MASKED***
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USERNAME=your-email@gmail.com
  SMTP_PASSWORD=***MASKED***
```

## Testing Your Bot Token

```bash
# Test that Telegram bot token works
python3 scripts/telegram_get_chat_id.py $(grep TELEGRAM_BOT_TOKEN .env | cut -d'=' -f2)
```

If the token is valid, you'll see:
```
No messages found. Have users send /start to your bot first!
```

If invalid:
```
Request failed: 401 Client Error: Unauthorized
Check that your bot token is correct.
```

## Troubleshooting

**"No .env file found"**
- Copy `.env.example` to `.env`
- Fill in your credentials

**"Unauthorized" from Telegram**
- Check token has no extra spaces
- Verify token is complete (should have `:` in the middle)
- Create a new bot if token is lost

**Email "Authentication failed"**
- Use app password, not account password
- Enable 2-Step Verification first
- Check username/password are correct

**"Module 'load_env' not found"**
- Make sure you're in the project root
- Check `scripts/load_env.py` exists
- Add scripts to path: `sys.path.insert(0, 'scripts')`

## Production Deployment

For production, use your platform's secrets management:

**Google Cloud Run**:
```bash
gcloud run deploy alert-system \
  --set-env-vars TELEGRAM_BOT_TOKEN=secretmanager://... \
  --set-env-vars SMTP_PASSWORD=secretmanager://...
```

**AWS Lambda**:
- Store in AWS Secrets Manager
- Reference in Lambda environment variables

**Kubernetes**:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: alert-secrets
type: Opaque
data:
  telegram-token: <base64-encoded-token>
```

See your platform's documentation for best practices.
