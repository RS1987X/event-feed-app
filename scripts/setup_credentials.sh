#!/bin/bash
# Quick setup script for alert system credentials

set -e

PROJECT_ROOT="/home/ichard/projects/event-feed-app"
cd "$PROJECT_ROOT"

echo "=================================="
echo "Alert System - Credentials Setup"
echo "=================================="
echo

# Check if .env already exists
if [ -f .env ]; then
    echo "⚠️  .env file already exists!"
    read -p "Overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file"
        echo "Edit it manually: nano .env"
        exit 0
    fi
fi

# Copy template
echo "Creating .env from template..."
cp .env.example .env

echo "✓ Created .env file"
echo

# Verify gitignore
if grep -q "^\.env$" .gitignore 2>/dev/null; then
    echo "✓ .env is safely gitignored"
else
    echo "⚠️  Adding .env to .gitignore..."
    echo ".env" >> .gitignore
fi

echo
echo "=================================="
echo "Next Steps:"
echo "=================================="
echo
echo "1. Get your Telegram bot token (5 minutes):"
echo "   - Open Telegram, search for @BotFather"
echo "   - Send: /newbot"
echo "   - Follow prompts to create your bot"
echo "   - Copy the token (looks like: 123456789:ABC-DEF...)"
echo
echo "2. Edit .env and add your token:"
echo "   nano .env"
echo
echo "3. (Optional) Add SMTP credentials for email alerts"
echo
echo "4. Test your setup:"
echo "   python3 scripts/load_env.py"
echo
echo "See docs/CREDENTIALS_SETUP.md for detailed instructions!"
echo
