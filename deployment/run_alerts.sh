#!/bin/bash
# Alert detection and delivery script
# Run by cron jobs - DO NOT MOVE without updating crontab

cd /home/ichard/projects/event-feed-app

# Load environment variables
set -a  # automatically export all variables
source .env
set +a

# Activate virtual environment
source venv/bin/activate

# Run alerts using entry point (defined in pyproject.toml)
event-alerts-run --days 1
