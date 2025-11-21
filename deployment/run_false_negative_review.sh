#!/bin/bash
# Daily false negative review script
# Sends 10 non-flagged PRs to Telegram for manual review

cd /home/ichard/projects/event-feed-app

# Load environment variables
set -a
source .env
set +a

# Activate virtual environment
source venv/bin/activate

# Run the review using entry point
# Sample from last 10 days for better variety (no category filter since it's not populated)
event-alerts-review --size 10 --days 10
