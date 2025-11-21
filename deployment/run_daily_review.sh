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

# Run the review script
python scripts/alerts/daily_false_negative_review.py --size 10 --days 1 --category earnings
