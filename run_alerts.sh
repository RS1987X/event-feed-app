#!/bin/bash
cd /home/ichard/projects/event-feed-app
set -a  # automatically export all variables
source .env
set +a
venv/bin/python -m event_feed_app.alerts.runner --days 1
