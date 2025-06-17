# sources/web_scraper.py

import feedparser
from core.event_types import Event
from datetime import datetime
from urllib.parse import urlparse

FEED_URL = "https://www.di.se/rss"

def fetch_latest_di_headlines(limit=5):
    feed = feedparser.parse(FEED_URL)
    events = []

    for entry in feed.entries[:limit]:
        published = entry.get("published_parsed")
        dt = datetime(*published[:6]) if published else datetime.now()

        event = Event(
            source="DI.se RSS",
            title=entry.get("title", "(No Title)"),
            timestamp=dt,
            content=entry.get("summary", ""),
            metadata={"link": entry.get("link")}
        )
        events.append(event)

    return events
