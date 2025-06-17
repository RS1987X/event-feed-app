# sources/web_scraper.py

import feedparser
from datetime import datetime
from core.event_types import Event
from core.company_loader import load_company_names
from core.company_matcher import detect_mentioned_company_NER
import logging
from dateutil import parser as dateparser  # more robust than strptime

logger = logging.getLogger(__name__)

COMPANY_NAMES = load_company_names()


DI_FEED_URL = "https://www.di.se/rss"

def fetch_latest_di_headlines(limit=5):
    return fetch_rss_events(DI_FEED_URL, "DI.se RSS", limit)

THOMSON_FEED = "https://ir.thomsonreuters.com/rss/news-releases.xml?items=15"

def fetch_thomson_rss(limit=10):
    return fetch_rss_events(THOMSON_FEED, "Thomson Reuters IR",limit)


def fetch_rss_events(
    feed_url: str,
    source_name: str,
    limit: int | None = None
) -> list[Event]:
    """
    Generic RSS → Event fetcher with company‐mention filtering.
    - feed_url: RSS/Atom URL
    - source_name: e.g. "DI.se RSS" or "Thomson Reuters IR"
    - limit: if set, only look at the first `limit` entries
    """
    feed = feedparser.parse(feed_url)
    events = []

    entries = feed.entries[:limit] if limit else feed.entries
    for entry in entries:
        try:
            # 1) Extract title & summary
            title   = entry.get("title", "").strip()
            summary = getattr(entry, "summary", "").strip()

            # 2) Parse a timestamp
            if entry.get("published_parsed"):
                timestamp = datetime(*entry.published_parsed[:6])
            elif entry.get("published"):
                timestamp = dateparser.parse(entry.published)
            else:
                timestamp = datetime.now()

            # 3) Company‐mention detection
            full_text = f"{title} {summary}"
            match = detect_mentioned_company_NER(full_text, COMPANY_NAMES)
            if not match:
                continue

            company, token = match

            # 4) Build the Event
            events.append(Event(
                source=source_name,
                title=title,
                timestamp=timestamp,
                content=summary,
                metadata={
                    "link": entry.get("link", ""),
                    "matched_company": company,
                    "matched_token": token
                }
            ))

        except Exception as e:
            logger.warning(
                f"[{source_name}] Error parsing entry '{title[:30]}…': {e}"
            )

    return events




# def fetch_latest_di_headlines(limit=5) -> list[Event]:
#     feed = feedparser.parse(FEED_URL)
#     events = []

#     for entry in feed.entries[:limit]:
#         try:
#             title = entry.get("title", "")
#             content = entry.get("summary", "")
#             combined = f"{title} {content}"

#             mentioned = detect_mentioned_company(combined, COMPANY_NAMES)
#             if not mentioned:
#                 continue  # Skip this entry if no company is detected

#             dt = datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else datetime.now()
#             link = entry.get("link", "")
#             matched_company, matched_token = mentioned
#             events.append(Event(
#                 source="DI.se RSS",
#                 title=title,
#                 timestamp=dt,
#                 content=content,
#                 metadata={
#                     "link": link,
#                     "company": mentioned,
#                     "type": "di/rss-event",
#                     "matched_company": matched_company,
#                     "matched_token": matched_token
#                 }
#             ))
#         except Exception as e:
#             logger.warning(f"Failed to parse DI entry: {e} — Title: {getattr(entry, 'title', '')}")

#     return events


# def fetch_thomson_rss() -> list[Event]:
#     feed = feedparser.parse(FEED_URL)
#     events = []

#     for entry in feed.entries:
#         try:
#             published = dateparser.parse(entry.published)
#             title = entry.title.strip()
#             summary = entry.summary.strip() if hasattr(entry, "summary") else ""
#             full_text = f"{title} {summary}"

#             matched_company = detect_mentioned_company(full_text, COMPANY_NAMES)
#             matched_company, matched_token = matched_company
#             if matched_company:
#                 event = Event(
#                     source="Thomson Reuters IR",
#                     title=title,
#                     timestamp=published,
#                     content=summary,
#                     metadata={
#                         "link": entry.link,
#                         "matched_company": matched_company,
#                         "matched_token": matched_token
#                     }
#                 )
#                 events.append(event)

#         except Exception as e:
#             logger.warning(f"Failed to parse Thomson Reuters entry: {e} — Title: {getattr(entry, 'title', '')}")

#     return events

