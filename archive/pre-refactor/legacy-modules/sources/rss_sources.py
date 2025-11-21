# sources/web_scraper.py

import feedparser
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
STOCKHOLM = ZoneInfo("Europe/Stockholm")
from core.event_types import Event
from core.company_loader import load_company_names
from core.company_matcher import detect_mentioned_company_NER
import logging
from dateutil import parser as dateparser  # more robust than strptime
import calendar
from utils.time_utils import parse_rss_timestamp, utc_now

logger = logging.getLogger(__name__)

COMPANY_NAMES = load_company_names()


DI_FEED_URL = "https://www.di.se/rss"

def fetch_latest_di_headlines(limit):
    return fetch_rss_events(DI_FEED_URL, "DI.se RSS", limit, display_tz= None)

THOMSON_FEED = "https://ir.thomsonreuters.com/rss/news-releases.xml?items=15"

def fetch_thomson_rss(limit):
    return fetch_rss_events(THOMSON_FEED, "Thomson Reuters IR",limit,display_tz=None)



# # Declare how each source treats its timestamps
# TZ_HANDLING = {
#     "DI.se RSS": {
#         "struct_is_utc": True,    # feedparser.published_parsed is UTC
#         "default_tz": "UTC",      # free‐form dates default to UTC
#     },
#     "Thomson Reuters IR": {
#         "struct_is_utc": True,   # published_parsed is actually local Stockholm
#         "default_tz": "UTC",
#     },
# }

# def parse_timestamp(entry, source_name: str) -> datetime:
#     """
#     Return a tz-aware UTC datetime for this entry, no matter what the feed gave you.
#     """
#     cfg = TZ_HANDLING.get(source_name, {})
#     struct_is_utc = cfg.get("struct_is_utc", True)
#     default_tz     = cfg.get("default_tz", "UTC")

#     # 1) If feedparser gave us published_parsed (always naive struct_time)
#     if entry.get("published_parsed"):
#         if struct_is_utc:
#             # trust it as UTC
#             utc_dt = datetime.fromtimestamp(
#                 calendar.timegm(entry.published_parsed),
#                 tz=timezone.utc
#             )
#         else:
#             # treat struct_time as local to default_tz, then convert to UTC
#             naive = datetime(*entry.published_parsed[:6])
#             local_dt = naive.replace(tzinfo=ZoneInfo(default_tz))
#             utc_dt = local_dt.astimezone(timezone.utc)

#     # 2) Else if we only have a free‐form published string
#     elif entry.get("published"):
#         parsed = dateparser.parse(
#             entry.published,
#             settings={
#                 "RETURN_AS_TIMEZONE_AWARE": True,
#                 "TIMEZONE": default_tz,
#                 "TO_TIMEZONE": "UTC",
#             },
#         )
#         utc_dt = parsed if parsed else datetime.now(timezone.utc)

#     # 3) No timestamp at all: fallback to now
#     else:
#         utc_dt = datetime.now(timezone.utc)

#     return utc_dt


def fetch_rss_events(
    feed_url: str,
    source_name: str,
    limit: int | None = None,
    display_tz: ZoneInfo | None = None
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

            # # parse as UTC first
            # if entry.get("published_parsed"):
            #     # published_parsed is a time.struct_time in UTC (no tzinfo)
            #     utc_dt = datetime.fromtimestamp(
            #         calendar.timegm(entry.published_parsed),
            #         tz=timezone.utc
            #         )
            #     #timestamp = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            # elif entry.get("published"):
            #     parsed = dateparser.parse(entry.published)
            #     # ensure tz-aware in UTC
            #     if parsed.tzinfo is None:
            #         utc_dt = parsed.replace(tzinfo=timezone.utc)
            #     else:
            #         utc_dt = parsed.astimezone(timezone.utc)
            # else:
            #     utc_dt = datetime.now(timezone.utc)
            
            # # 2) Optionally convert to a display timezone
            # if display_tz:
            #     ts = utc_dt.astimezone(display_tz)
            # else:
            #     ts = utc_dt

            #fetched_at = datetime.now()

            # 1) Normalize published date into a UTC datetime
            utc_dt = parse_rss_timestamp(entry, source_name)

            # 2) Store both timestamp & fetched_at in UTC
            #    If you ever want a local display version, do that in your UI
            ts = utc_dt                         # tz-aware UTC
            fetched_at = utc_now()

            # 3) Company‐mention detection
            full_text = f"{title} {summary}"
            matches = detect_mentioned_company_NER(full_text, COMPANY_NAMES)
            if not matches:
                continue
            
            #company, token = match
            # Only take first match for simplicity
            company_names = [name for name, _ in matches]
            #tickers = [c["ticker"] for c in COMPANY_NAMES if c["name"] in company_names]
            tickers = [
                c.get("ticker", "")
                for c in COMPANY_NAMES
                if c.get("name") in company_names and c.get("ticker", "")
            ]
            # 4) Build the Event
            events.append(Event(
                source=source_name,
                title=title,
                timestamp=ts,
                fetched_at=fetched_at,
                content="",
                metadata={
                    "summary" : summary,
                    "link":  entry.get("link", ""),
                    "company": ", ".join(company_names),
                    "ticker": ", ".join(tickers),
                    "matches": matches  # keep full list if needed later
                }
            ))


            # for company, token in matches:
            #     events.append(Event(
            #         source=source_name,
            #         title=title,
            #         timestamp=timestamp,
            #         content=summary,
            #         metadata={
            #             "link": entry.get("link", ""),
            #             "matched_company": company,
            #             "matched_token": token
            #         }
            #     ))

        except Exception as e:
            logger.warning(
                f"[{source_name}] Error parsing entry '{title[:30]}…': {e}"
            )

    return events
