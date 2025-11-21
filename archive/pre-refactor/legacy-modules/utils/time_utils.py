# utils/time_utils.py

from datetime import datetime, timezone
import calendar
from zoneinfo import ZoneInfo
from dateutil import parser as dateparser  # more robust than strptime

# Your RSS‐style parser config
TZ_HANDLING = {
    "DI.se RSS": {
        "struct_is_utc": True,
        "default_tz":   "UTC",
    },
    "Thomson Reuters IR": {
        "struct_is_utc": True,
        "default_tz":   "UTC",
    },
}


def parse_rss_timestamp(entry, source_name: str) -> datetime:
    """
    Normalize an RSS/Atom entry to a tz‐aware UTC datetime.
    """
    cfg = TZ_HANDLING.get(source_name, {})
    struct_is_utc = cfg.get("struct_is_utc", True)
    default_tz    = cfg.get("default_tz", "UTC")

    if entry.get("published_parsed"):
        if struct_is_utc:
            return datetime.fromtimestamp(
                calendar.timegm(entry.published_parsed),
                tz=timezone.utc,
            )
        naive = datetime(*entry.published_parsed[:6])
        return naive.replace(tzinfo=ZoneInfo(default_tz)) \
                    .astimezone(timezone.utc)

    elif entry.get("published"):
        parsed = dateparser.parse(
            entry.published,
            settings={
                "RETURN_AS_TIMEZONE_AWARE": True,
                "TIMEZONE": default_tz,
                "TO_TIMEZONE": "UTC",
            },
        )
        return parsed if parsed else datetime.now(timezone.utc)

    else:
        return datetime.now(timezone.utc)


def utc_from_epoch_ms(ms: int) -> datetime:
    """
    Convert a millisecond‐since‐epoch timestamp to tz‐aware UTC datetime.
    """
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def utc_now() -> datetime:
    """Return tz‐aware UTC "now"."""
    return datetime.now(timezone.utc)
