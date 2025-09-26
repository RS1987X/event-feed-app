# event_feed_app/taxonomy/keywords/keyword.py
from importlib import resources
import json, os

# Choose JSON via env or bundled resource
_RESOURCE = os.getenv("KEYWORDS_JSON_RESOURCE", "keywords_taxonomy-v4_lexicon-v1.4.0.json")
_PATH     = os.getenv("KEYWORDS_JSON_PATH")  # absolute/relative filesystem path (optional)

def _load_keywords_json():
    if _PATH:
        with open(_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        source = _PATH
    else:
        # Load from package data in this package (event_feed_app.taxonomy.keywords)
        with resources.files(__package__).joinpath(_RESOURCE).open("r", encoding="utf-8") as f:
            data = json.load(f)
        source = _RESOURCE
    return data, source

_data, SOURCE = _load_keywords_json()   # 👈 now exported and importable

SCHEMA_VERSION   = _data["schema_version"]
LEXICON_VERSION  = _data["lexicon_version"]
TAXONOMY_VERSION = _data["taxonomy_version"]
CATEGORIES       = _data["categories"]

# Public alias used by the rules engine
keywords = CATEGORIES

__all__ = (
    "keywords",
    "SCHEMA_VERSION",
    "LEXICON_VERSION",
    "TAXONOMY_VERSION",
    "SOURCE",
)
