
from importlib import import_module
from functools import lru_cache
from typing import List
from event_feed_app.taxonomy.schema import Category, Taxonomy

# ---- internal loader for your existing files ----
def _mod_path(version: str, lang: str) -> str:
    return f"event_feed_app.taxonomy.l1_{version}_{lang}"

@lru_cache(maxsize=None)
def _load_raw(version: str, lang: str):
    mod = import_module(_mod_path(version, lang))
    triples = getattr(mod, "L1_CATEGORIES")
    return [(cid, name, desc) for cid, name, desc in triples]

# ---- public API ----
def load(version: str, lang: str = "en", *, fallback_to_en: bool = True) -> Taxonomy:
    """
    Load a taxonomy for version/lang. Falls back to English if requested lang missing.
    """
    try:
        triples = _load_raw(version, lang)
    except Exception:
        if not fallback_to_en or lang == "en":
            raise
        triples = _load_raw(version, "en")
        lang = "en"
    cats = [Category(*t) for t in triples]
    return Taxonomy(version=version, lang=lang, categories=cats)

from importlib import import_module

def load_taxonomy(version: str, lang: str):
    """
    Attempts to import event_taxonomies.l1_<version>_<lang>.
    Falls back to English if that language module doesn't exist.
    """
    mods_to_try = [f"taxonomy.l1_{version}_{lang}",
                   f"taxonomy.l1_{version}_en"]
    last_err = None
    for mod in mods_to_try:
        try:
            return import_module(mod).L1_CATEGORIES
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load taxonomy for {version}/{lang}: {last_err}")

def make_cat_texts_in_base_order(base_L1, localized_L1):
    """
    Return ['<name>. <desc>', ...] in the exact CID order of base_L1.
    """
    # maps in localized taxonomy: cid -> (name, desc)
    loc_by_cid = {cid: (name, desc) for cid, name, desc in localized_L1}
    cat_texts = []
    for cid, _, _ in base_L1:
        name, desc = loc_by_cid.get(cid, ("", ""))  # fall back to empty strings if missing
        cat_texts.append(f"{name}. {desc}".strip(". "))  # keep tidy even if one part is empty
    return cat_texts