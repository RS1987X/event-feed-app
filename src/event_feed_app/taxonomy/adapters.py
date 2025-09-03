from dataclasses import dataclass
from importlib import import_module
from functools import lru_cache
from typing import List

# ---- canonical data shapes ----
@dataclass(frozen=True)
class Category:
    cid: str
    name: str
    desc: str

@dataclass(frozen=True)
class Taxonomy:
    version: str
    lang: str
    categories: List[Category]

    @property
    def ordered_cids(self) -> List[str]:
        return [c.cid for c in self.categories]

    @property
    def cid2name(self) -> dict:
        return {c.cid: c.name for c in self.categories}

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

def texts_lsa(tax: Taxonomy) -> List[str]:
    """ '<name>. <desc>' in taxonomy CID order (for TF-IDF). """
    return [f"{c.name}. {c.desc}".strip(". ") for c in tax.categories]

def texts_emb(tax: Taxonomy, *, e5_prefix: bool) -> List[str]:
    """ Query strings for embedding models (e5 optional prefix). """
    qs = [f"category: {c.name}. description: {c.desc}" for c in tax.categories]
    return [("query: " + q) if e5_prefix else q for q in qs]

def align_texts(base: Taxonomy, localized: Taxonomy) -> List[str]:
    """
    Build localized '<name>. <desc>' but strictly in the CID order of `base`.
    Missing CIDs produce empty strings so alignment never breaks.
    """
    loc_by_cid = {c.cid: (c.name, c.desc) for c in localized.categories}
    out = []
    for c in base.categories:
        name, desc = loc_by_cid.get(c.cid, ("", ""))
        out.append(f"{name}. {desc}".strip(". "))
    return out



from importlib import import_module

def load_taxonomy(version: str, lang: str):
    """
    Attempts to import event_taxonomies.l1_<version>_<lang>.
    Falls back to English if that language module doesn't exist.
    """
    mods_to_try = [f"event_taxonomies.l1_{version}_{lang}",
                   f"event_taxonomies.l1_{version}_en"]
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