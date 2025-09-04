from typing import List
from event_feed_app.taxonomy.adapters import Taxonomy

def align_texts(base: Taxonomy, localized: Taxonomy) -> List[str]:
    """Localized '<name>. <desc>' in base CID order; missing CIDs → ''."""
    loc_by_cid = {c.cid: (c.name, c.desc) for c in localized.categories}
    out = []
    for c in base.categories:
        name, desc = loc_by_cid.get(c.cid, ("", ""))
        out.append(f"{name}. {desc}".strip(". "))
    return out