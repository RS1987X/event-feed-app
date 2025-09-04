from event_feed_app.taxonomy.adapters import Category, Taxonomy
from typing import List

def texts_lsa(tax: Taxonomy) -> List[str]:
    """ '<name>. <desc>' in taxonomy CID order (for TF-IDF). """
    return [f"{c.name}. {c.desc}".strip(". ") for c in tax.categories]
