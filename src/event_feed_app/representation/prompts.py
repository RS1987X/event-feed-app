# event_feed_app/representation/prompts.py
from enum import Enum
from typing import List
from event_feed_app.taxonomy.schema import Category, Taxonomy

class PromptStyle(str, Enum):
    E5_QUERY   = "e5_query"
    E5_PASSAGE = "e5_passage"
    PLAIN      = "plain"

# ----- Generic builders (preferred) -----

def doc_prompt(title: str, body: str, style: PromptStyle = PromptStyle.E5_PASSAGE) -> str:
    title = (title or "").strip()
    body  = (body or "").strip()
    txt   = f"[TITLE] {title}\n\n{body}" if title else body
    if style == PromptStyle.E5_QUERY:   return "query: "   + txt
    if style == PromptStyle.E5_PASSAGE: return "passage: " + txt
    return txt

def cat_prompt(name: str, desc: str, style: PromptStyle = PromptStyle.E5_QUERY) -> str:
    base = f"category: {name}. description: {desc}"
    if style == PromptStyle.E5_QUERY:   return "query: "   + base
    if style == PromptStyle.E5_PASSAGE: return "passage: " + base
    return base

# ----- Convenience wrappers (optional) -----

def cat_prompts_e5_query(tax: Taxonomy) -> List[str]:
    """Categories as E5 queries."""
    return [cat_prompt(c.name, c.desc, style=PromptStyle.E5_QUERY) for c in tax.categories]

def cat_prompts_plain(tax: Taxonomy) -> List[str]:
    """Categories without any prefix."""
    return [cat_prompt(c.name, c.desc, style=PromptStyle.PLAIN) for c in tax.categories]

def doc_prompt_e5_passage(title: str, body: str) -> str:
    """Doc as E5 passage."""
    return doc_prompt(title, body, style=PromptStyle.E5_PASSAGE)
