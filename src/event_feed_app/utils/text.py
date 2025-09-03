
def build_doc_for_embedding(title: str, body: str, e5_prefix: bool) -> str:
    title = (title or "").strip()
    body  = (body or "").strip()
    txt   = f"[TITLE] {title}\n\n{body}" if title else body
    return f"passage: {txt}" if e5_prefix else txt

def build_cat_query(name: str, desc: str, e5_prefix: bool) -> str:
    q = f"category: {name}. description: {desc}"
    return "query: " + q if e5_prefix else q
