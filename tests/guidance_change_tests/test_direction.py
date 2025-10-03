# tests/test_direction.py
def test_direction_hint_up(plugin, base_doc):
    base_doc["title"] = "Company updates outlook"
    base_doc["body"] = (
        "Management expects to improve operating margin to about 12% in FY2025 guidance."
    )
    base_doc["title_clean"] = base_doc["title"]
    base_doc["body_clean"] = base_doc["body"]

    cands = list(plugin.detect(base_doc))
    assert cands, "Extraction should succeed"
    # At least one candidate should carry a direction hint of 'up'
    assert any(c.get("direction_hint") == "up" for c in cands)
