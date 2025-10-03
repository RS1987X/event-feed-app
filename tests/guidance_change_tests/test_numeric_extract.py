# tests/test_numeric_extract.py
def test_currency_level_extraction_under_guidance(plugin, base_doc):
    base_doc["title"] = "Company issues full-year guidance"
    base_doc["body"] = (
        "FY2025 guidance: revenue EUR 1,750 million. "
        "The company also updates outlook."
    )
    base_doc["title_clean"] = base_doc["title"]
    base_doc["body_clean"] = base_doc["body"]

    cands = list(plugin.detect(base_doc))
    # Expect at least one 'revenue' / 'ccy' candidate
    has_revenue_ccy = any(c["metric"] == "revenue" and c["unit"] == "ccy" for c in cands)
    assert has_revenue_ccy, cands

def test_percent_margin_extraction_under_guidance(plugin, base_doc):
    base_doc["title"] = "Company raises guidance"
    base_doc["body"] = (
        "The company raises full-year guidance: operating margin expected to be "
        "about 12%–13% in FY2025."
    )
    base_doc["title_clean"] = base_doc["title"]
    base_doc["body_clean"] = base_doc["body"]

    cands = list(plugin.detect(base_doc))
    # Expect a pct candidate for operating_margin
    pct = [c for c in cands if c["unit"] == "pct" and c["metric_kind"] == "margin"]
    assert pct, cands
    # Approx band range should convert to range type
    assert pct[0]["value_type"] in {"range", "point"}


