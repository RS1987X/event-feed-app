# tests/test_end_to_end.py
def test_end_to_end_build_rows(plugin, base_doc):
    base_doc["title"] = "Company raises guidance"
    base_doc["body"] = (
        "FY2025 guidance: revenue EUR 1,750 million. "
        "Operating margin expected to be approximately 12%–13%."
    )
    base_doc["title_clean"] = base_doc["title"]
    base_doc["body_clean"] = base_doc["body"]

    cands = list(plugin.detect(base_doc))
    assert cands

    # Wire through normalize/compare/decide/build
    out_rows = []
    for cand in cands:
        # set company_id expected by downstream steps
        cand["company_id"] = "name:acme-corp"
        cand["_doc_title"] = base_doc["title"]
        cand["_doc_body"]  = base_doc["body"]

        norm = plugin.normalize(cand)
        comp = plugin.compare(norm, prior=None)
        is_sig, score, feats = plugin.decide_and_score(norm, comp, plugin.cfg)
        versions_row, event_row, audit_row = plugin.build_rows(base_doc, norm, comp, is_sig, score)

        assert versions_row and event_row and audit_row
        assert "event_id" in event_row
        out_rows.append((versions_row, event_row, audit_row))

    # We should have at least the revenue (ccy) and one margin (pct)
    assert any(r[0]["unit"] == "ccy" for r in out_rows)
    assert any(r[0]["unit"] == "pct" for r in out_rows)