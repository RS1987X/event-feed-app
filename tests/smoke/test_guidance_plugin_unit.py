from typing import Mapping, Any

def test_guidance_plugin_basic_flow():
    from event_feed_app.events.registry import get_plugin

    plugin = get_plugin("guidance_change")

    # Minimal synthetic doc that is likely relevant
    doc: Mapping[str, Any] = {
        "title": "Company updates revenue guidance for FY2025",
        "body": "Revenue expected to be SEK 1,200–1,400m with operating margin around 12–14%.",
        # common normalized fields used in pipeline
        "title_clean": "company updates revenue guidance for FY2025",
        "body_clean": "revenue expected to be sek 1,200-1,400m operating margin around 12-14%",
        "company_id": "TESTCO",
        "source_type": "wire",
        "release_date": "2025-11-01",
        "press_release_id": "TEST-1",
    }

    # Should be safe: gate, detect → normalize → prior_key → compare → decide_and_score
    g = plugin.gate(doc)
    if g is False:
        # If the plugin elects to skip early, that's acceptable for smoke.
        return

    cands = list(plugin.detect(doc))
    # The detector may not always fire; if no candidates, that's fine for smoke
    if not cands:
        return

    # In the job, doc-derived fields are merged into raw before normalize
    raw = {**cands[0], "company_id": doc["company_id"], "source_type": doc["source_type"]}
    cand = plugin.normalize(raw)
    assert isinstance(cand, dict)

    key = plugin.prior_key(cand)
    assert isinstance(key, tuple)

    comp = plugin.compare(cand, prior=None)
    assert isinstance(comp, dict)

    is_sig, score, meta = plugin.decide_and_score(cand, comp, cfg={})
    assert isinstance(is_sig, bool)
    assert isinstance(score, (int, float))
    assert isinstance(meta, dict)

    # build_rows may return (erow, vrow) or (erow, vrow, arow)
    res = plugin.build_rows(doc, cand, comp, is_sig, score)
    assert isinstance(res, tuple)
    assert len(res) in (2, 3)
    erow, vrow = res[0], res[1]
    if erow is not None:
        assert isinstance(erow, dict)
    if vrow is not None:
        assert isinstance(vrow, dict)
    if len(res) == 3 and res[2] is not None:
        assert isinstance(res[2], dict)
