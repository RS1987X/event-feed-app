# tests/test_suppressors.py
def test_invitation_in_same_sentence_is_suppressed(plugin, base_doc):
    # Invite + provide guidance in the SAME sentence → suppressed by cue-sentence checks
    base_doc["title"] = "Webcast"
    base_doc["body"] = "We invite investors and will provide full-year guidance for FY2025."
    base_doc["title_clean"] = base_doc["title"]
    base_doc["body_clean"] = base_doc["body"]

    cands = list(plugin.detect(base_doc))
    assert not cands, "Cue sentence contains invitation → should be suppressed"

def test_analyst_voice_suppressed_for_non_issuer(plugin, base_doc):
    base_doc["source_type"] = "wire"
    base_doc["title"] = "Analyst update"
    base_doc["body"] = (
        "Bloomberg consensus says the company raises guidance. FY2025 operating margin 12%."
    )
    base_doc["title_clean"] = base_doc["title"]
    base_doc["body_clean"] = base_doc["body"]

    cands = list(plugin.detect(base_doc))
    assert not cands, "Analyst voice (wire source) should be suppressed"
