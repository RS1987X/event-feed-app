# tests/test_proximity_rules.py
def test_prox_rule_guidance_change_allows_numeric(plugin, base_doc):
    # verb + guidance within a short window should open the gate
    base_doc["title"] = "Company raises outlook"
    base_doc["body"] = "FY2025: operating margin expected ~12%."
    base_doc["title_clean"] = base_doc["title"]
    base_doc["body_clean"] = base_doc["body"]

    cands = list(plugin.detect(base_doc))
    assert cands, "Proximity rule should open the gate and allow extraction"

def test_initiate_requires_period_token(plugin, base_doc):
    # Contains 'provide guidance' but NO period token → should NOT pass the initiate rule
    base_doc["title"] = "Company will provide guidance"
    base_doc["body"] = "The company will provide guidance shortly."
    base_doc["title_clean"] = base_doc["title"]
    base_doc["body_clean"] = base_doc["body"]

    cands = list(plugin.detect(base_doc))
    # Gate may still fail; at minimum we assert we didn't emit any text-only stub
    assert not cands

    # Now add a period token nearby → should pass
    base_doc["body"] = "The company will provide guidance for FY2025."
    base_doc["body_clean"] = base_doc["body"]
    c2 = list(plugin.detect(base_doc))
    assert c2, "Initiate rule with period token should pass (may yield a stub or numeric)"
