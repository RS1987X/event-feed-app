# tests/guidance_change_tests/test_config.py
def test_yaml_loads_and_compiles(plugin, loaded_yaml):
    # strong triggers compiled
    assert hasattr(plugin, "trigger_rx")
    assert plugin.trigger_regexes

    # proximity rule sets parsed
    assert plugin.prox_rules, "proximity_triggers should be populated"
    names = {r["name"] for r in plugin.prox_rules}
    assert "en_guidance_change" in names
    assert "en_guidance_initiate" in names

    # unchanged markers resolved
    assert plugin.unchanged_markers["en"], "unchanged markers should be loaded"

    # basic thresholds present
    expected_tokens = next(e for e in loaded_yaml["events"]
                           if e["key"] == "guidance_change")["thresholds"]["proximity"]["fwd_cue_metric_tokens"]
    assert plugin.prox_tokens == expected_tokens