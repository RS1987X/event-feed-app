def test_yaml_rules_are_well_formed(loaded_yaml, plugin):
    plugin.configure(loaded_yaml)
    for r in plugin.prox_rules:
        assert r["name"]
        assert r["type"] in {"guidance_change","guidance_unchanged","guidance_initiate"}
        assert r["metrics"]
        assert bool(r["verbs"]) ^ bool(r["unchanged_tokens"])  # exactly one arm populated
        assert r["window_tokens"] > 0