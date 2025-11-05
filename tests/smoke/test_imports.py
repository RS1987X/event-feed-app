def test_import_orchestrator_module():
    import importlib
    mod = importlib.import_module("event_feed_app.pipeline.orchestrator")
    # Module should define 'run' and 'main' entry points
    assert hasattr(mod, "run") and callable(getattr(mod, "run"))
    assert hasattr(mod, "main") and callable(getattr(mod, "main"))


def test_import_guidance_plugin_and_contract():
    from event_feed_app.events.registry import get_plugin
    plugin = get_plugin("guidance_change")

    # Minimal contract from EventPlugin Protocol
    required_callables = [
        "gate",
        "detect",
        "normalize",
        "prior_key",
        "compare",
        "decide_and_score",
        "build_rows",
    ]
    for name in required_callables:
        assert hasattr(plugin, name), f"plugin missing {name}()"
        assert callable(getattr(plugin, name)), f"plugin.{name} is not callable"

    # Required identifiers
    assert hasattr(plugin, "event_key") and isinstance(plugin.event_key, str)
    assert hasattr(plugin, "taxonomy_class") and isinstance(plugin.taxonomy_class, str)
