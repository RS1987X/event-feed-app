# tests/guidance_change_tests/conftest.py
from datetime import datetime, timezone
import pathlib, yaml
import pytest
from event_feed_app.events.guidance_change.plugin import GuidanceChangePlugin

YAML_PATH = pathlib.Path("tests/guidance_change_tests/data/guidance_change_test.yml")

@pytest.fixture(scope="session")
def loaded_yaml():
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@pytest.fixture
def plugin(loaded_yaml):
    ev = next(e for e in loaded_yaml["events"] if e["key"] == "guidance_change")
    p = GuidanceChangePlugin()
    p.configure(ev)  # IMPORTANT: pass the single event block
    return p

@pytest.fixture
def base_doc():
    return {
        "doc_id": "test-doc-1",
        "cluster_id": "test-clu-1",
        "source_type": "issuer_pr",
        "title": "",
        "body": "",
        "title_clean": "",
        "body_clean": "",
        "published_utc": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "source_url": "https://example.com/test",
    }


@pytest.fixture
def doc_factory(base_doc):
    def make(title, body, **kwargs):
        d = dict(base_doc)
        d.update({
            "title": title,
            "body": body,
            "title_clean": title,
            "body_clean": body,
        })
        # let tests override source_type, etc.
        d.update(kwargs)
        return d
    return make