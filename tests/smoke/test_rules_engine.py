import pandas as pd

def test_rules_engine_classify_batch_minimal():
    from event_feed_app.taxonomy.rules_engine import classify_batch
    from event_feed_app.taxonomy.keywords.keyword import keywords

    df = pd.DataFrame([
        {"title_clean": "Company announces product launch partnership", "full_text_clean": "New product partnership announced today."},
        {"title_clean": "Revenue guidance updated for FY2025", "full_text_clean": "The company updates its revenue outlook."},
    ])

    out = classify_batch(df.copy(), text_cols=("title_clean", "full_text_clean"), keywords=keywords)

    # Columns that the orchestrator relies on later
    assert "rule_pred_cid" in out.columns
    assert "rule_pred_details" in out.columns

    # Values should be strings or None
    assert all(cid is None or isinstance(cid, str) for cid in out["rule_pred_cid"])