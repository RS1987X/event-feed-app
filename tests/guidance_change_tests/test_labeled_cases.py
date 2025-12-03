"""
Fast feedback loop for guidance detection using real labeled examples.

Run with: pytest tests/guidance_change_tests/test_labeled_cases.py -v

This gives you immediate feedback on precision/recall after config changes.
"""
import pytest
import pandas as pd
import yaml
from pathlib import Path
from event_feed_app.events.guidance_change.plugin import GuidanceChangePlugin
from event_feed_app.config import Settings
from event_feed_app.utils.gcs_io import load_parquet_df_partitioned
import gcsfs


def load_press_release_by_id(pr_id: str) -> dict:
    """Load a single press release from GCS silver bucket by ID."""
    cfg = Settings()
    fs = gcsfs.GCSFileSystem()
    
    # Try gmail source first, then globenewswire, then prnewswire
    sources = []
    if "globenewswire__" in pr_id:
        sources = ["globenewswire"]
    elif "prnewswire__" in pr_id:
        sources = ["prnewswire"]
    else:
        sources = ["gmail"]  # Default to gmail for short IDs
    
    # Add fallback to all sources
    sources.extend(["gmail", "globenewswire", "prnewswire", "businesswire"])
    
    for source in sources:
        try:
            uri = f"{cfg.gcs_silver_root}/source={source}"
            df = load_parquet_df_partitioned(
                uri=uri,
                columns=["press_release_id", "title", "full_text", "company_name"],
                max_rows=None
            )
            
            # Filter for specific PR ID
            matches = df[df["press_release_id"] == pr_id]
            if len(matches) > 0:
                row = matches.iloc[0]
                return {
                    "press_release_id": pr_id,
                    "title": row.get("title", ""),
                    "full_text": row.get("full_text", ""),
                    "company_name": row.get("company_name", ""),
                }
        except Exception:
            continue
    
    raise ValueError(f"Press release {pr_id} not found in GCS")


@pytest.fixture(scope="module")
def plugin():
    """Load plugin with current config."""
    config_path = Path(__file__).parent.parent.parent / "src/event_feed_app/configs/significant_events.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    plugin = GuidanceChangePlugin()
    plugin.configure(cfg)
    return plugin


@pytest.fixture(scope="module")
def true_positives():
    """Load labeled true positives."""
    df = pd.read_csv("guidance_true_positives.csv")
    return df


@pytest.fixture(scope="module")
def false_positives():
    """Load labeled false positives."""
    df = pd.read_csv("guidance_false_positives.csv")
    return df


@pytest.fixture(scope="module")
def false_negatives():
    """Load labeled false negatives."""  
    df = pd.read_csv("guidance_false_negatives.csv")
    return df


def test_true_positive_recall(plugin, true_positives):
    """
    Test that we detect all labeled true positives.
    
    Recall = TP / (TP + FN)
    
    Goal: 80%+ recall on labeled TPs
    """
    detected = 0
    missed = []
    errors = []
    
    for idx, row in true_positives.iterrows():
        pr_id = row.get('press_release_id')
        if not pr_id or pd.isna(pr_id):
            errors.append(f"Missing press_release_id in row {idx}")
            continue
        
        # Load the press release from GCS
        try:
            doc = load_press_release_by_id(pr_id)
        except Exception as e:
            errors.append(f"Could not load PR {pr_id}: {e}")
            continue
        
        # Detect
        results = list(plugin.detect(doc))
        
        if len(results) > 0:
            detected += 1
        else:
            missed.append({
                'pr_id': pr_id,
                'title': doc.get('title', '')[:80],
            })
    
    total = len(true_positives)
    recall = detected / total if total > 0 else 0
    
    print(f"\n=== True Positive Recall ===")
    print(f"Detected: {detected}/{total} ({recall:.1%})")
    
    if errors:
        print(f"\nErrors loading data ({len(errors)}):")
        for e in errors[:5]:
            print(f"  - {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    if missed:
        print(f"\nMissed {len(missed)} examples:")
        for m in missed[:5]:
            print(f"  - {m['pr_id']}: {m['title']}")
        if len(missed) > 5:
            print(f"  ... and {len(missed) - 5} more")
    
    # Assert goal: 80%+ recall
    assert recall >= 0.80, f"Recall too low: {recall:.1%} (goal: 80%+)"


def test_false_positive_precision(plugin, false_positives):
    """
    Test that we DON'T detect labeled false positives.
    
    Precision = TP / (TP + FP)
    
    Goal: 90%+ precision (i.e., <10% of these FPs detected)
    """
    detected = 0
    wrongly_detected = []
    errors = []
    
    for idx, row in false_positives.iterrows():
        pr_id = row.get('press_release_id')
        if not pr_id or pd.isna(pr_id):
            errors.append(f"Missing press_release_id in row {idx}")
            continue
        
        # Load the press release from GCS
        try:
            doc = load_press_release_by_id(pr_id)
        except Exception as e:
            errors.append(f"Could not load PR {pr_id}: {e}")
            continue
        
        # Detect
        results = list(plugin.detect(doc))
        
        if len(results) > 0:
            detected += 1
            wrongly_detected.append({
                'pr_id': pr_id,
                'title': doc.get('title', '')[:80],
                'count': len(results),
            })
    
    total = len(false_positives)
    fp_rate = detected / total if total > 0 else 0
    
    print(f"\n=== False Positive Rate ===")
    print(f"Wrongly detected: {detected}/{total} ({fp_rate:.1%})")
    print(f"Precision (inverse): {1 - fp_rate:.1%}")
    
    if errors:
        print(f"\nErrors loading data ({len(errors)}):")
        for e in errors[:5]:
            print(f"  - {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    if wrongly_detected:
        print(f"\nStill detecting {len(wrongly_detected)} false positives:")
        for w in wrongly_detected[:5]:
            print(f"  - {w['pr_id']}: {w['title']} ({w['count']} candidates)")
        if len(wrongly_detected) > 5:
            print(f"  ... and {len(wrongly_detected) - 5} more")
    
    # Assert goal: <20% false positive rate
    assert fp_rate < 0.20, f"FP rate too high: {fp_rate:.1%} (goal: <20%)"


def test_false_negative_coverage(plugin, false_negatives):
    """
    Test that we now detect previously missed examples.
    
    Goal: Detect 50%+ of labeled false negatives (continuous improvement)
    """
    detected = 0
    still_missed = []
    errors = []
    
    for idx, row in false_negatives.iterrows():
        pr_id = row.get('press_release_id')
        if not pr_id or pd.isna(pr_id):
            errors.append(f"Missing press_release_id in row {idx}")
            continue
        
        # Load the press release from GCS
        try:
            doc = load_press_release_by_id(pr_id)
        except Exception as e:
            errors.append(f"Could not load PR {pr_id}: {e}")
            continue
        
        # Detect
        results = list(plugin.detect(doc))
        
        if len(results) > 0:
            detected += 1
        else:
            still_missed.append({
                'pr_id': pr_id,
                'title': doc.get('title', '')[:80],
            })
    
    total = len(false_negatives)
    coverage = detected / total if total > 0 else 0
    
    print(f"\n=== False Negative Recovery ===")
    print(f"Now detecting: {detected}/{total} ({coverage:.1%})")
    
    if errors:
        print(f"\nErrors loading data ({len(errors)}):")
        for e in errors[:5]:
            print(f"  - {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    if still_missed:
        print(f"\nStill missing:")
        for m in still_missed:
            print(f"  - {m['pr_id']}: {m['title']}")
    
    # This is informational - don't fail the test
    print(f"\nGoal: Continuously improve this metric")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
