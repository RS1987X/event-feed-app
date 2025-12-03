#!/usr/bin/env python3
"""
Diagnose why a specific press release was flagged as guidance.

Usage:
    python scripts/diagnose_alert.py globenewswire__0943085f6f199ae2
    python scripts/diagnose_alert.py --pr-id globenewswire__0943085f6f199ae2
"""
import sys
import os
from pathlib import Path
import argparse

# Add project to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from load_env import load_env
load_env()

from event_feed_app.config import Settings
from event_feed_app.events.guidance_change.plugin import GuidanceChangePlugin
from event_feed_app.alerts.alert_payload_store import AlertPayloadStore
import pyarrow.parquet as pq
import pyarrow.compute as pc
import json
from datetime import datetime, timedelta


def load_pr_from_alert_payload(pr_id: str):
    """Try to load PR from alert payloads first (much faster)."""
    print("📦 Searching alert payloads (fast path)...")
    
    try:
        from google.cloud import storage
        client = storage.Client()
        
        # Try different bucket names
        for bucket_name in ["event-feed-app-data", "event-feed-app-463206-alerts"]:
            try:
                bucket = client.bucket(bucket_name)
                
                # Search recent payloads (last 30 days, newest first)
                prefix = "feedback/alert_payloads/guidance_change/"
                blobs = list(bucket.list_blobs(prefix=prefix))
                
                # Sort by time_created descending (newest first)
                blobs.sort(key=lambda b: b.time_created, reverse=True)
                
                print(f"   Searching {len(blobs)} payloads in {bucket_name}...")
                
                for blob in blobs:
                    # Quick check: if PR ID is in blob name or recent enough
                    if pr_id in blob.name or blob.time_created > datetime.now(blob.time_created.tzinfo) - timedelta(days=30):
                        try:
                            content = blob.download_as_text()
                            if pr_id in content:
                                payload = json.loads(content)
                                if payload.get("press_release_id") == pr_id:
                                    print(f"   ✅ Found in alert payload!")
                                    
                                    # Extract PR data from metadata
                                    metadata = payload.get("metadata", {})
                                    pr_dict = {
                                        "press_release_id": pr_id,
                                        "title": metadata.get("title", ""),
                                        "body": metadata.get("body", ""),
                                        "body_clean": metadata.get("body", ""),
                                        "full_text": metadata.get("body", ""),
                                        "company_name": payload.get("company_name", ""),
                                        "source": metadata.get("source", ""),
                                        "press_release_url": metadata.get("press_release_url", ""),
                                        "source_url": metadata.get("press_release_url", ""),
                                    }
                                    return pr_dict
                        except:
                            continue
            except:
                continue
        
        print("   ❌ Not found in alert payloads")
        return None
        
    except Exception as e:
        print(f"   ⚠️  Could not search payloads: {e}")
        return None


def load_pr_from_gcs(pr_id: str):
    """Load a press release from GCS silver data (slower fallback)."""
    settings = Settings()
    # Use GCS_SILVER_ROOT from env
    gcs_silver = os.getenv("GCS_SILVER_ROOT", "gs://event-feed-app-data/silver_normalized/table=press_releases")
    if not gcs_silver.startswith("gs://"):
        gcs_silver = f"gs://{gcs_silver}"
    gcs_path = gcs_silver.rstrip("/")
    
    print(f"📦 Loading PR from GCS silver (slow path): {gcs_path}")
    print(f"   Filtering by press_release_id = {pr_id}")
    print(f"   ⏳ This may take 30-60 seconds due to partition scanning...")
    
    try:
        # Read table without filters first to avoid schema merge issues
        import pyarrow.dataset as ds
        dataset = ds.dataset(gcs_path, format="parquet")
        
        # Scan with filter - use available schema fields
        table = dataset.to_table(
            filter=ds.field("press_release_id") == pr_id,
            columns=[
                "press_release_id", "title", "full_text",
                "source", "company_name", "source_url"
            ]
        )
        
        if len(table) == 0:
            print(f"❌ PR {pr_id} not found in GCS")
            return None
        
        # Convert to dict and normalize field names
        pr_dict = {}
        for col in table.column_names:
            pr_dict[col] = table[col][0].as_py()
        
        # Map full_text to body/body_clean for plugin compatibility
        if "full_text" in pr_dict:
            pr_dict["body"] = pr_dict["full_text"]
            pr_dict["body_clean"] = pr_dict["full_text"]
        if "source_url" in pr_dict:
            pr_dict["press_release_url"] = pr_dict["source_url"]
        
        print(f"✅ Found PR: {pr_dict.get('title', 'Untitled')}")
        return pr_dict
        
    except Exception as e:
        print(f"❌ Error loading from GCS: {e}")
        return None


def check_exclusions(pr: dict, plugin: GuidanceChangePlugin):
    """Check if PR should be excluded and why."""
    text = f"{pr.get('title', '')} {pr.get('body_clean', '')}".lower()
    
    print("\n" + "=" * 80)
    print("🔍 EXCLUSION ANALYSIS")
    print("=" * 80)
    
    # Check securities lawsuits
    print("\n📋 Securities Lawsuits Check:")
    lawsuit_cfg = plugin.exclusions.get("securities_lawsuits", {})
    if lawsuit_cfg:
        indicators = [ind.lower() for ind in lawsuit_cfg.get("indicators", [])]
        law_firms = [firm.lower() for firm in lawsuit_cfg.get("law_firms", [])]
        
        found_indicators = [ind for ind in indicators if ind in text]
        found_firms = [firm for firm in law_firms if firm in text]
        
        if found_indicators:
            print(f"  ✅ Found indicators: {', '.join(found_indicators)}")
        else:
            print(f"  ❌ No indicators found")
            
        if found_firms:
            print(f"  ✅ Found law firms: {', '.join(found_firms)}")
        else:
            print(f"  ❌ No law firms found")
        
        should_exclude = bool(found_indicators or found_firms)
        if should_exclude:
            print(f"  ⚠️  SHOULD BE EXCLUDED (OR logic: indicator OR law_firm)")
        else:
            print(f"  ✓ Passes securities check")
    else:
        print("  (No securities exclusion config)")
    
    # Check market research
    print("\n📋 Market Research Check:")
    research_cfg = plugin.exclusions.get("market_research", {})
    if research_cfg:
        indicators = [ind.lower() for ind in research_cfg.get("indicators", [])]
        firms = [firm.lower() for firm in research_cfg.get("firms", [])]
        
        found_indicators = [ind for ind in indicators if ind in text]
        found_firms = [firm for firm in firms if firm in text]
        
        if found_indicators:
            print(f"  ✅ Found indicators ({len(found_indicators)}): {', '.join(found_indicators[:5])}")
        else:
            print(f"  ❌ No indicators found")
            
        if found_firms:
            print(f"  ✅ Found firms: {', '.join(found_firms)}")
        else:
            print(f"  ❌ No firms found")
        
        # Logic: "2+ indicators OR (indicator AND firm)"
        should_exclude = len(found_indicators) >= 2 or (len(found_indicators) >= 1 and found_firms)
        if should_exclude:
            print(f"  ⚠️  SHOULD BE EXCLUDED (2+ indicators OR indicator+firm)")
        else:
            print(f"  ✓ Passes market research check")
    else:
        print("  (No market research exclusion config)")
    
    # Run actual exclusion check
    print("\n📋 Plugin Exclusion Result:")
    excluded = plugin._should_exclude_document(f"{pr.get('title', '')} {pr.get('body_clean', '')}")
    if excluded:
        print("  ⚠️  EXCLUDED by plugin")
    else:
        print("  ✓ NOT excluded by plugin")
    
    return excluded


def analyze_detections(pr: dict, plugin: GuidanceChangePlugin):
    """Run detection and show what was found."""
    print("\n" + "=" * 80)
    print("🎯 DETECTION ANALYSIS")
    print("=" * 80)
    
    candidates = list(plugin.detect(pr))
    
    if not candidates:
        print("\n❌ No guidance candidates detected")
        return
    
    print(f"\n✅ Found {len(candidates)} candidate(s):")
    
    for idx, cand in enumerate(candidates, 1):
        print(f"\n  Candidate #{idx}:")
        print(f"    Metric: {cand.get('metric', 'N/A')}")
        print(f"    Metric Kind: {cand.get('metric_kind', 'N/A')}")
        print(f"    Unit: {cand.get('unit', 'N/A')}")
        print(f"    Period: {cand.get('period', 'N/A')}")
        print(f"    Value Type: {cand.get('value_type', 'N/A')}")
        
        if cand.get('value_low') is not None:
            print(f"    Value: {cand.get('value_low')} - {cand.get('value_high')}")
        
        print(f"    Trigger Source: {cand.get('_trigger_source', 'N/A')}")
        print(f"    Trigger Label: {cand.get('_trigger_label', 'N/A')}")
        
        if cand.get('_trigger_match'):
            print(f"    Trigger Match: '{cand['_trigger_match'][:100]}'")
        
        if cand.get('snippet'):
            print(f"    Snippet: '{cand['snippet'][:150]}'")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose why a press release was flagged as guidance"
    )
    parser.add_argument(
        "pr_id",
        nargs="?",
        help="Press release ID (e.g., globenewswire__0943085f6f199ae2)"
    )
    parser.add_argument(
        "--pr-id",
        dest="pr_id_flag",
        help="Press release ID (alternative flag)"
    )
    
    args = parser.parse_args()
    
    pr_id = args.pr_id or args.pr_id_flag
    
    if not pr_id:
        parser.print_help()
        sys.exit(1)
    
    print("=" * 80)
    print(f"🔍 ALERT DIAGNOSTIC: {pr_id}")
    print("=" * 80)
    
    # Load PR
    pr = load_pr_from_gcs(pr_id)
    if not pr:
        sys.exit(1)
    
    # Display PR info
    print("\n" + "=" * 80)
    print("📄 PRESS RELEASE INFO")
    print("=" * 80)
    print(f"ID: {pr.get('press_release_id')}")
    print(f"Company: {pr.get('company_name', 'N/A')}")
    print(f"Source: {pr.get('source', 'N/A')}")
    print(f"Source Type: {pr.get('source_type', 'N/A')}")
    print(f"Release Date: {pr.get('release_date', 'N/A')}")
    if pr.get('press_release_url'):
        print(f"URL: {pr['press_release_url']}")
    
    print(f"\nTitle: {pr.get('title', 'N/A')}")
    
    body = pr.get('body_clean', '') or pr.get('body', '') or pr.get('full_text', '')
    print(f"\nBody Preview ({len(body)} chars):")
    print(body[:500])
    if len(body) > 500:
        print("...")
    
    # Initialize plugin with config
    print("\n" + "=" * 80)
    print("⚙️  INITIALIZING PLUGIN")
    print("=" * 80)
    
    config_path = project_root / "src" / "event_feed_app" / "configs" / "significant_events.yaml"
    print(f"Loading config from: {config_path}")
    
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    plugin = GuidanceChangePlugin()
    guidance_config = config.get("guidance_change", {})
    plugin.configure(guidance_config)
    
    print(f"✅ Plugin configured")
    print(f"   Exclusions loaded: {list(plugin.exclusions.keys())}")
    
    # Check exclusions
    excluded = check_exclusions(pr, plugin)
    
    # Run detection
    if not excluded:
        analyze_detections(pr, plugin)
    else:
        print("\n⚠️  PR was excluded, skipping detection analysis")
    
    print("\n" + "=" * 80)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
