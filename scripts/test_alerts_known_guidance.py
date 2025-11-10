#!/usr/bin/env python3
"""
Test alert system with known guidance events from cache.
Fetches specific docs that plugin2 has previously identified as guidance-related.
"""
import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from load_env import load_env
load_env()

import pandas as pd
import logging

from event_feed_app.config import Settings
from event_feed_app.utils.io import load_from_gcs
from event_feed_app.alerts.orchestration import AlertOrchestrator
from event_feed_app.alerts.store import AlertStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Known guidance doc_ids from .cache/guidance_events.csv
KNOWN_GUIDANCE_DOCS = [
    "197efa8c4b16d384", "1980c9dc5960fd2a", "1980c93f9f819a6d", "1980cad3e0410e27",
    "1980c905e9d00dfa", "19816c15b06d67c9", "1981be815a0d0e4f", "19830b83edc2fb19",
    "198838b9adf96ee2", "198a44959e94f27e", "198a44eda6fdf4db", "198a92bb04fe9f70",
    "198a92fa66e28c9d", "198a92facd31b7f5", "198a9300fde362bc", "198a92de191a51ab",
]


def main():
    logger.info("="*60)
    logger.info("Testing Alerts with Known Guidance Events")
    logger.info("="*60)
    
    # Load full silver data
    cfg = Settings()
    logger.info(f"Loading from: {cfg.gcs_silver_root}")
    df = load_from_gcs(cfg.gcs_silver_root)
    logger.info(f"Loaded {len(df)} total records")
    
    # Filter to known guidance docs
    if 'press_release_id' in df.columns:
        df_guidance = df[df['press_release_id'].isin(KNOWN_GUIDANCE_DOCS)].copy()
    elif 'doc_id' in df.columns:
        df_guidance = df[df['doc_id'].isin(KNOWN_GUIDANCE_DOCS)].copy()
    else:
        logger.error("No press_release_id or doc_id column found")
        return
    
    logger.info(f"Found {len(df_guidance)} matching guidance docs")
    
    if len(df_guidance) == 0:
        logger.warning("No matching docs found. Cache might be stale or column names don't match.")
        return
    
    # Ensure clean columns exist
    if "title_clean" not in df_guidance.columns:
        df_guidance["title_clean"] = df_guidance["title"].fillna("").astype(str)
    if "full_text_clean" not in df_guidance.columns:
        df_guidance["full_text_clean"] = df_guidance["full_text"].fillna("").astype(str)
    
    # Load plugin config
    import yaml
    config_path = project_root / "src" / "event_feed_app" / "configs" / "significant_events.yaml"
    with open(config_path) as f:
        yaml_cfg = yaml.safe_load(f)
    
    events = yaml_cfg.get("events", [])
    guidance_cfg = next((e for e in events if e.get("key") == "guidance_change"), {})
    
    # Build config with alert_all enabled
    config = {
        "min_significance": 0.3,
        "alert_all_guidance": True,  # Alert on any guidance commentary
        "guidance_plugin_config": guidance_cfg,
        "smtp": {},
        "telegram": {"bot_token": os.getenv("TELEGRAM_BOT_TOKEN")}
    }
    
    # Setup Telegram user if env provided
    store = AlertStore()
    tg_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if tg_chat_id:
        user_id = os.getenv("ALERT_USER_ID", f"tg:{tg_chat_id}")
        prefs = {
            "user_id": user_id,
            "telegram_chat_id": tg_chat_id,
            "email_address": os.getenv("ALERT_EMAIL", ""),
            "alert_types": ["guidance_change"],
            "min_significance": 0.0,  # Allow all with alert_all
            "delivery_channels": ["telegram"],
            "active": True,
        }
        store.save_user_preferences(user_id, prefs)
        logger.info(f"Configured Telegram user: {user_id}")
    
    # Run alerts
    orchestrator = AlertOrchestrator(config=config, enable_document_dedup=False)
    
    logger.info(f"\nProcessing {len(df_guidance)} known guidance docs...")
    docs = df_guidance.to_dict("records")
    stats = orchestrator.process_documents(docs)
    
    logger.info("\n" + "="*60)
    logger.info("Results:")
    logger.info(f"  Documents processed: {stats['docs_processed']}")
    logger.info(f"  Alerts detected: {stats['alerts_detected']}")
    logger.info(f"  Alerts delivered: {stats['alerts_delivered']}")
    logger.info(f"  Delivery failures: {stats['delivery_failures']}")
    logger.info("="*60)
    
    if stats['alerts_delivered'] > 0:
        logger.info("\n✓ Check your Telegram for alerts!")
    elif stats['alerts_detected'] > 0:
        logger.info("\n⚠ Alerts detected but not delivered (check user preferences)")
    else:
        logger.warning("\n⚠ No alerts generated from known guidance docs - investigate detector/plugin")


if __name__ == "__main__":
    main()
