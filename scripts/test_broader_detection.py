#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# scripts/test_broader_detection.py
"""
Test signal detection on a broader sample to validate before full backfill.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import yaml
from datetime import date
from collections import defaultdict

from event_feed_app.alerts.detector import GuidanceAlertDetector
from event_feed_app.signals.store import SignalStore
from event_feed_app.alerts.runner import fetch_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_broader_sample():
    """Test detection on broader sample without writing signals."""
    
    logger.info("="*80)
    logger.info("TESTING BROADER DETECTION SAMPLE")
    logger.info("="*80)
    
    # Load config
    config_path = Path("src/event_feed_app/configs/significant_events.yaml")
    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)
    
    # Build detector config (no signal writing for this test)
    config = {
        'write_signals': False,
        'alert_all_guidance': False,
        'guidance_plugin_config': yaml_config
    }
    
    # Initialize detector
    detector = GuidanceAlertDetector(config=config)
    
    # Test multiple time periods
    test_periods = [
        ("2025-10-01", "2025-10-07", "Early October"),
        ("2025-10-15", "2025-10-21", "Mid October"),
        ("2025-11-01", "2025-11-07", "Early November"),
        ("2025-11-15", "2025-11-21", "Mid November"),
    ]
    
    all_stats = []
    
    for start, end, label in test_periods:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing: {label} ({start} to {end})")
        logger.info(f"{'='*80}")
        
        # Fetch data
        df = fetch_data(
            start_date=start,
            end_date=end,
            max_rows=500,  # Sample per period
            source=None
        )
        
        logger.info(f"Fetched {len(df)} PRs")
        
        if df.empty:
            logger.warning("No data for this period")
            continue
        
        # Process
        docs = df.to_dict('records')
        alerts = []
        exclusions_count = 0
        
        for doc in docs:
            try:
                doc_alerts = detector.detect_alerts([doc])
                if doc_alerts:
                    alerts.extend(doc_alerts)
            except Exception as e:
                if "exclude" in str(e).lower():
                    exclusions_count += 1
        
        # Stats
        companies = set()
        periods = defaultdict(int)
        metrics = defaultdict(int)
        
        for alert in alerts:
            companies.add(alert.get('company_name', 'unknown'))
            for item in alert.get('guidance_items', []):
                cand = item.get('candidate', {})
                periods[cand.get('period', 'unknown')] += 1
                metrics[cand.get('metric', 'unknown')] += 1
        
        stats = {
            'period': label,
            'prs': len(df),
            'alerts': len(alerts),
            'companies': len(companies),
            'detection_rate': f"{100*len(alerts)/len(df):.2f}%" if df.shape[0] > 0 else "0%",
        }
        all_stats.append(stats)
        
        logger.info(f"\n📊 Results for {label}:")
        logger.info(f"  PRs processed: {len(df)}")
        logger.info(f"  Alerts generated: {len(alerts)}")
        logger.info(f"  Unique companies: {len(companies)}")
        logger.info(f"  Detection rate: {stats['detection_rate']}")
        
        if periods:
            logger.info(f"  Periods detected: {dict(periods)}")
        if metrics:
            logger.info(f"  Metrics detected: {dict(metrics)}")
        
        # Show sample alerts
        if alerts:
            logger.info(f"\n  Sample alerts:")
            for alert in alerts[:3]:
                logger.info(f"    - {alert.get('company_name')}: {alert.get('alert_id')[:12]}")
    
    # Overall summary
    logger.info(f"\n{'='*80}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'='*80}")
    
    total_prs = sum(s['prs'] for s in all_stats)
    total_alerts = sum(s['alerts'] for s in all_stats)
    
    logger.info(f"Total PRs tested: {total_prs}")
    logger.info(f"Total alerts: {total_alerts}")
    logger.info(f"Overall detection rate: {100*total_alerts/total_prs:.2f}%" if total_prs > 0 else "0%")
    
    logger.info(f"\nPer-period breakdown:")
    for stat in all_stats:
        logger.info(f"  {stat['period']:20} {stat['alerts']:4} alerts / {stat['prs']:4} PRs = {stat['detection_rate']:6}")
    
    # Assessment
    logger.info(f"\n{'='*80}")
    logger.info("ASSESSMENT")
    logger.info(f"{'='*80}")
    
    if total_alerts == 0:
        logger.warning("⚠️  NO ALERTS DETECTED - Detection may be too strict!")
        logger.info("   Recommendations:")
        logger.info("   1. Check exclusion rules aren't too broad")
        logger.info("   2. Review plugin detection patterns")
        logger.info("   3. Manually inspect some PRs for expected guidance")
    elif total_alerts < total_prs * 0.01:
        logger.warning(f"⚠️  LOW DETECTION RATE ({100*total_alerts/total_prs:.2f}%) - May be too conservative")
        logger.info("   Consider reviewing detection thresholds")
    elif total_alerts > total_prs * 0.1:
        logger.warning(f"⚠️  HIGH DETECTION RATE ({100*total_alerts/total_prs:.2f}%) - May be too permissive")
        logger.info("   Consider tightening exclusions or detection criteria")
    else:
        logger.info(f"✅ Detection rate looks reasonable ({100*total_alerts/total_prs:.2f}%)")
        logger.info("   Ready to proceed with backfill if patterns look correct")
    
    logger.info(f"{'='*80}\n")
    
    return total_alerts > 0


if __name__ == "__main__":
    success = test_broader_sample()
    sys.exit(0 if success else 1)
