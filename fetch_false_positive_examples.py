#!/usr/bin/env python3
"""
Fetch specific false positive examples to analyze patterns.
"""
import json
from event_feed_app.alerts.alert_payload_store import AlertPayloadStore

# False positive alert IDs from CSV
lawsuit_alerts = [
    "201218de72984717",  # Securities fraud lawsuit
    "a19e43d86efeab74",  # Securities fraud lawsuit
]

market_research_alerts = [
    "916e47ac8e4dac95",  # Market research report
]

payload_store = AlertPayloadStore()

print("=" * 80)
print("SECURITIES FRAUD LAWSUIT FALSE POSITIVES")
print("=" * 80)

for alert_id in lawsuit_alerts:
    print(f"\n{'='*80}")
    print(f"Alert ID: {alert_id}")
    print("=" * 80)
    
    payload = payload_store.get_alert_payload(alert_id, "guidance_change")
    if payload:
        print(f"Company: {payload.get('company', 'N/A')}")
        print(f"Title: {payload.get('title', 'N/A')}")
        print(f"\nFull Text Preview (first 2000 chars):")
        print("-" * 80)
        full_text = payload.get('full_text', '')
        print(full_text[:2000])
        print("-" * 80)
        
        # Look for lawsuit indicators
        text_lower = full_text.lower()
        lawsuit_keywords = [
            'class action', 'lawsuit', 'litigation', 'complaint', 
            'securities fraud', 'plaintiff', 'defendant', 'law firm',
            'rosen law firm', 'kahn swick', 'bragar eagel', 'faruqi'
        ]
        
        found_keywords = [kw for kw in lawsuit_keywords if kw in text_lower]
        if found_keywords:
            print(f"\nLawsuit Keywords Found: {found_keywords}")
    else:
        print("Could not load payload")

print("\n\n")
print("=" * 80)
print("MARKET RESEARCH REPORT FALSE POSITIVES")
print("=" * 80)

for alert_id in market_research_alerts:
    print(f"\n{'='*80}")
    print(f"Alert ID: {alert_id}")
    print("=" * 80)
    
    payload = payload_store.get_alert_payload(alert_id, "guidance_change")
    if payload:
        print(f"Company: {payload.get('company', 'N/A')}")
        print(f"Title: {payload.get('title', 'N/A')}")
        print(f"\nFull Text Preview (first 2000 chars):")
        print("-" * 80)
        full_text = payload.get('full_text', '')
        print(full_text[:2000])
        print("-" * 80)
        
        # Look for market research indicators
        text_lower = full_text.lower()
        research_keywords = [
            'market research', 'market report', 'market forecast',
            'industry report', 'market analysis', 'market intelligence',
            'cagr', 'compound annual growth rate', 'market size',
            'market share', 'market trends', 'forecast period',
            'research and markets', 'technavio', 'mordor intelligence',
            'grand view research', 'fortune business insights'
        ]
        
        found_keywords = [kw for kw in research_keywords if kw in text_lower]
        if found_keywords:
            print(f"\nMarket Research Keywords Found: {found_keywords}")
    else:
        print("Could not load payload")
