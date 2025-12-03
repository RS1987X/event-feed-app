#!/usr/bin/env python3
"""
Analyze period detection issues - show WHERE in the text periods are found.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
import re
from collections import defaultdict

# Download a few guidance alert payloads and examine them
def analyze_periods():
    import json
    import subprocess
    
    print("Downloading sample guidance alerts from GCS...")
    result = subprocess.run([
        'gsutil', 'cat', 
        'gs://event-feed-app-data/feedback/alert_payloads/signal_type=guidance_change/2025-11-24.jsonl'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error downloading: {result.stderr}")
        return
    
    lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
    print(f"Found {len(lines)} alerts\n")
    
    # Period extraction patterns (from plugin.py)
    PERIOD_REGEX = re.compile(
        r"\b((FY|FY-)?(?P<y>20\d{2})|(Q(?P<q>[1-4])[-\s]?(?P<y2>20\d{2}))|(H(?P<h>[12])[-\s]?(?P<y3>20\d{2}))|full[-\s]?year|helår|kvartal\s?Q?[1-4])\b",
        re.I
    )
    
    period_issues = defaultdict(list)
    
    for i, line in enumerate(lines[:20], 1):  # First 20 alerts
        data = json.loads(line)
        
        body = data.get('metadata', {}).get('body', '')
        title = data.get('metadata', {}).get('title', '')
        period = data.get('metadata', {}).get('period', 'N/A')
        company = data.get('company_name', 'Unknown')
        
        # Find all period mentions in the full text
        full_text = title + "\n" + body
        all_matches = list(PERIOD_REGEX.finditer(full_text))
        
        if all_matches:
            print(f"\n{'='*80}")
            print(f"Alert {i}: {company}")
            print(f"Detected period: {period}")
            print(f"Release date: {data.get('metadata', {}).get('release_date')}")
            print(f"\nAll period mentions found ({len(all_matches)}):")
            
            for j, m in enumerate(all_matches[:10], 1):  # Show first 10
                year = m.group('y') or m.group('y2') or m.group('y3')
                context_start = max(0, m.start() - 50)
                context_end = min(len(full_text), m.end() + 50)
                context = full_text[context_start:context_end].replace('\n', ' ')
                
                print(f"  {j}. Year={year}, Match='{m.group(0)}' in context: ...{context}...")
                
                # Flag suspicious years
                if year and int(year) < 2024:
                    period_issues['old_years'].append({
                        'company': company,
                        'year': year,
                        'period': period,
                        'context': context
                    })
            
            # Show which one was actually selected
            print(f"\n  ✓ Selected period: {period}")
            if period.startswith('FY') and len(period) > 2:
                year_in_period = period.replace('FY', '').replace('-UNKNOWN', '')
                if year_in_period.isdigit() and int(year_in_period) < 2024:
                    print(f"  ⚠️  WARNING: Selected year {year_in_period} is before 2024 (PR from Nov 2025!)")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"\nISSUES FOUND:")
    print(f"  Old years (< 2024): {len(period_issues['old_years'])} instances")
    
    if period_issues['old_years']:
        print(f"\n  Examples of old year detection:")
        for item in period_issues['old_years'][:5]:
            print(f"    - {item['company']}: detected {item['year']}, period={item['period']}")
            print(f"      Context: {item['context'][:100]}...")

if __name__ == "__main__":
    analyze_periods()
