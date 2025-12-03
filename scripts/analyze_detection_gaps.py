"""
Quick analysis of what's being missed or wrongly detected.

Usage: python scripts/analyze_detection_gaps.py
"""
import pandas as pd
import yaml
from event_feed_app.events.guidance_change.plugin import GuidanceChangePlugin
from event_feed_app.loaders.gcs import load_press_release
from collections import Counter


def analyze_gaps():
    # Load config
    with open("src/event_feed_app/configs/significant_events.yaml") as f:
        cfg = yaml.safe_load(f)
    
    plugin = GuidanceChangePlugin()
    plugin.configure(cfg)
    
    # Load labeled data
    tps = pd.read_csv("guidance_true_positives.csv")
    fps = pd.read_csv("guidance_false_positives.csv")
    
    print("=== ANALYZING DETECTION GAPS ===\n")
    
    # Analyze missed TPs
    print("1. TRUE POSITIVES WE'RE MISSING:\n")
    missed_patterns = []
    
    for idx, row in tps.iterrows():
        pr_id = row['press_release_id']
        try:
            doc = load_press_release(pr_id)
            results = list(plugin.detect(doc))
            
            if len(results) == 0:
                # Missed it - analyze why
                title = doc.get('title', '')
                body_snippet = doc.get('body', '')[:200]
                
                print(f"  PR {pr_id}:")
                print(f"    Title: {title}")
                print(f"    Body: {body_snippet}...")
                
                # Look for common patterns
                text = title.lower() + " " + body_snippet.lower()
                if 'guidance' in text or 'outlook' in text or 'forecast' in text:
                    if 'raise' in text or 'rais' in text:
                        missed_patterns.append('raise_guidance')
                    elif 'lower' in text or 'cut' in text or 'reduc' in text:
                        missed_patterns.append('lower_guidance')
                    elif 'reaffirm' in text or 'maintain' in text or 'unchanged' in text:
                        missed_patterns.append('reaffirm_guidance')
                    elif 'initiate' in text or 'provide' in text or 'issue' in text:
                        missed_patterns.append('initiate_guidance')
                    else:
                        missed_patterns.append('other_guidance')
                
                print()
        except Exception as e:
            print(f"  ERROR loading PR {pr_id}: {e}")
    
    if missed_patterns:
        print(f"\nMissed pattern frequency:")
        for pattern, count in Counter(missed_patterns).most_common():
            print(f"  {pattern}: {count}x")
    
    # Analyze FPs
    print("\n\n2. FALSE POSITIVES WE'RE DETECTING:\n")
    fp_patterns = []
    
    for idx, row in fps.iterrows():
        pr_id = row['press_release_id']
        try:
            doc = load_press_release(pr_id)
            results = list(plugin.detect(doc))
            
            if len(results) > 0:
                # False positive - analyze why
                title = doc.get('title', '')
                body_snippet = doc.get('body', '')[:200]
                
                print(f"  PR {pr_id} ({len(results)} candidates):")
                print(f"    Title: {title}")
                print(f"    Trigger: {results[0].get('_trigger_source')}")
                if results[0].get('_trigger_match'):
                    print(f"    Match: {results[0].get('_trigger_match')}")
                
                # Categorize FP type
                text = title.lower() + " " + body_snippet.lower()
                if 'earnings call' in text or 'webcast' in text or 'conference call' in text:
                    fp_patterns.append('ir_announcement')
                elif 'analyst' in text or 'coverage' in text:
                    fp_patterns.append('analyst_report')
                elif 'historical' in text or 'reported' in text or 'was' in text[:100]:
                    fp_patterns.append('historical_statement')
                else:
                    fp_patterns.append('other_fp')
                
                print()
        except Exception as e:
            print(f"  ERROR loading PR {pr_id}: {e}")
    
    if fp_patterns:
        print(f"\nFalse positive pattern frequency:")
        for pattern, count in Counter(fp_patterns).most_common():
            print(f"  {pattern}: {count}x")
    
    print("\n\n=== RECOMMENDATIONS ===\n")
    
    if 'raise_guidance' in missed_patterns:
        print("• Add more 'raise' verb variations to proximity rules")
    if 'reaffirm_guidance' in missed_patterns:
        print("• Check 'unchanged' marker rules and verb proximity")
    if 'ir_announcement' in fp_patterns:
        print("• Strengthen IR announcement suppression (already added 'enable_text_stubs: false')")
    if 'analyst_report' in fp_patterns:
        print("• Add analyst report suppressors (check for source_type != 'issuer_pr')")
    if 'historical_statement' in fp_patterns:
        print("• Add past-tense detection to suppress historical statements")


if __name__ == "__main__":
    analyze_gaps()
