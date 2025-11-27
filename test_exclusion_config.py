#!/usr/bin/env python3
"""Test that exclusion config loads correctly."""
import yaml
from event_feed_app.events.guidance_change.plugin import GuidanceChangePlugin

# Load config
with open("src/event_feed_app/configs/significant_events.yaml") as f:
    cfg = yaml.safe_load(f)

# Initialize and configure plugin
plugin = GuidanceChangePlugin()
plugin.configure(cfg)

print("=" * 80)
print("LOADED EXCLUSION RULES")
print("=" * 80)
print()

for exc_name, rule in plugin.exclusion_rules.items():
    print(f"\n{exc_name}:")
    print(f"  Indicators: {rule.get('indicators', [])}")
    print(f"  Secondary: {rule.get('secondary', [])}")
    print(f"  Logic: {rule.get('logic', 'N/A')}")

print("\n" + "=" * 80)
print("TESTING EXCLUSION LOGIC")
print("=" * 80)

# Test lawsuit
lawsuit_text = """class action lawsuit against Molina Healthcare by the rosen law firm"""
print(f"\nTest text: {lawsuit_text}")
print(f"Should exclude: {plugin._should_exclude(lawsuit_text)}")

# Test market research
market_text = """market report from verified market research shows cagr of 7.85%"""
print(f"\nTest text: {market_text}")
print(f"Should exclude: {plugin._should_exclude(market_text)}")

# Test normal
normal_text = """the company raised its revenue guidance for the full year"""
print(f"\nTest text: {normal_text}")
print(f"Should exclude: {plugin._should_exclude(normal_text)}")
