#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# scripts/alert_cli.py
"""
Command-line tool for managing alerts and user preferences.
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from event_feed_app.alerts.store import AlertStore


def main():
    parser = argparse.ArgumentParser(description="Manage alert system")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List alerts
    list_parser = subparsers.add_parser("list", help="List recent alerts")
    list_parser.add_argument("--limit", type=int, default=10, help="Number of alerts to show")
    
    # Show user preferences
    user_parser = subparsers.add_parser("user", help="Manage user preferences")
    user_parser.add_argument("user_id", help="User ID (email)")
    user_parser.add_argument("--show", action="store_true", help="Show preferences")
    user_parser.add_argument("--email", help="Set email address")
    user_parser.add_argument("--watchlist", help="Comma-separated company names")
    user_parser.add_argument("--min-significance", type=float, help="Min significance (0-1)")
    user_parser.add_argument("--channels", help="Comma-separated delivery channels")
    user_parser.add_argument("--webhook", help="Webhook URL")
    user_parser.add_argument("--active", type=bool, help="Active status")
    
    # List all users
    users_parser = subparsers.add_parser("users", help="List all users")
    
    # Test alert delivery
    test_parser = subparsers.add_parser("test", help="Send test alert")
    test_parser.add_argument("user_id", help="User ID to send test alert to")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    store = AlertStore()
    
    if args.command == "list":
        list_alerts(store, args.limit)
    elif args.command == "user":
        manage_user(store, args)
    elif args.command == "users":
        list_users(store)
    elif args.command == "test":
        send_test_alert(store, args.user_id)


def list_alerts(store: AlertStore, limit: int):
    """List recent alerts."""
    alerts = store.get_recent_alerts(limit=limit)
    
    if not alerts:
        print("No alerts found")
        return
    
    print(f"\n{'=' * 80}")
    print(f"Recent Alerts ({len(alerts)})")
    print(f"{'=' * 80}\n")
    
    for alert in alerts:
        print(f"ID:      {alert['alert_id']}")
        print(f"Type:    {alert['alert_type']}")
        print(f"Company: {alert['company_name']}")
        print(f"Score:   {alert['significance_score']:.3f}")
        print(f"Time:    {alert['detected_at']}")
        print(f"Summary: {alert['summary']}")
        
        if alert.get('metrics'):
            print("Metrics:")
            for metric in alert['metrics']:
                print(f"  - {metric.get('metric')}: {metric.get('direction', 'N/A')}")
        
        print(f"{'-' * 80}")


def manage_user(store: AlertStore, args):
    """Manage user preferences."""
    if args.show:
        prefs = store.get_user_preferences(args.user_id)
        if prefs:
            print(json.dumps(prefs, indent=2))
        else:
            print(f"No preferences found for user {args.user_id}")
        return
    
    # Update preferences
    existing = store.get_user_preferences(args.user_id) or {}
    
    prefs = {
        "email_address": args.email or existing.get("email_address", args.user_id),
        "watchlist": args.watchlist.split(",") if args.watchlist else existing.get("watchlist", []),
        "alert_types": existing.get("alert_types", ["guidance_change"]),
        "min_significance": args.min_significance if args.min_significance is not None else existing.get("min_significance", 0.7),
        "delivery_channels": args.channels.split(",") if args.channels else existing.get("delivery_channels", ["email"]),
        "webhook_url": args.webhook or existing.get("webhook_url", ""),
        "active": args.active if args.active is not None else existing.get("active", True),
    }
    
    store.save_user_preferences(args.user_id, prefs)
    print(f"✓ Preferences saved for {args.user_id}")
    print(json.dumps(prefs, indent=2))


def list_users(store: AlertStore):
    """List all active users."""
    users = store.get_all_active_users()
    
    if not users:
        print("No active users found")
        return
    
    print(f"\n{'=' * 80}")
    print(f"Active Users ({len(users)})")
    print(f"{'=' * 80}\n")
    
    for user in users:
        print(f"User ID:  {user['user_id']}")
        print(f"Email:    {user.get('email_address', 'N/A')}")
        print(f"Watchlist: {', '.join(user.get('watchlist', [])) or 'All companies'}")
        print(f"Channels: {', '.join(user.get('delivery_channels', []))}")
        print(f"Min Score: {user.get('min_significance', 0.7)}")
        print(f"{'-' * 80}")


def send_test_alert(store: AlertStore, user_id: str):
    """Send a test alert."""
    from event_feed_app.alerts.delivery import AlertDelivery
    from datetime import datetime, timezone
    
    # Check if user exists
    user = store.get_user_preferences(user_id)
    if not user:
        print(f"Error: User {user_id} not found. Create preferences first.")
        return
    
    # Create test alert
    test_alert = {
        "alert_id": "test_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "event_id": "test_event",
        "alert_type": "guidance_change",
        "company_name": "Test Company Inc.",
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "significance_score": 0.85,
        "summary": "Revenue guidance raised for FY2025",
        "metrics": [
            {
                "metric": "revenue",
                "direction": "up",
                "range": "10-15%",
                "unit": "%",
            }
        ],
        "metadata": {
            "press_release_url": "https://example.com/press-release",
            "period": "FY2025",
        }
    }
    
    # Deliver
    delivery = AlertDelivery(store=store)
    
    try:
        delivery.deliver(test_alert, [user])
        print(f"✓ Test alert sent to {user_id}")
        print(f"  Channels: {', '.join(user.get('delivery_channels', []))}")
    except Exception as e:
        print(f"✗ Failed to send test alert: {e}")


if __name__ == "__main__":
    main()
