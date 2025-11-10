#!/usr/bin/env python3
"""
Helper script to get Telegram chat IDs for users.

Usage (three options):
  A) Pass token explicitly:
     python3 scripts/telegram_get_chat_id.py 123456789:ABCdef...
  B) Put TELEGRAM_BOT_TOKEN in .env then just run:
     python3 scripts/telegram_get_chat_id.py
  C) Set export TELEGRAM_BOT_TOKEN=... in shell then run script

Prerequisites:
  1. Create a bot with @BotFather (copy the token)
  2. Users send /start to your bot (so updates appear)
  3. Run this script to list chat IDs
"""
import requests
import sys
import os
import time
from pathlib import Path


def _load_env_if_present():
    """Load .env manually (avoid extra dependency) if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        with env_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                val = val.strip().strip('"').strip("'")
                # Don't overwrite an existing env var
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass  # Silent fail; not critical


def check_token(bot_token: str, timeout: float = 8.0):
    """Validate token using getMe (no updates required). Returns (ok, data)."""
    url = f"https://api.telegram.org/bot{bot_token}/getMe"
    r = requests.get(url, timeout=(3.0, timeout))
    r.raise_for_status()
    data = r.json()
    return bool(data.get("ok")), data


def get_chat_ids(bot_token: str, timeout: float = 12.0, retries: int = 2):
    """Fetch recent updates and print chat IDs."""
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"

    try:
        last_exc = None
        for attempt in range(retries + 1):
            try:
                response = requests.get(url, timeout=(4.0, timeout))
                response.raise_for_status()
                break
            except requests.RequestException as e:
                last_exc = e
                if attempt < retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise
        data = response.json()

        if not data.get("ok"):
            print(f"Error from Telegram API: {data.get('description', 'Unknown error')}")
            print("• Double‑check the bot token. If regenerated, update .env.")
            return

        updates = data.get("result", [])

        if not updates:
            print("No messages found.")
            print("→ Have users (including you) send /start to the bot in Telegram.")
            print("→ Optionally send a test message so it appears in getUpdates.")
            return

        print("\n=== Chat IDs ===\n")
        seen_chats = set()

        for update in updates:
            message = update.get("message", {}) or update.get("edited_message", {})
            chat = message.get("chat", {})
            chat_id = chat.get("id")

            if chat_id and chat_id not in seen_chats:
                seen_chats.add(chat_id)
                first_name = chat.get("first_name", "")
                last_name = chat.get("last_name", "")
                username = chat.get("username", "")

                name_parts = [p for p in [first_name, last_name] if p]
                name = " ".join(name_parts) if name_parts else (username or "Unknown")

                print(f"Chat ID: {chat_id}")
                print(f"  Name: {name}")
                if username:
                    print(f"  Username: @{username}")
                print()

        print(f"Found {len(seen_chats)} chat(s).")
        print("\nUse these chat_ids in user preferences (field: telegram_chat_id).")

    except requests.RequestException as e:
        print(f"Network/API request failed: {e}")
        print("• Check internet connectivity.")
        print("• Verify token is correct (format: <digits>:<alphanumeric>).")
    except Exception as e:
        print(f"Unexpected error: {e}")


def resolve_token(argv) -> str | None:
    """Resolve bot token from CLI arg, env var, or None."""
    if len(argv) >= 2 and argv[1].strip():
        return argv[1].strip()
    # Try env var (after loading .env)
    _load_env_if_present()
    env_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if env_token:
        return env_token.strip()
    return None


def print_usage():
    print(__doc__)
    print("\nExamples:")
    print("  python3 scripts/telegram_get_chat_id.py 123456789:ABCdef...")
    print("  TELEGRAM_BOT_TOKEN=123456789:ABCdef... python3 scripts/telegram_get_chat_id.py")
    print("  (Put token in .env then) python3 scripts/telegram_get_chat_id.py")


if __name__ == "__main__":
    token = resolve_token(sys.argv)
    if not token:
        print("⚠️  No bot token provided or found in environment.")
        print_usage()
        sys.exit(1)

    if ":" not in token:
        print("⚠️  Token format looks wrong (missing ':').")
        print_usage()
        sys.exit(1)

    print("✓ Using bot token (masked):", token.split(":", 1)[0] + ":********")
    try:
        ok, data = check_token(token)
        if ok:
            user = data.get("result", {})
            uname = user.get("username") or "(no username)"
            print("✓ Token validated via getMe")
            print(f"  Bot: {user.get('first_name', 'Bot')} @" + str(uname))
        else:
            print("⚠️  Token did not validate (getMe returned ok=false)")
    except requests.RequestException as e:
        print(f"Network/API request failed during token validation: {e}")
        print("• If this keeps happening, try a different network or VPN.")
    except Exception as e:
        print(f"Unexpected error during token validation: {e}")

    get_chat_ids(token)
