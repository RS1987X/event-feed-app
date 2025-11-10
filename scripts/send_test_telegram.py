#!/usr/bin/env python3
"""
Send a test Telegram alert message to a chat.

Usage options:
  A) Put TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env then run:
       python3 scripts/send_test_telegram.py
  B) Pass chat id explicitly (token from .env/env):
       python3 scripts/send_test_telegram.py 123456789
  C) Pass both token and chat id:
       python3 scripts/send_test_telegram.py 123456789:ABCdef... 123456789

Message contents mimic a guidance alert formatting.
"""
import os
import sys
import requests
from pathlib import Path
from datetime import datetime

ENV_PATH = Path(__file__).resolve().parent.parent / '.env'


def load_env():
    if not ENV_PATH.exists():
        return
    try:
        with ENV_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def resolve_args(argv):
    """Return (token, chat_id)."""
    token = None
    chat_id = None

    if len(argv) >= 3:
        token = argv[1].strip()
        chat_id = argv[2].strip()
    elif len(argv) == 2:
        # Could be chat id only or token only; decide by presence of ':'
        if ':' in argv[1]:
            token = argv[1].strip()
        else:
            chat_id = argv[1].strip()

    if not token:
        token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not chat_id:
        chat_id = os.getenv('TELEGRAM_CHAT_ID')

    return token, chat_id


def validate_token(token: str) -> bool:
    try:
        r = requests.get(f'https://api.telegram.org/bot{token}/getMe', timeout=(3, 8))
        r.raise_for_status()
        data = r.json()
        return bool(data.get('ok'))
    except Exception:
        return False


def send_message(token: str, chat_id: str, text: str):
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': False,
    }
    r = requests.post(url, json=payload, timeout=(4, 12))
    r.raise_for_status()
    return r.json()


def build_test_message() -> str:
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    lines = [
        '🔔 *Test Guidance Alert*',
        '',
        '📈 *Revenue*: $1.2B ↑20% (prior $1.0B)',
        '📉 *EBITDA*: $300M ↓5% (prior $315M)',
        '',
        f'🕒 Sent: `{now}`',
        '⭐ Significance: *85/100* (demo)',
        '',
        '[📄 Example Press Release](https://example.com/pr)',
    ]
    return '\n'.join(lines)


def main(argv):
    load_env()
    token, chat_id = resolve_args(argv)

    if not token:
        print('⚠️  Missing bot token. Provide as arg or TELEGRAM_BOT_TOKEN in .env')
        sys.exit(1)
    if ':' not in token:
        print('⚠️  Bot token format invalid (missing colon).')
        sys.exit(1)
    if not chat_id:
        # Interactive prompt fallback to make first-run easier
        print('⚠️  Missing chat id. Provide as arg or TELEGRAM_CHAT_ID in .env')
        entered = input('Enter your Telegram chat id (numeric) and press Enter, or leave blank to abort: ').strip()
        if entered:
            chat_id = entered
        else:
            print('Tip: Run `python3 scripts/telegram_get_chat_id.py` after sending /start to your bot to discover your chat id.')
            sys.exit(1)

    print('✓ Using bot token (masked):', token.split(':', 1)[0] + ':********')
    print('✓ Target chat id:', chat_id)

    if not validate_token(token):
        print('⚠️  Token failed validation via getMe.')
        sys.exit(1)
    print('✓ Token validated')

    msg = build_test_message()
    try:
        resp = send_message(token, chat_id, msg)
        if resp.get('ok'):
            print('✅ Test message sent successfully.')
        else:
            print('⚠️  Telegram API responded with ok=false:', resp)
    except requests.RequestException as e:
        print('❌ Failed to send message:', e)
        print('• Check chat id (must have /start chat with bot)')
        print('• Ensure bot is not blocked and token is correct')
    sys.exit(1)


if __name__ == '__main__':
    main(sys.argv)
