# sources/gmail_fetcher.py

from auth.gmail_auth import get_gmail_service
from core.event_types import Event
from datetime import datetime
import base64
import os
import re

def fetch_unread_emails(max_results=10):
    service = get_gmail_service()
    results = service.users().messages().list(
        userId='me',
        labelIds=['INBOX'],
        q='is:unread',
        maxResults=max_results
    ).execute()

    messages = results.get('messages', [])
    events = []

    for msg in messages:
        full_msg = service.users().messages().get(userId='me', id=msg['id']).execute()
        payload = full_msg.get('payload', {})
        headers = payload.get('headers', [])

        subject = _extract_header(headers, 'Subject') or "(No Subject)"
        sender = _extract_header(headers, 'From') or "Unknown Sender"
        timestamp = int(full_msg['internalDate']) / 1000
        dt = datetime.fromtimestamp(timestamp)

        body = _extract_email_body(payload)

        event = Event(
            source="Gmail",
            title=subject,
            timestamp=dt,
            content=f"From: {sender}\n\n{body}",
            metadata={"id": msg['id']}
        )
        events.append(event)

    return events


def test_fetch_any_email(max_results=5):
    service = get_gmail_service()

    results = service.users().messages().list(
        userId='me',
        maxResults=max_results
    ).execute()

    messages = results.get('messages', [])

    print(f"✅ Found {len(messages)} messages")

    for msg in messages:
        full_msg = service.users().messages().get(userId='me', id=msg['id']).execute()
        payload = full_msg.get('payload', {})
        headers = payload.get('headers', [])

        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), "(No Subject)")
        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), "Unknown")

        timestamp = int(full_msg['internalDate']) / 1000
        dt = datetime.fromtimestamp(timestamp)

        snippet = full_msg.get('snippet', '')

        print(f"\n📧 [{dt}] {subject}")
        print(f"From: {sender}")
        print(f"Snippet: {snippet}")

        
def _extract_header(headers, name):
    for h in headers:
        if h['name'].lower() == name.lower():
            return h['value']
    return None


def _extract_email_body(payload):
    if payload.get('body', {}).get('data'):
        return _decode_base64(payload['body']['data'])

    for part in payload.get('parts', []):
        if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
            return _decode_base64(part['body']['data'])

    return "(No plain text body found)"


def _decode_base64(data):
    data = data.replace('-', '+').replace('_', '/')
    padded = data + '=' * (-len(data) % 4)
    return base64.b64decode(padded).decode('utf-8', errors='ignore')
