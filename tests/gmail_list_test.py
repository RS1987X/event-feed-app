# tests/gmail_list_test.py
import json, os
from google.cloud import secretmanager
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

PROJECT_ID   = os.environ.get("PROJECT_ID", "event-feed-app-463206")
GMAIL_LABEL  = os.environ.get("GMAIL_LABEL", "INBOX")

def _secret_text(name: str) -> str:
    sm = secretmanager.SecretManagerServiceClient()
    rn = f"projects/{PROJECT_ID}/secrets/{name}/versions/latest"
    return sm.access_secret_version(request={"name": rn}).payload.data.decode()

def gmail_creds_from_existing_secrets() -> Credentials:
    client_json = json.loads(_secret_text("GMAIL_OAUTH_JSON"))
    client_blob = client_json.get("installed") or client_json.get("web") or {}
    token_json  = json.loads(_secret_text("GMAIL_OAUTH_CREDENTIALS"))
    return Credentials(
        None,
        refresh_token=token_json["refresh_token"],
        token_uri=token_json.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=client_blob["client_id"],
        client_secret=client_blob["client_secret"],
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
    )

def main():
    creds = gmail_creds_from_existing_secrets()
    svc = build("gmail", "v1", credentials=creds, cache_discovery=False)

    # Resolve the label name to its ID (label queries accept either, but IDs are robust)
    labels = svc.users().labels().list(userId="me").execute().get("labels", [])
    label = next((l for l in labels if l["name"] == GMAIL_LABEL), None)
    if not label:
        print(f"Label '{GMAIL_LABEL}' not found. Create it in Gmail or set GMAIL_LABEL to an existing one.")
        print("Available labels:", [l["name"] for l in labels])
        return

    print(f"Using label '{label['name']}' (id: {label['id']})")

    resp = svc.users().messages().list(userId="me", labelIds=[label["id"]], maxResults=100).execute()
    ids = [m["id"] for m in resp.get("messages", [])]
    print(f"Found {len(ids)} message(s) under label '{GMAIL_LABEL}':", ids)

if __name__ == "__main__":
    main()