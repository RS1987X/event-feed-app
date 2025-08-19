from gcr_job_main import gmail_creds_from_existing_secrets
from googleapiclient.discovery import build
import os
creds = gmail_creds_from_existing_secrets()
svc = build("gmail", "v1", credentials=creds, cache_discovery=False)

resp = svc.users().messages().list(userId="me", q="").execute()
print("Message count:", len(resp.get("messages", [])))