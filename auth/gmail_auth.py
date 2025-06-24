from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os, json

def get_gmail_service():

    creds_data = json.loads(os.environ["GMAIL_OAUTH_CREDENTIALS"])
    creds = Credentials(
        token=None,
        refresh_token=creds_data["refresh_token"],
        token_uri=creds_data["token_uri"],
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=[
            "https://www.googleapis.com/auth/gmail.readonly"
        ]
    )
    creds.refresh(Request())
    return build("gmail", "v1", credentials=creds)

def get_drive_service():
    with open("/secrets/GMAIL_OAUTH_CREDENTIALS", "r") as f:
        creds_data = json.load(f)

    creds = Credentials(
        token=None,
        refresh_token=creds_data["refresh_token"],
        token_uri=creds_data["token_uri"],
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=[
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/drive.file"
        ]
    )
    creds.refresh(Request())
    return build("drive", "v3", credentials=creds)