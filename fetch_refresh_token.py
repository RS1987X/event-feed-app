from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.cloud import secretmanager

def get_secret(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/YOUR_PROJECT_ID/secrets/{secret_id}/versions/latest"
    return client.access_secret_version(name=name).payload.data.decode('UTF-8')

def get_gmail_service():
    refresh_token = get_secret("GMAIL_REFRESH_TOKEN")
    client_id = get_secret("GMAIL_CLIENT_ID")
    client_secret = get_secret("GMAIL_CLIENT_SECRET")

    creds = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/gmail.readonly"]
    )
    creds.refresh()  # Fetch a fresh access token

    return build("gmail", "v1", credentials=creds)