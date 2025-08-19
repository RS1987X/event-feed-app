from google.cloud import secretmanager

PROJECT_ID = "event-feed-app-463206"
client = secretmanager.SecretManagerServiceClient()

name = f"projects/{PROJECT_ID}/secrets/GMAIL_OAUTH_JSON/versions/latest"
response = client.access_secret_version(request={"name": name})
print("Got secret:", response.payload.data.decode("utf-8")[:200], "...")