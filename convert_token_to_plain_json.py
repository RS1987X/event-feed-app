# convert_token_to_plain_json.py
import json
import os

# 1) Load your token.json from disk.  
#    If it was pickled, uncomment the pickle code;  
#    if it was written via creds.to_json(), use the JSON loader.

USE_PICKLE = True

if USE_PICKLE:
    import pickle
    with open("token.json","rb") as f:
        creds = pickle.load(f)
else:
    # from google.oauth2.credentials import Credentials
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    # If it was saved via creds.to_json() you can rehydrate:
    creds = Credentials.from_authorized_user_file("token.json", scopes=["https://www.googleapis.com/auth/gmail.readonly"])

# 2) Do a sanity check that we have what we need:
assert creds.refresh_token, "No refresh_token found in token.json!"
assert creds.client_id and creds.client_secret and creds.token_uri

# 3) Assemble the minimal JSON
plain = {
    "client_id":     creds.client_id,
    "client_secret": creds.client_secret,
    "refresh_token": creds.refresh_token,
    "token_uri":     creds.token_uri,
}

# 4) Write it out as UTF-8 JSON
with open("gmail_credentials.json","w", encoding="utf-8") as f:
    json.dump(plain, f, indent=2)

print("✅ Wrote gmail_credentials.json with keys:", list(plain.keys()))