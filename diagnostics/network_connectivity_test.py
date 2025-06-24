import requests

try:
    response = requests.get("https://gmail.com", timeout=5)
    print(f"HTTP status: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Network error: {e}")