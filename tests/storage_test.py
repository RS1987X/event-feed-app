from google.cloud import storage

PROJECT_ID = "event-feed-app-463206"
BUCKET = "event-feed-app-data"

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET)
blob = bucket.blob("smoke_test/hello.txt")
blob.upload_from_string("hello gcs")
print("Uploaded to GCS")

print("Downloaded:", blob.download_as_text())