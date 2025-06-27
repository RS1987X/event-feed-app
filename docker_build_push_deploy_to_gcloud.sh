#!/usr/bin/env bash
set -euo pipefail

# === CONFIGURATION ===
PROJECT_ID="event-feed-app-463206"
IMAGE_NAME="event-feed-app"
TAG="latest"

# === 1) Build (no cache, always pull base) ===
docker build \
  --no-cache \
  --pull \
  -t "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}" \
  .

# === 2) Push to GCR ===
docker push "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "✅ gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG} built & pushed successfully"

# === 3) Deploy to Cloud Run ===
gcloud run deploy ${IMAGE_NAME} \
  --image "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}" \
  --region europe-north1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --service-account event-feed-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --update-secrets=/secrets/sa-key.json=EVENT_FEED_SA_KEY:latest \
  --update-secrets=GMAIL_OAUTH_CREDENTIALS=GMAIL_OAUTH_CREDENTIALS:latest
  #--set-env-vars=DVC_REMOTE_MYGDRIVE_GDRIVE_SERVICE_ACCOUNT_JSON_FILE_PATH=/secrets/sa-key.json

echo "🚀 Deployed ${IMAGE_NAME} to Cloud Run"
