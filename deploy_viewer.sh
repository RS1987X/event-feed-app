#!/bin/bash
# Deploy Press Release Viewer to Google Cloud Run

set -e

# Configuration
PROJECT_ID="event-feed-app-463206"
SERVICE_NAME="pr-viewer"
REGION="europe-north1"  # Finland - closest to Sweden
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying Press Release Viewer to Cloud Run..."

# Step 1: Build container image
echo "📦 Building Docker image..."
docker build -t ${IMAGE_NAME} -f Dockerfile.viewer .
docker push ${IMAGE_NAME}

# Step 2: Deploy to Cloud Run
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars "GCS_SILVER_ROOT=event-feed-app-data/silver_normalized/table=press_releases"

# Step 3: Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo "🔗 Service URL: ${SERVICE_URL}"
echo ""
echo "Next steps:"
echo "1. Add this to your .env file:"
echo "   VIEWER_BASE_URL=${SERVICE_URL}"
echo ""
echo "2. Test the viewer:"
echo "   ${SERVICE_URL}/?id=1980c93f9f819a6d"
echo ""
echo "3. Cost monitoring:"
echo "   gcloud run services describe ${SERVICE_NAME} --region ${REGION}"
