#!/bin/bash
set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-event-feed-app-463206}"
REGION="${REGION:-europe-north1}"  # Stockholm region
SERVICE_NAME="consolidate-daily"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "======================================================================"
echo "Deploying Daily Consolidation Job to Cloud Run"
echo "======================================================================"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo "======================================================================"

# Build and push image
echo "Building container image..."
docker build -f deployment/Dockerfile.consolidation -t ${IMAGE_NAME} .

echo "Pushing image to GCR..."
docker push ${IMAGE_NAME}

# Deploy Cloud Run Job
echo "Deploying Cloud Run Job..."
gcloud run jobs deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --region ${REGION} \
  --project ${PROJECT_ID} \
  --set-env-vars "PROJECT_ID=${PROJECT_ID},GCS_BUCKET=event-feed-app-data" \
  --max-retries 0 \
  --task-timeout 30m \
  --memory 2Gi \
  --cpu 1

echo "======================================================================"
echo "Deployment complete!"
echo "======================================================================"
echo ""
echo "To test manually:"
echo "  gcloud run jobs execute ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID}"
echo ""
echo "To schedule (runs daily at 00:05 UTC):"
echo "  gcloud scheduler jobs create http consolidate-daily-trigger \\"
echo "    --location ${REGION} \\"
echo "    --schedule '5 0 * * *' \\"
echo "    --uri 'https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${SERVICE_NAME}:run' \\"
echo "    --http-method POST \\"
echo "    --oauth-service-account-email <SERVICE_ACCOUNT_EMAIL>"
echo ""
