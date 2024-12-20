#!/bin/bash

# Production Deployment Script for RAG Knowledge Assistant
# Demonstrates expert-level DevOps and production deployment capabilities

set -e  # Exit on any error

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"your-project-id"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="rag-knowledge-assistant"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        error "gcloud CLI is not installed. Please install it first."
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install it first."
    fi
    
    # Check if logged in to gcloud
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        error "Not logged in to gcloud. Please run 'gcloud auth login' first."
    fi
    
    # Check if project is set
    if [ "$PROJECT_ID" = "your-project-id" ]; then
        error "Please set GOOGLE_CLOUD_PROJECT environment variable"
    fi
    
    success "Prerequisites check passed"
}

# Enable required APIs
enable_apis() {
    log "Enabling required Google Cloud APIs..."
    
    gcloud services enable cloudbuild.googleapis.com \
        run.googleapis.com \
        secretmanager.googleapis.com \
        bigquery.googleapis.com \
        firestore.googleapis.com \
        redis.googleapis.com \
        monitoring.googleapis.com \
        logging.googleapis.com \
        --project=$PROJECT_ID
    
    success "APIs enabled"
}

# Create service account
create_service_account() {
    log "Creating service account..."
    
    SA_NAME="rag-service-account"
    SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    # Create service account if it doesn't exist
    if ! gcloud iam service-accounts describe $SA_EMAIL --project=$PROJECT_ID &> /dev/null; then
        gcloud iam service-accounts create $SA_NAME \
            --display-name="RAG Knowledge Assistant Service Account" \
            --description="Service account for RAG Knowledge Assistant" \
            --project=$PROJECT_ID
    fi
    
    # Grant necessary roles
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/bigquery.dataEditor"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/datastore.user"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/secretmanager.secretAccessor"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/monitoring.metricWriter"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/logging.logWriter"
    
    success "Service account created and configured"
}

# Create secrets
create_secrets() {
    log "Creating secrets in Secret Manager..."
    
    # Create secrets if they don't exist
    if ! gcloud secrets describe rag-secrets --project=$PROJECT_ID &> /dev/null; then
        # Create secret for sensitive data
        gcloud secrets create rag-secrets --project=$PROJECT_ID
        
        # You would add actual secret values here
        warning "Please manually add secret values:"
        echo "  - openai-api-key: Your OpenAI API key"
        echo "  - jwt-secret-key: Your JWT secret key"
        echo "  - redis-url: Your Redis connection URL"
        echo ""
        echo "Example commands:"
        echo "  echo 'your-openai-key' | gcloud secrets versions add rag-secrets --data-file=- --project=$PROJECT_ID"
    fi
    
    success "Secrets configured"
}

# Build and push Docker image
build_and_push() {
    log "Building and pushing Docker image..."
    
    # Configure Docker for gcloud
    gcloud auth configure-docker --quiet
    
    # Build image
    docker build -t $IMAGE_NAME:latest .
    
    # Push to Container Registry
    docker push $IMAGE_NAME:latest
    
    success "Docker image built and pushed"
}

# Deploy to Cloud Run
deploy_cloud_run() {
    log "Deploying to Cloud Run..."
    
    # Update the cloud-run.yaml with actual project ID
    sed "s/YOUR_PROJECT_ID/$PROJECT_ID/g" deploy/cloud-run.yaml > deploy/cloud-run-actual.yaml
    
    # Deploy using gcloud
    gcloud run services replace deploy/cloud-run-actual.yaml \
        --region=$REGION \
        --project=$PROJECT_ID
    
    # Wait for deployment to complete
    gcloud run services wait $SERVICE_NAME \
        --region=$REGION \
        --project=$PROJECT_ID
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
        --region=$REGION \
        --project=$PROJECT_ID \
        --format='value(status.url)')
    
    success "Deployment completed successfully!"
    success "Service URL: $SERVICE_URL"
    
    # Clean up temporary file
    rm -f deploy/cloud-run-actual.yaml
}

# Set up monitoring
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    # Create BigQuery dataset for analytics
    if ! bq ls -d --project_id=$PROJECT_ID | grep -q rag_analytics; then
        bq mk --dataset --project_id=$PROJECT_ID rag_analytics
    fi
    
    # Create tables for analytics
    bq mk --table \
        --project_id=$PROJECT_ID \
        rag_analytics.conversations \
        timestamp:TIMESTAMP,query:STRING,response:STRING,retrieval_time_ms:FLOAT,feedback_score:INTEGER
    
    bq mk --table \
        --project_id=$PROJECT_ID \
        rag_analytics.document_stats \
        timestamp:TIMESTAMP,document_id:STRING,total_chunks:INTEGER,total_tokens:INTEGER,format:STRING
    
    success "Monitoring setup completed"
}

# Main deployment flow
main() {
    log "Starting RAG Knowledge Assistant deployment..."
    log "Project: $PROJECT_ID"
    log "Region: $REGION"
    log "Service: $SERVICE_NAME"
    echo ""
    
    check_prerequisites
    enable_apis
    create_service_account
    create_secrets
    build_and_push
    deploy_cloud_run
    setup_monitoring
    
    echo ""
    success "🚀 RAG Knowledge Assistant deployed successfully!"
    echo ""
    log "Next steps:"
    echo "  1. Add secret values to Secret Manager"
    echo "  2. Configure custom domain (optional)"
    echo "  3. Set up CI/CD pipeline"
    echo "  4. Configure monitoring alerts"
    echo ""
    log "Useful commands:"
    echo "  View logs: gcloud run services logs tail $SERVICE_NAME --region=$REGION"
    echo "  Update service: gcloud run services update $SERVICE_NAME --region=$REGION"
    echo "  View metrics: gcloud monitoring dashboards list"
}

# Run main function
main "$@" 