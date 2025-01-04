# 🚀 PRODUCTION DEPLOYMENT GUIDE
**Expert-Level GenAI Portfolio Deployment**

## 📋 Overview

This guide will deploy your complete GenAI portfolio to Google Cloud Platform with enterprise-grade infrastructure, monitoring, and security.

## 🎯 What You'll Get

- ✅ **5 Production Applications** deployed to Google Cloud Run
- ✅ **Enterprise Infrastructure** (VPC, PostgreSQL, Redis, BigQuery)
- ✅ **Monitoring & Alerting** (Prometheus, Grafana, Cloud Monitoring)
- ✅ **Security** (IAM, Secret Manager, VPC networking)
- ✅ **Scalability** (Auto-scaling, Load balancing)
- ✅ **CI/CD Pipeline** (Automated deployments)

## 🛠️ Prerequisites

### 1. Install Required Tools

```bash
# Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Terraform (Optional, for infrastructure as code)
wget https://releases.hashicorp.com/terraform/1.5.0/terraform_1.5.0_darwin_amd64.zip
unzip terraform_1.5.0_darwin_amd64.zip
sudo mv terraform /usr/local/bin/

# Kubectl (Optional, for Kubernetes)
gcloud components install kubectl
```

### 2. Set Up Google Cloud Project

```bash
# Create new project (or use existing)
gcloud projects create your-genai-portfolio-2024 --name="GenAI Portfolio"

# Set as default project
gcloud config set project your-genai-portfolio-2024

# Enable billing (required for Cloud Run, etc.)
# Go to: https://console.cloud.google.com/billing
```

## 🔧 Step-by-Step Deployment

### Step 1: Clone and Prepare

```bash
# Navigate to your portfolio directory
cd /Users/kapilnahariya/Projects/Projects

# Make deployment script executable
chmod +x production-deployment/master-deploy.sh
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp production-deployment/.env.template .env

# Edit with your actual values
nano .env
```

**Required Environment Variables:**
```bash
GOOGLE_CLOUD_PROJECT=your-genai-portfolio-2024
GCP_REGION=us-central1
OPENAI_API_KEY=sk-your-actual-openai-key
DOMAIN=your-domain.com  # Optional
```

### Step 3: Set Up API Keys

You need to obtain the following API keys:

#### 🔑 OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create new API key
3. Add to `.env` file

#### 🔑 Google Cloud APIs
```bash
# Enable required APIs
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  bigquery.googleapis.com \
  firestore.googleapis.com
```

#### 🔑 Other API Keys (Optional)
- **Anthropic**: For Claude models
- **Google Translate**: For multilingual features
- **Sentry**: For error monitoring

### Step 4: Run Master Deployment

```bash
# Set environment variables
export GOOGLE_CLOUD_PROJECT=your-genai-portfolio-2024
export GCP_REGION=us-central1

# Run master deployment script
./production-deployment/master-deploy.sh
```

This will automatically:
- ✅ Check prerequisites
- ✅ Enable Google Cloud APIs
- ✅ Create IAM service accounts
- ✅ Set up infrastructure (VPC, databases, etc.)
- ✅ Build and push Docker images
- ✅ Deploy all 5 applications
- ✅ Configure monitoring
- ✅ Run health checks

### Step 5: Add Secrets

After deployment, add your API keys to Google Secret Manager:

```bash
# Add OpenAI API Key
echo "sk-your-actual-openai-key" | gcloud secrets versions add rag-assistant-secrets --data-file=- --project=your-genai-portfolio-2024

# Add other secrets for each service
echo "your-jwt-secret" | gcloud secrets versions add workflow-system-secrets --data-file=- --project=your-genai-portfolio-2024
```

### Step 6: Custom Domain (Optional)

If you have a domain:

```bash
# Set up domain
export DOMAIN=your-domain.com
./production-deployment/master-deploy.sh
```

## 🌐 Access Your Applications

After successful deployment, you'll get URLs like:

- **RAG Assistant**: https://rag-assistant-xxx-uc.a.run.app
- **Workflow System**: https://workflow-system-xxx-uc.a.run.app
- **Code Assistant**: https://code-assistant-xxx-uc.a.run.app
- **Benchmark Dashboard**: https://benchmark-dashboard-xxx-uc.a.run.app
- **Multilingual AI**: https://multilingual-ai-xxx-uc.a.run.app

## 📊 Monitoring & Analytics

### Access Monitoring Dashboards

1. **Google Cloud Console**: 
   - Monitoring: https://console.cloud.google.com/monitoring
   - Logs: https://console.cloud.google.com/logs

2. **Custom Dashboards**:
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

### Key Metrics to Monitor

- **Response Times**: < 2 seconds
- **Error Rates**: < 1%
- **CPU Usage**: < 80%
- **Memory Usage**: < 85%
- **Cost**: Monitor BigQuery for daily spend

## 🔐 Security Best Practices

### Implemented Security Features

- ✅ **VPC Networking**: Private communication between services
- ✅ **IAM Roles**: Least privilege access
- ✅ **Secret Manager**: Secure API key storage
- ✅ **HTTPS Only**: SSL/TLS encryption
- ✅ **Health Checks**: Automated monitoring
- ✅ **Rate Limiting**: API protection

### Additional Security Steps

```bash
# Enable audit logging
gcloud logging sinks create audit-sink \
  bigquery.googleapis.com/projects/your-genai-portfolio-2024/datasets/audit_logs \
  --log-filter='protoPayload.serviceName="cloudresourcemanager.googleapis.com"'

# Set up firewall rules
gcloud compute firewall-rules create allow-genai-apps \
  --allow tcp:8080 \
  --source-ranges 0.0.0.0/0 \
  --description "Allow GenAI applications"
```

## 💰 Cost Management

### Estimated Monthly Costs

- **Cloud Run**: $20-50/month (depends on usage)
- **Cloud SQL**: $15-30/month
- **Redis**: $25-50/month
- **BigQuery**: $10-25/month
- **Storage**: $5-15/month
- **Total**: ~$75-170/month

### Cost Optimization

```bash
# Set up budget alerts
gcloud billing budgets create \
  --billing-account=YOUR-BILLING-ACCOUNT-ID \
  --display-name="GenAI Portfolio Budget" \
  --budget-amount=100USD \
  --threshold-rule=percent:50 \
  --threshold-rule=percent:90
```

## 🔄 CI/CD Pipeline

### Automated Deployments

```bash
# Set up Cloud Build trigger
gcloud builds triggers create github \
  --repo-name=genai-portfolio \
  --repo-owner=your-username \
  --branch-pattern=main \
  --build-config=cloudbuild.yaml
```

### Rolling Updates

```bash
# Update individual service
gcloud run deploy rag-assistant \
  --image=gcr.io/your-genai-portfolio-2024/rag-assistant:v2.0 \
  --region=us-central1
```

## 🆘 Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check build logs
   gcloud builds log $(gcloud builds list --limit=1 --format="value(id)")
   ```

2. **Deployment Errors**
   ```bash
   # Check service logs
   gcloud run services logs read rag-assistant --region=us-central1
   ```

3. **Memory Issues**
   ```bash
   # Increase memory allocation
   gcloud run services update rag-assistant \
     --memory=8Gi \
     --region=us-central1
   ```

### Health Check Commands

```bash
# Test all services
curl https://rag-assistant-xxx-uc.a.run.app/_stcore/health
curl https://workflow-system-xxx-uc.a.run.app/_stcore/health
curl https://code-assistant-xxx-uc.a.run.app/_stcore/health
curl https://benchmark-dashboard-xxx-uc.a.run.app/_stcore/health
curl https://multilingual-ai-xxx-uc.a.run.app/_stcore/health
```

## 🎓 Expert-Level Features

### Advanced Monitoring

```bash
# Custom metrics
gcloud logging metrics create response_time_metric \
  --description="Application response time" \
  --log-filter='resource.type="cloud_run_revision"'
```

### Performance Optimization

```bash
# Enable CPU boost
gcloud run services update rag-assistant \
  --cpu-boost \
  --region=us-central1
```

### Multi-Region Deployment

```bash
# Deploy to multiple regions
export GCP_REGION=europe-west1
./production-deployment/master-deploy.sh
```

## 🏆 Success Metrics

Your deployment is successful when:

- ✅ All 5 applications are running and accessible
- ✅ Health checks pass for all services
- ✅ Response times < 2 seconds
- ✅ Error rate < 1%
- ✅ Monitoring dashboards show green status
- ✅ Cost within budget
- ✅ Security scans pass

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review Google Cloud Console logs
3. Verify environment variables are set correctly
4. Ensure API quotas are not exceeded
5. Check billing account is active

---

**🎉 Congratulations! You now have a production-ready, expert-level GenAI portfolio deployed to Google Cloud!**

This deployment demonstrates:
- Advanced cloud architecture
- Enterprise security practices
- Production monitoring
- Scalable infrastructure
- Cost management
- DevOps best practices

Perfect for showcasing in technical interviews for senior GenAI roles! 🚀 