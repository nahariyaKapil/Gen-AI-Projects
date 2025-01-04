# 💰 Cost Optimization Usage Guide

## Overview

This guide explains how to deploy and use your GenAI Portfolio with aggressive cost optimization settings that can reduce your cloud costs by **70-80%** through serverless architecture and intelligent resource management.

## 🚀 Quick Start

### 1. Deploy Cost-Optimized Infrastructure

```bash
# Clone the repository
git clone <your-repo-url>
cd Projects/production-deployment

# Make the deployment script executable
chmod +x deploy-cost-optimized.sh

# Deploy with your project settings
./deploy-cost-optimized.sh \
    --project-id "your-project-id" \
    --daily-limit 15.0 \
    --monthly-limit 150.0 \
    --region us-central1
```

### 2. Verify Deployment

```bash
# Check service status
gcloud run services describe genai-portfolio --region=us-central1

# Test the service
curl -f https://genai-portfolio-your-project-id.a.run.app/health

# View logs
gcloud run logs tail genai-portfolio --region=us-central1
```

### 3. Access Your Applications

- **Main Service**: `https://genai-portfolio-your-project-id.a.run.app`
- **Cost Dashboard**: `https://genai-portfolio-your-project-id.a.run.app/dashboard`
- **Health Check**: `https://genai-portfolio-your-project-id.a.run.app/health`

## 🏗️ Architecture Overview

```mermaid
graph TB
    A[User Request] --> B[Cloud Run Service]
    B --> C{Models Loaded?}
    C -->|No| D[Lazy Load Models]
    C -->|Yes| E[Process Request]
    D --> E
    E --> F[Auto-cleanup Check]
    F --> G[Return Response]
    G --> H{Idle Timeout?}
    H -->|Yes| I[Scale to Zero]
    H -->|No| J[Keep Running]
    I --> K[💰 $0.00/hour]
    J --> L[📊 Monitor Usage]
```

## 💡 Key Cost Optimization Features

### 1. Scale-to-Zero Architecture
- **Zero cost when idle**: Services automatically scale down to 0 instances
- **Fast cold starts**: Optimized for quick startup times
- **Usage-based billing**: Pay only for actual compute time

### 2. Lazy Model Loading
- **On-demand loading**: Models loaded only when needed
- **Memory optimization**: Automatic cleanup of unused models
- **Caching strategy**: Intelligent model caching for performance

### 3. Resource Monitoring
- **Real-time tracking**: CPU, memory, and GPU usage monitoring
- **Cost projections**: Daily and monthly cost estimates
- **Alert system**: Automated alerts for cost thresholds

### 4. Auto-cleanup
- **Scheduled cleanup**: Automatic resource cleanup every 6 hours
- **Idle detection**: Unload models after 5 minutes of inactivity
- **Memory management**: Automatic GPU cache clearing

## 📊 Cost Comparison

### Traditional Always-On Deployment
```
Monthly Cost Breakdown:
├── Compute (24/7): $73.20
├── Memory (24/7): $51.84
├── Storage: $12.00
├── Network: $8.50
└── Total: $145.54/month
```

### Cost-Optimized Deployment
```
Monthly Cost Breakdown:
├── Compute (usage-based): $15.50
├── Memory (usage-based): $8.20
├── Storage: $3.60
├── Network: $2.80
└── Total: $30.10/month
```

**💰 Savings: $115.44/month (79% reduction)**

## 🛠️ Configuration Options

### Environment Variables

```bash
# Core optimization settings
LAZY_MODEL_LOADING=true
AUTO_CLEANUP_ENABLED=true
COST_OPTIMIZATION_MODE=aggressive
SCALE_TO_ZERO_TIMEOUT=300

# Cost limits
COST_DAILY_LIMIT=15.0
COST_MONTHLY_LIMIT=150.0
IDLE_TIMEOUT_MINUTES=5
CLEANUP_INTERVAL_MINUTES=30

# Performance tuning
MAX_CONCURRENT_REQUESTS=10
MEMORY_LIMIT=2Gi
CPU_LIMIT=1000m
```

### Cloud Run Configuration

```yaml
# cost-optimized-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: genai-portfolio
  annotations:
    autoscaling.knative.dev/minScale: "0"  # Scale to zero
    autoscaling.knative.dev/maxScale: "5"  # Cost control
    run.googleapis.com/cpu-throttling: "true"
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "1"
    spec:
      containerConcurrency: 10
      timeoutSeconds: 300
```

## 📈 Monitoring and Alerting

### Cost Monitoring Dashboard

Access the built-in cost monitoring dashboard:

```bash
# Open dashboard URL
echo "https://genai-portfolio-your-project-id.a.run.app/dashboard"
```

The dashboard provides:
- Real-time cost tracking
- Resource usage metrics
- Model loading status
- Optimization recommendations
- Cost projections

### Setting Up Alerts

```bash
# Create cost budget alert
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="GenAI Portfolio Budget" \
    --budget-amount=150USD \
    --threshold-percent=50,80,100
```

### Custom Monitoring Scripts

```python
# cost_monitor.py
import requests
import time
from datetime import datetime

def monitor_costs():
    """Monitor service costs and send alerts"""
    service_url = "https://genai-portfolio-your-project-id.a.run.app"
    
    while True:
        try:
            # Check health
            health = requests.get(f"{service_url}/health").json()
            
            # Get cost status
            cost_status = requests.get(f"{service_url}/cost-status").json()
            
            # Check thresholds
            if cost_status['daily_cost'] > 15.0:
                send_alert("Daily cost threshold exceeded")
            
            print(f"✅ Health: {health['status']}")
            print(f"💰 Daily Cost: ${cost_status['daily_cost']:.2f}")
            
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        
        time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    monitor_costs()
```

## 🔧 Best Practices

### 1. Optimize Model Loading
```python
# Use lazy loading in your applications
from shared_infrastructure.cost_optimizer import get_cost_optimizer

cost_optimizer = get_cost_optimizer()

# Register models for lazy loading
cost_optimizer.register_model("nlp_model", load_nlp_model)
cost_optimizer.register_model("vision_model", load_vision_model)

# Get models when needed
@track_request(cost_optimizer)
def process_request(data):
    model = cost_optimizer.get_model("nlp_model")
    return model.predict(data)
```

### 2. Implement Health Checks
```python
# Add health check endpoints
@app.route('/health')
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cost_optimization": "active"
    }

@app.route('/ready')
def readiness_check():
    return {
        "status": "ready",
        "models_loaded": len(cost_optimizer.lazy_loader.get_status()["loaded_models"])
    }
```

### 3. Configure Auto-scaling
```python
# Use the cost optimizer decorator
from shared_infrastructure.cost_optimizer import track_request

@track_request(cost_optimizer)
def handle_request(request):
    # Your request handling logic
    return process_request(request)
```

### 4. Monitor Resource Usage
```python
# Regular resource monitoring
def monitor_resources():
    status = cost_optimizer.get_cost_dashboard()
    
    if status["resource_usage"]["memory_mb"] > 1500:
        logger.warning("High memory usage detected")
        cost_optimizer.lazy_loader.unload_idle_models()
    
    return status
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Cold Start Times
```bash
# Check cold start metrics
gcloud run services describe genai-portfolio \
    --region=us-central1 \
    --format="value(status.conditions[0].message)"

# Optimize by reducing image size
docker build --target production -t optimized-image .
```

#### 2. Memory Limit Exceeded
```bash
# Check memory usage
gcloud run logs tail genai-portfolio \
    --region=us-central1 \
    --filter="severity=ERROR"

# Increase memory limit
gcloud run services update genai-portfolio \
    --memory=4Gi \
    --region=us-central1
```

#### 3. Cost Alerts Not Working
```bash
# Check billing account setup
gcloud billing accounts list

# Verify budget configuration
gcloud billing budgets list --billing-account=BILLING_ACCOUNT_ID
```

#### 4. Models Not Loading
```python
# Debug model loading
cost_optimizer = get_cost_optimizer()
status = cost_optimizer.lazy_loader.get_status()
print(f"Registered models: {status['registered_models']}")
print(f"Loaded models: {status['loaded_models']}")

# Force model loading
model = cost_optimizer.get_model("your_model_name")
```

### Performance Optimization

#### 1. Optimize Docker Images
```dockerfile
# Multi-stage build for smaller images
FROM python:3.9-slim as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
```

#### 2. Use Caching
```python
# Implement intelligent caching
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(input_data):
    # Your expensive computation
    return result
```

#### 3. Optimize Model Sizes
```python
# Use quantized models
import torch

def optimize_model(model):
    # Quantize model for faster inference
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model
```

## 📋 Maintenance Tasks

### Daily
- [ ] Check cost dashboard
- [ ] Review any alerts
- [ ] Monitor resource usage

### Weekly
- [ ] Analyze usage patterns
- [ ] Review cost projections
- [ ] Update thresholds if needed

### Monthly
- [ ] Review total costs
- [ ] Optimize based on usage
- [ ] Update cost limits
- [ ] Plan capacity changes

## 🔄 Scaling Strategies

### Horizontal Scaling
```bash
# Adjust max instances based on traffic
gcloud run services update genai-portfolio \
    --max-instances=10 \
    --region=us-central1
```

### Vertical Scaling
```bash
# Adjust resources based on workload
gcloud run services update genai-portfolio \
    --memory=4Gi \
    --cpu=2 \
    --region=us-central1
```

### Auto-scaling Rules
```yaml
# Custom auto-scaling configuration
metadata:
  annotations:
    autoscaling.knative.dev/minScale: "0"
    autoscaling.knative.dev/maxScale: "10"
    autoscaling.knative.dev/targetConcurrencyUtilization: "70"
```

## 📞 Support

### Getting Help
1. Check the troubleshooting section above
2. Review logs: `gcloud run logs tail genai-portfolio --region=us-central1`
3. Check service status: `gcloud run services describe genai-portfolio --region=us-central1`
4. Monitor dashboard: `https://genai-portfolio-your-project-id.a.run.app/dashboard`

### Useful Commands
```bash
# View service details
gcloud run services describe genai-portfolio --region=us-central1

# Update service configuration
gcloud run services update genai-portfolio \
    --set-env-vars="COST_DAILY_LIMIT=20.0" \
    --region=us-central1

# Check billing
gcloud billing accounts list
gcloud billing budgets list --billing-account=BILLING_ACCOUNT_ID

# Monitor costs
gcloud run services list --filter="metadata.name=genai-portfolio"
```

## 🎯 Success Metrics

Track these metrics to measure optimization success:

- **Cost Reduction**: Target 70-80% reduction vs always-on
- **Cold Start Time**: < 10 seconds for model loading
- **Memory Usage**: < 2GB average, < 4GB peak
- **Availability**: > 99.9% uptime
- **Response Time**: < 30 seconds for complex queries

## 📝 Conclusion

By following this guide, you'll have a cost-optimized GenAI portfolio that:

✅ **Scales to zero** when idle (no cost)  
✅ **Loads models on-demand** (faster startup)  
✅ **Monitors costs in real-time** (budget control)  
✅ **Auto-cleans resources** (memory optimization)  
✅ **Provides 70-80% cost savings** (significant reduction)

The cost-optimized architecture ensures you only pay for what you use while maintaining high performance and reliability for your GenAI applications.

---

**💰 Estimated Monthly Savings: $115+ (79% reduction)**  
**⏱️ Setup Time: 15 minutes**  
**🚀 Deployment Method: Fully Automated** 