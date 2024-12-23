# 🔄 Self-Healing LLM Workflow System

**Production-Ready Multi-Agent Workflow Orchestration with AI**

Advanced workflow system that uses multiple AI agents to execute complex tasks with self-healing capabilities, adaptive learning, and real-time performance monitoring.

## 🚀 **Quick Start**

### 1. **Configure OpenAI API Key**
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```

### 2. **Deploy Locally**
```bash
python3 deploy.py local
```

### 3. **Access Application**
Open your browser at: **http://localhost:8503**

---

## 📦 **Installation**

### **Prerequisites**
- Python 3.8+
- OpenAI API key

### **Setup**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key (create .streamlit/secrets.toml)
echo 'OPENAI_API_KEY = "your_key_here"' > .streamlit/secrets.toml

# 3. Start application
streamlit run main.py --server.port 8503
```

---

## ☁️ **Cloud Deployment (Google Cloud Run)**

### **Generate Deployment Commands**
```bash
python3 deploy.py cloud
```

### **Manual Deployment**
```bash
# 1. Build image
docker build -t gcr.io/YOUR_PROJECT_ID/self-healing-workflow .

# 2. Push to registry
docker push gcr.io/YOUR_PROJECT_ID/self-healing-workflow

# 3. Deploy to Cloud Run
gcloud run deploy self-healing-workflow \
  --image gcr.io/YOUR_PROJECT_ID/self-healing-workflow \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars OPENAI_API_KEY="your_api_key_here"
```

---

## 🎯 **Features**

### **Multi-Agent System**
- **LLM Reasoning Agent**: Complex problem solving and decision making
- **Code Generator Agent**: Automated code generation and optimization
- **Data Processor Agent**: Data analysis and transformation
- **Quality Checker Agent**: Automated quality assurance and validation

### **Self-Healing Capabilities**
- **Automatic Error Detection**: Real-time monitoring and issue identification
- **Recovery Strategies**: Intelligent retry mechanisms and fallback procedures
- **Learning from Failures**: Adaptive improvement based on past errors

### **Advanced Memory Management**
- **Working Memory**: Short-term task execution context
- **Long-term Memory**: Persistent knowledge storage with vector embeddings
- **Episodic Memory**: Conversation history and interaction patterns

### **Production Features**
- **Real-time Monitoring**: Performance metrics and system health
- **Auto-scaling**: Dynamic resource allocation based on demand
- **Cost Optimization**: Intelligent API usage and resource management
- **Comprehensive Logging**: Full audit trail and debugging capabilities

---

## 🔧 **Usage Examples**

### **1. AI Content Generation**
```python
# Create content generation workflow
workflow = {
    "name": "Blog Post Generation",
    "type": "content_generation",
    "tasks": [
        {"agent": "llm_reasoning", "task": "research_topic"},
        {"agent": "llm_reasoning", "task": "generate_content"},
        {"agent": "quality_checker", "task": "review_quality"}
    ]
}
```

### **2. Data Analysis Pipeline**
```python
# Create data analysis workflow
workflow = {
    "name": "Sales Data Analysis",
    "type": "data_analysis",
    "tasks": [
        {"agent": "data_processor", "task": "clean_data"},
        {"agent": "data_processor", "task": "analyze_trends"},
        {"agent": "llm_reasoning", "task": "generate_insights"}
    ]
}
```

### **3. Code Generation & Review**
```python
# Create code generation workflow
workflow = {
    "name": "API Development",
    "type": "code_generation",
    "tasks": [
        {"agent": "code_generator", "task": "generate_api"},
        {"agent": "quality_checker", "task": "code_review"},
        {"agent": "code_generator", "task": "fix_issues"}
    ]
}
```

---

## 📊 **Performance Metrics**

The system provides comprehensive monitoring:

- **Success Rate**: Workflow completion percentage
- **Healing Rate**: Self-recovery success rate
- **Response Time**: Average task execution time
- **Cost Tracking**: OpenAI API usage and costs
- **Agent Performance**: Individual agent success rates

---

## 🛠️ **Configuration**

### **Environment Variables**
```bash
OPENAI_API_KEY=your_api_key_here
MAX_WORKERS=8
TIMEOUT=60
RETRY_ATTEMPTS=5
ENABLE_ADVANCED_MEMORY=true
ENABLE_SELF_HEALING=true
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_AUTO_SCALING=true
```

### **Production Settings**
- **Memory**: 2Gi minimum
- **CPU**: 2 cores recommended
- **Concurrency**: 80 concurrent requests
- **Timeout**: 300 seconds
- **Auto-scaling**: Up to 10 instances

---

## 📁 **Project Structure**

```
02-self-healing-llm-workflow/
├── main.py                    # Main Streamlit application
├── deploy.py                  # Production deployment script
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── Procfile                   # Deployment configuration
├── .streamlit/
│   └── secrets.toml          # API key configuration
└── core/
    ├── workflow_engine.py     # Workflow orchestration
    ├── agent_manager.py       # Multi-agent system
    ├── memory_system.py       # Advanced memory management
    └── healing_orchestrator.py # Self-healing capabilities
```

---

## 🔍 **Troubleshooting**

### **Common Issues**

**1. "OpenAI API key not found"**
- Verify `.streamlit/secrets.toml` exists and contains your API key
- Check API key format starts with `sk-`

**2. "No module named 'streamlit'"**
- Ensure dependencies are installed: `pip install -r requirements.txt`
- Activate virtual environment if using one

**3. "Port already in use"**
- Use different port: `streamlit run main.py --server.port 8504`
- Kill existing processes: `pkill -f streamlit`

**4. Performance Issues**
- Increase memory allocation in cloud deployment
- Monitor API usage and rate limits
- Check system metrics in dashboard

---

## 🔒 **Security**

- API keys stored securely in secrets.toml
- No sensitive data logged or exposed
- Rate limiting and request validation
- Container security best practices

---

## 📈 **Scaling**

The system is designed for production scale:

- **Horizontal Scaling**: Multiple instances with load balancing
- **Vertical Scaling**: Configurable CPU and memory allocation
- **Auto-scaling**: Dynamic resource allocation based on demand
- **Cost Optimization**: Intelligent API usage patterns

---

## 🤝 **Support**

For issues or questions:
1. Check the troubleshooting section above
2. Review system logs for error details
3. Verify configuration and API key setup
4. Monitor performance metrics in the dashboard

---

## 🎯 **Success!**

Your **Self-Healing LLM Workflow System** is now ready for production use!

**Key Features:**
- ✅ Multi-agent workflow orchestration
- ✅ Self-healing and adaptive learning
- ✅ Real-time performance monitoring
- ✅ Production-ready deployment
- ✅ Comprehensive error handling

**Next Steps:**
1. Create your first workflow
2. Monitor performance metrics
3. Scale based on usage patterns
4. Explore advanced features

---

## 🎨 **UI/UX Features**

### **Professional Color Scheme**
- **High Contrast Design**: Improved readability with dark text on light backgrounds
- **Professional Blue Theme**: Modern gradient headers with proper text shadows
- **Status Indicators**: Color-coded status with high contrast (green/amber/red)
- **Clean White Cards**: Task and metric cards with subtle shadows and borders

### **Enhanced Typography**
- **Larger Font Sizes**: Improved readability across all components
- **Bold Headings**: Clear hierarchy with proper font weights
- **Better Spacing**: Consistent padding and margins throughout
- **Text Shadows**: Enhanced readability on gradient backgrounds

### **Interactive Elements**
- **Focus States**: Clear visual feedback on form inputs
- **Hover Effects**: Subtle animations and state changes
- **Professional Buttons**: Gradient buttons with hover animations
- **Better Form Fields**: High contrast input fields with proper borders

### **Accessibility**
- **WCAG Compliant**: High contrast ratios for better accessibility
- **Clear Labels**: Descriptive labels and status indicators
- **Consistent Layout**: Predictable interface patterns
- **Mobile Friendly**: Responsive design elements

**Happy building! 🚀** 