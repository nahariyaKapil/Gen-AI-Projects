# Personalized Avatar Generator

A cutting-edge AI-powered avatar generation system that uses **Stable Diffusion + LoRA fine-tuning** to create photorealistic, personalized avatars while preserving user identity.

## 🚀 Features

- **LoRA Fine-tuning**: Custom fine-tuning of Stable Diffusion models on user images
- **Face Detection & Processing**: Automatic face detection and image preprocessing
- **Multiple Styles**: Professional, artistic, casual, fantasy, vintage, and modern styles
- **Identity Preservation**: Advanced face consistency validation
- **Real-time Training**: Asynchronous training with progress tracking
- **GPU Acceleration**: CUDA support for faster training and inference
- **RESTful API**: Complete FastAPI backend with comprehensive endpoints
- **Cloud Ready**: Docker containerization for easy deployment

## 🛠️ Tech Stack

- **PyTorch**: Deep learning framework
- **Diffusers**: Stable Diffusion model implementation
- **LoRA**: Low-Rank Adaptation for efficient fine-tuning
- **CLIP**: Image-text embedding model
- **FastAPI**: Modern web framework
- **OpenCV**: Computer vision operations
- **Face Recognition**: Face detection and encoding
- **Pillow**: Image processing

## 📋 Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ storage space

## 🔧 Installation

### Option 1: Docker (Recommended)

```bash
# Build the Docker image
docker build -t avatar-generator .

# Run with GPU support
docker run --gpus all -p 8000:8000 -v $(pwd)/data:/app/data avatar-generator
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd 06-personalized-avatar-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 🚀 Quick Start

1. **Start the server**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. **Upload training images**:
   ```bash
   curl -X POST "http://localhost:8000/upload-training-images" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg" \
     -F "files=@image3.jpg" \
     -F "files=@image4.jpg" \
     -F "files=@image5.jpg"
   ```

3. **Start LoRA training**:
   ```bash
   curl -X POST "http://localhost:8000/train-lora" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user123", "training_steps": 500}'
   ```

4. **Generate avatars**:
   ```bash
   curl -X POST "http://localhost:8000/generate-avatar" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "professional headshot", "user_id": "user123", "style": "professional"}'
   ```

## 📚 API Documentation

### Upload Training Images
```http
POST /upload-training-images
Content-Type: multipart/form-data

files: List[UploadFile] (5-20 images)
user_id: str (optional, default: "default")
```

### Train LoRA Model
```http
POST /train-lora
Content-Type: application/json

{
  "user_id": "string",
  "training_steps": 500,
  "learning_rate": 1e-4,
  "rank": 32
}
```

### Generate Avatar
```http
POST /generate-avatar
Content-Type: application/json

{
  "prompt": "string",
  "user_id": "string",
  "style": "professional|artistic|casual|fantasy|vintage|modern",
  "num_images": 1,
  "guidance_scale": 7.5,
  "num_inference_steps": 20
}
```

### Check Training Status
```http
GET /training-status/{user_id}
```

### Get Available Styles
```http
GET /styles
```

### Download Generated Avatar
```http
GET /download-avatar/{user_id}/{filename}
```

## 🎨 Available Styles

- **Professional**: Business headshots with clean backgrounds
- **Artistic**: Creative portraits with artistic lighting
- **Casual**: Natural, relaxed portraits
- **Fantasy**: Creative fantasy-style artwork
- **Vintage**: Classic, retro-styled portraits
- **Modern**: Contemporary, minimalist portraits

## ⚙️ Configuration

Environment variables:

```bash
# Model Configuration
MODEL_ID=runwayml/stable-diffusion-v1-5
USE_CUDA=true

# Training Configuration
DEFAULT_TRAINING_STEPS=500
DEFAULT_LEARNING_RATE=1e-4
DEFAULT_RANK=32
BATCH_SIZE=1

# Image Processing
MAX_IMAGE_SIZE=1024
MIN_IMAGE_SIZE=256
TARGET_SIZE=512

# API Configuration
MAX_UPLOAD_SIZE=10
MAX_CONCURRENT_TRAININGS=2
RATE_LIMIT_PER_MINUTE=10

# Logging
LOG_LEVEL=INFO
```

## 🔒 Security Features

- File type validation
- Image size limits
- Face consistency validation
- Rate limiting
- CORS configuration
- Input sanitization

## 📊 Monitoring

The system includes comprehensive logging and monitoring:

- Training progress tracking
- System resource monitoring
- Error logging and handling
- Performance metrics
- User activity tracking

## 🚀 Deployment

### Cloud Deployment Options

1. **Google Cloud Run**:
   ```bash
   gcloud run deploy avatar-generator \
     --image gcr.io/PROJECT_ID/avatar-generator \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 8Gi \
     --cpu 4 \
     --gpu 1 \
     --gpu-type nvidia-tesla-t4
   ```

2. **AWS ECS with GPU**:
   ```bash
   aws ecs create-service \
     --cluster avatar-cluster \
     --service-name avatar-generator \
     --task-definition avatar-generator:1 \
     --desired-count 1
   ```

3. **Replicate** (recommended for easy GPU access):
   ```bash
   cog push r8.im/username/avatar-generator
   ```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: avatar-generator
spec:
  replicas: 2
  selector:
    matchLabels:
      app: avatar-generator
  template:
    metadata:
      labels:
        app: avatar-generator
    spec:
      containers:
      - name: avatar-generator
        image: avatar-generator:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
```

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
pytest tests/

# Load testing
pytest tests/test_load.py

# Integration tests
pytest tests/test_integration.py
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use gradient checkpointing
   - Enable VAE slicing

2. **Training Fails**:
   - Check face consistency scores
   - Verify image quality
   - Ensure minimum 5 images

3. **Poor Avatar Quality**:
   - Increase training steps
   - Adjust learning rate
   - Use higher quality training images

### Performance Optimization

- Use mixed precision training
- Enable attention slicing
- Optimize batch sizes
- Use VAE tiling for large images

## 📈 Roadmap

- [ ] Web interface (Streamlit/Next.js)
- [ ] Multi-user support with database
- [ ] Advanced style customization
- [ ] Video avatar generation
- [ ] API key authentication
- [ ] Batch processing
- [ ] Advanced face editing
- [ ] Style transfer capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

---

**Note**: This system requires significant computational resources. For production use, deploy on GPU-enabled cloud instances or use services like Replicate for serverless GPU access. 