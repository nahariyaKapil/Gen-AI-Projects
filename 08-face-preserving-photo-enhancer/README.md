# Face-preserving AI Dating Photo Enhancer

A sophisticated AI-powered photo enhancement pipeline that improves dating photos while preserving facial identity using advanced face detection, identity preservation, and style transfer techniques.

## 🎯 Overview

This project implements a multi-step pipeline that enhances photos specifically for dating applications:

1. **Face Detection & Analysis** - MediaPipe-based face detection and quality assessment
2. **Identity Preservation** - Face encoding extraction and similarity validation
3. **Style Enhancement** - Stable Diffusion-based style transfer with multiple preset styles
4. **Validation & Correction** - Identity preservation validation with adaptive correction
5. **Quality Control** - Comprehensive feedback system for continuous improvement

## ✨ Key Features

- **Advanced Face Detection**: MediaPipe integration for robust face detection and landmark extraction
- **Identity Preservation**: Face encoding-based validation ensures the person's identity remains unchanged
- **Multiple Enhancement Styles**: 6 preset styles (professional, glamorous, casual, artistic, vintage, modern)
- **Adaptive Processing**: Automatically adjusts enhancement strength based on image quality
- **Real-time Progress Tracking**: WebSocket-based progress updates for long-running operations
- **Comprehensive Validation**: Identity similarity scoring with automatic correction
- **Quality Feedback System**: User feedback collection for continuous improvement
- **Batch Processing**: Support for processing multiple images simultaneously
- **GPU Acceleration**: CUDA support for faster processing

## 🏗️ Architecture

```
├── app.py                 # FastAPI application
├── core/
│   ├── face_detector.py      # MediaPipe face detection
│   ├── identity_preservor.py # Face encoding & validation
│   ├── style_enhancer.py     # Stable Diffusion enhancement
│   ├── pipeline_manager.py   # Pipeline orchestration
│   └── utils.py              # Utilities and config
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
└── README.md            # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 08-face-preserving-photo-enhancer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

### Docker Setup

1. **Build the image**
   ```bash
   docker build -t face-enhancer .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 -v $(pwd)/uploads:/app/uploads face-enhancer
   ```

### GPU Support (Docker)

For GPU acceleration, use the CUDA-enabled Dockerfile:

```bash
# Uncomment CUDA section in Dockerfile
docker build -t face-enhancer-gpu .
docker run --gpus all -p 8000:8000 -v $(pwd)/uploads:/app/uploads face-enhancer-gpu
```

## 📚 API Documentation

### Core Endpoints

#### 1. Enhance Photo
```http
POST /enhance
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- style: Enhancement style (default: "professional")
- enhancement_level: Float 0-1 (default: 0.8)
- preserve_identity: Boolean (default: true)
- output_format: "jpeg" or "png" (default: "jpeg")
- resolution: "1024x1024" (default)
- additional_prompts: Comma-separated string
```

#### 2. Get Enhancement Status
```http
GET /status/{task_id}
```

#### 3. Get Enhancement Result
```http
GET /result/{task_id}
```

#### 4. Download Enhanced Image
```http
GET /download/{task_id}/{image_type}
```

#### 5. Submit Feedback
```http
POST /feedback
Content-Type: application/json

{
  "task_id": "string",
  "rating": 1-5,
  "feedback": "string",
  "issues": ["string"]
}
```

### Utility Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /styles` - Available enhancement styles
- `GET /tasks` - List all tasks
- `DELETE /cleanup/{task_id}` - Clean up task files

## 🎨 Enhancement Styles

### 1. Professional
- **Use Case**: Business profiles, LinkedIn, professional dating apps
- **Features**: Clean, polished look with enhanced lighting
- **Strength**: 0.3 (subtle enhancement)

### 2. Glamorous
- **Use Case**: Premium dating apps, special occasions
- **Features**: Enhanced features with dramatic lighting
- **Strength**: 0.4 (moderate enhancement)

### 3. Casual
- **Use Case**: Everyday dating apps, social media
- **Features**: Natural enhancement with warm tones
- **Strength**: 0.25 (minimal enhancement)

### 4. Artistic
- **Use Case**: Creative platforms, artistic dating profiles
- **Features**: Stylized aesthetic with unique flair
- **Strength**: 0.5 (strong enhancement)

### 5. Vintage
- **Use Case**: Themed profiles, retro aesthetic
- **Features**: Film-inspired tones and textures
- **Strength**: 0.4 (moderate enhancement)

### 6. Modern
- **Use Case**: Contemporary dating apps, social media
- **Features**: Sharp details and vibrant colors
- **Strength**: 0.35 (balanced enhancement)

## 🔍 Identity Preservation

### How It Works

1. **Face Encoding Extraction**: Uses face_recognition library to generate 128-dimensional face encodings
2. **Similarity Calculation**: Computes cosine similarity between original and enhanced images
3. **Threshold Validation**: Configurable similarity threshold (default: 0.6)
4. **Adaptive Correction**: Automatically reduces enhancement strength if identity is compromised

### Validation Metrics

- **Similarity Score**: 0-1 scale (higher = more similar)
- **Quality Rating**: Excellent (0.9+), Very Good (0.8+), Good (0.6+), Fair (0.4+), Poor (<0.4)
- **Preservation Status**: Pass/fail based on threshold

## 📊 Performance Optimization

### Memory Management
- **Attention Slicing**: Reduces GPU memory usage for Stable Diffusion
- **VAE Slicing**: Further memory optimization for large images
- **Model Caching**: Keeps models in memory for faster subsequent requests

### Processing Speed
- **GPU Acceleration**: CUDA support for PyTorch operations
- **Batch Processing**: Efficient handling of multiple images
- **Async Processing**: Non-blocking API responses with background tasks

### Quality Adaptation
- **Face Quality Assessment**: Automatic quality scoring and adjustment
- **Enhancement Level Scaling**: Reduces enhancement for low-quality images
- **Error Recovery**: Graceful handling of processing failures

## 🔧 Configuration

### Environment Variables

```bash
# Model Configuration
STABLE_DIFFUSION_MODEL=runwayml/stable-diffusion-v1-5
FACE_DETECTION_MODEL=hog
IDENTITY_THRESHOLD=0.6

# Processing Settings
MAX_IMAGE_SIZE=2048
DEFAULT_ENHANCEMENT_LEVEL=0.8
DEFAULT_STYLE=professional
GPU_MEMORY_FRACTION=0.8

# Storage Settings
UPLOAD_DIR=uploads
MAX_FILE_SIZE=10485760  # 10MB
```

### Configuration File (config.yaml)

```yaml
models:
  stable_diffusion: "runwayml/stable-diffusion-v1-5"
  face_detection: "hog"
  identity_threshold: 0.6

processing:
  max_image_size: 2048
  default_enhancement_level: 0.8
  default_style: "professional"
  gpu_memory_fraction: 0.8

storage:
  upload_dir: "uploads"
  max_file_size: 10485760
  allowed_extensions: [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

api:
  max_concurrent_requests: 5
  request_timeout: 300
  cleanup_interval: 3600
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/test_face_detector.py
pytest tests/test_identity_preservor.py
pytest tests/test_style_enhancer.py
pytest tests/test_pipeline_manager.py
```

### Integration Tests
```bash
pytest tests/test_api.py
pytest tests/test_pipeline_integration.py
```

### Performance Tests
```bash
pytest tests/test_performance.py
```

## 📈 Monitoring

### Health Checks
- **Component Status**: Face detector, identity preservor, style enhancer
- **System Resources**: GPU memory, disk space, CPU usage
- **Model Loading**: Validation of all AI models

### Logging
- **Structured Logging**: JSON format for easy parsing
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Rotation**: Automatic log file rotation

### Metrics
- **Processing Time**: Track enhancement duration
- **Success Rate**: Monitor successful vs. failed enhancements
- **Identity Preservation**: Track similarity scores
- **User Feedback**: Collect and analyze user ratings

## 🚨 Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   - Reduce batch size or image resolution
   - Enable memory optimization features
   - Check GPU memory usage

2. **Face Detection Failures**
   - Ensure face is clearly visible
   - Check image quality and lighting
   - Verify face size in image

3. **Identity Preservation Issues**
   - Lower enhancement strength
   - Use higher quality input images
   - Adjust similarity threshold

4. **Slow Processing**
   - Enable GPU acceleration
   - Optimize model loading
   - Use appropriate image sizes

### Performance Optimization

1. **Memory Usage**
   ```python
   # Enable memory optimizations
   pipe.enable_attention_slicing()
   pipe.enable_vae_slicing()
   ```

2. **Processing Speed**
   ```python
   # Use optimized schedulers
   pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
   ```

3. **Quality vs Speed**
   - Reduce inference steps for faster processing
   - Use smaller models for real-time applications
   - Implement caching for repeated requests

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Face detection and landmark extraction
- **Stable Diffusion**: AI-powered image enhancement
- **face_recognition**: Face encoding and similarity calculation
- **FastAPI**: Modern web framework for APIs
- **OpenCV**: Computer vision operations
- **Hugging Face**: Model hosting and diffusion pipelines

## 📞 Support

For support, please:
1. Check the troubleshooting section
2. Review the API documentation
3. Create an issue on GitHub
4. Contact the development team

---

**Note**: This project is designed for educational and research purposes. Please ensure compliance with applicable laws and platform terms of service when using for commercial applications. 