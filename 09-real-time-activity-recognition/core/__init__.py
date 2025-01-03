"""
Real-Time Human Activity Recognition Core Module

This module provides the core functionality for real-time human activity recognition
using state-of-the-art deep learning models with ONNX optimization.

Components:
- VideoProcessor: Video stream processing and frame management
- ActivityRecognizer: Activity recognition using I3D, SlowFast, and ViViT models
- ONNXOptimizer: PyTorch to ONNX conversion and optimization
- ObjectDetector: YOLOv8-based object detection for context
- Utils: Utility functions and configurations
"""

from .video_processor import VideoProcessor
from .activity_recognizer import ActivityRecognizer
from .onnx_optimizer import ONNXOptimizer
from .object_detector import ObjectDetector
from .utils import (
    setup_logging,
    load_config,
    format_confidence,
    format_time,
    calculate_fps,
    get_device_info,
    get_memory_usage,
    ACTIVITY_LABELS,
    get_activity_label
)

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"
__description__ = "Real-Time Human Activity Recognition with ONNX Optimization"

# Module-level constants
SUPPORTED_MODELS = ["I3D", "SlowFast", "ViViT"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

# Default configurations
DEFAULT_CONFIG = {
    "video_processor": {
        "frame_size": (224, 224),
        "buffer_size": 16,
        "fps_limit": 30
    },
    "activity_recognizer": {
        "default_model": "I3D",
        "confidence_threshold": 0.5,
        "batch_size": 1
    },
    "onnx_optimizer": {
        "models_dir": "models/onnx",
        "optimization_level": "all",
        "quantization": False
    },
    "object_detector": {
        "model_size": "yolov8n",
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45,
        "max_detections": 100
    }
}

def get_version():
    """Get the current version of the module"""
    return __version__

def get_supported_models():
    """Get list of supported activity recognition models"""
    return SUPPORTED_MODELS.copy()

def get_supported_formats():
    """Get supported video and image formats"""
    return {
        "video": SUPPORTED_VIDEO_FORMATS.copy(),
        "image": SUPPORTED_IMAGE_FORMATS.copy()
    }

def get_default_config():
    """Get default configuration"""
    return DEFAULT_CONFIG.copy()

# Initialize module logger
import logging
logger = logging.getLogger(__name__)
logger.info(f"Real-Time Activity Recognition Core Module v{__version__} loaded")

__all__ = [
    "VideoProcessor",
    "ActivityRecognizer", 
    "ONNXOptimizer",
    "ObjectDetector",
    "setup_logging",
    "load_config",
    "format_confidence",
    "format_time",
    "calculate_fps",
    "get_device_info",
    "get_memory_usage",
    "ACTIVITY_LABELS",
    "get_activity_label",
    "get_version",
    "get_supported_models",
    "get_supported_formats",
    "get_default_config",
    "SUPPORTED_MODELS",
    "SUPPORTED_VIDEO_FORMATS", 
    "SUPPORTED_IMAGE_FORMATS",
    "DEFAULT_CONFIG"
] 