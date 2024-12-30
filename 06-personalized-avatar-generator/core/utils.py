import os
import logging
import json
from typing import Dict, Any
from pathlib import Path
import sys

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "avatar_generator.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")

def get_config() -> Dict[str, Any]:
    """Load configuration from environment variables and config file"""
    
    config = {
        # Model configuration
        "model_id": os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5"),
        "torch_dtype": "float16" if os.getenv("USE_CUDA", "true").lower() == "true" else "float32",
        
        # Training configuration
        "default_training_steps": int(os.getenv("DEFAULT_TRAINING_STEPS", "500")),
        "default_learning_rate": float(os.getenv("DEFAULT_LEARNING_RATE", "1e-4")),
        "default_rank": int(os.getenv("DEFAULT_RANK", "32")),
        "batch_size": int(os.getenv("BATCH_SIZE", "1")),
        
        # Image processing configuration
        "max_image_size": int(os.getenv("MAX_IMAGE_SIZE", "1024")),
        "min_image_size": int(os.getenv("MIN_IMAGE_SIZE", "256")),
        "target_size": int(os.getenv("TARGET_SIZE", "512")),
        
        # Generation configuration
        "default_num_inference_steps": int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", "20")),
        "default_guidance_scale": float(os.getenv("DEFAULT_GUIDANCE_SCALE", "7.5")),
        "max_num_images": int(os.getenv("MAX_NUM_IMAGES", "4")),
        
        # Storage configuration
        "max_storage_per_user": int(os.getenv("MAX_STORAGE_PER_USER", "1000")),  # MB
        "cleanup_after_days": int(os.getenv("CLEANUP_AFTER_DAYS", "30")),
        
        # API configuration
        "max_upload_size": int(os.getenv("MAX_UPLOAD_SIZE", "10")),  # MB
        "max_concurrent_trainings": int(os.getenv("MAX_CONCURRENT_TRAININGS", "2")),
        
        # Security configuration
        "allowed_origins": os.getenv("ALLOWED_ORIGINS", "*").split(","),
        "rate_limit_per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", "10")),
        
        # Logging configuration
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        
        # GPU configuration
        "use_cuda": os.getenv("USE_CUDA", "true").lower() == "true",
        "gpu_memory_fraction": float(os.getenv("GPU_MEMORY_FRACTION", "0.8")),
        
        # Hugging Face configuration
        "hf_token": os.getenv("HF_TOKEN", ""),
        "hf_cache_dir": os.getenv("HF_CACHE_DIR", ""),
        
        # Cloud storage configuration (optional)
        "cloud_storage_enabled": os.getenv("CLOUD_STORAGE_ENABLED", "false").lower() == "true",
        "cloud_storage_bucket": os.getenv("CLOUD_STORAGE_BUCKET", ""),
        "cloud_storage_region": os.getenv("CLOUD_STORAGE_REGION", ""),
        
        # Database configuration (optional)
        "database_url": os.getenv("DATABASE_URL", ""),
        "redis_url": os.getenv("REDIS_URL", ""),
        
        # Monitoring configuration
        "enable_metrics": os.getenv("ENABLE_METRICS", "false").lower() == "true",
        "metrics_port": int(os.getenv("METRICS_PORT", "8001")),
        
        # Development configuration
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "reload": os.getenv("RELOAD", "false").lower() == "true",
    }
    
    # Load config file if exists
    config_file = Path("config.json")
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load config file: {str(e)}")
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters"""
    
    logger = logging.getLogger(__name__)
    
    # Validate required parameters
    required_params = [
        "model_id",
        "default_training_steps",
        "default_learning_rate",
        "target_size"
    ]
    
    for param in required_params:
        if param not in config:
            logger.error(f"Missing required configuration parameter: {param}")
            return False
    
    # Validate numeric ranges
    numeric_validations = {
        "default_training_steps": (100, 5000),
        "default_learning_rate": (1e-6, 1e-2),
        "default_rank": (1, 256),
        "target_size": (256, 1024),
        "max_image_size": (512, 2048),
        "min_image_size": (128, 512),
        "default_num_inference_steps": (5, 100),
        "default_guidance_scale": (1.0, 20.0),
        "max_num_images": (1, 10),
        "gpu_memory_fraction": (0.1, 1.0),
    }
    
    for param, (min_val, max_val) in numeric_validations.items():
        if param in config:
            value = config[param]
            if not (min_val <= value <= max_val):
                logger.error(f"Configuration parameter {param} out of range: {value} (valid: {min_val}-{max_val})")
                return False
    
    # Validate model ID format
    if not config["model_id"] or "/" not in config["model_id"]:
        logger.error("Invalid model_id format. Expected format: 'organization/model-name'")
        return False
    
    logger.info("Configuration validation passed")
    return True

def create_directories():
    """Create necessary directories"""
    
    directories = [
        "uploads",
        "outputs", 
        "models",
        "logs",
        "static",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info("Created necessary directories")

def get_device_info():
    """Get device information for logging"""
    
    import torch
    
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
    }
    
    if torch.cuda.is_available():
        device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        device_info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    
    return device_info

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string"""
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"

def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get model information"""
    
    try:
        # This would typically query the Hugging Face API
        # For now, return basic info
        return {
            "model_id": model_id,
            "type": "stable-diffusion",
            "version": "1.5" if "v1-5" in model_id else "unknown",
            "size": "approximately 4GB",
            "supported": True
        }
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting model info: {str(e)}")
        return {
            "model_id": model_id,
            "supported": False,
            "error": str(e)
        }

def cleanup_temp_files():
    """Clean up temporary files"""
    
    temp_dir = Path("temp")
    if temp_dir.exists():
        for file in temp_dir.glob("*"):
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to delete temp file {file}: {str(e)}")

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics"""
    
    import psutil
    import torch
    
    stats = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "gpu_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        stats["gpu_memory_used"] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        stats["gpu_memory_reserved"] = torch.cuda.memory_reserved() / torch.cuda.max_memory_reserved() * 100
    
    return stats

def save_config(config: Dict[str, Any], file_path: str = "config.json"):
    """Save configuration to file"""
    
    try:
        with open(file_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Configuration saved to {file_path}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save configuration: {str(e)}")
        raise 