import logging
import os
from typing import Dict, Any
from pathlib import Path
import json
import yaml

def setup_logging(level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Set up handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        handlers.append(logging.FileHandler(log_dir / "face_enhancer.log"))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Reduce noise from some libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "models": {
            "stable_diffusion": "runwayml/stable-diffusion-v1-5",
            "face_detection": "hog",
            "identity_threshold": 0.6
        },
        "processing": {
            "max_image_size": 2048,
            "default_enhancement_level": 0.8,
            "default_style": "professional",
            "gpu_memory_fraction": 0.8
        },
        "storage": {
            "upload_dir": "uploads",
            "max_file_size": 10485760,  # 10MB
            "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        },
        "api": {
            "max_concurrent_requests": 5,
            "request_timeout": 300,
            "cleanup_interval": 3600
        }
    }
    
    # Try to load configuration file
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)
            
            # Merge with default config
            default_config.update(user_config)
            
        except Exception as e:
            logging.warning(f"Could not load config file {config_path}: {e}")
    
    # Override with environment variables
    env_overrides = {
        "STABLE_DIFFUSION_MODEL": ("models", "stable_diffusion"),
        "FACE_DETECTION_MODEL": ("models", "face_detection"),
        "IDENTITY_THRESHOLD": ("models", "identity_threshold"),
        "MAX_IMAGE_SIZE": ("processing", "max_image_size"),
        "DEFAULT_ENHANCEMENT_LEVEL": ("processing", "default_enhancement_level"),
        "DEFAULT_STYLE": ("processing", "default_style"),
        "GPU_MEMORY_FRACTION": ("processing", "gpu_memory_fraction"),
        "UPLOAD_DIR": ("storage", "upload_dir"),
        "MAX_FILE_SIZE": ("storage", "max_file_size"),
        "MAX_CONCURRENT_REQUESTS": ("api", "max_concurrent_requests"),
        "REQUEST_TIMEOUT": ("api", "request_timeout")
    }
    
    for env_var, (section, key) in env_overrides.items():
        if env_var in os.environ:
            try:
                value = os.environ[env_var]
                # Try to convert to appropriate type
                if key in ["max_image_size", "max_file_size", "max_concurrent_requests", "request_timeout"]:
                    value = int(value)
                elif key in ["identity_threshold", "default_enhancement_level", "gpu_memory_fraction"]:
                    value = float(value)
                
                default_config[section][key] = value
                
            except ValueError as e:
                logging.warning(f"Invalid value for {env_var}: {e}")
    
    return default_config

def validate_image_file(file_path: str, max_size: int = 10485760) -> Dict[str, Any]:
    """
    Validate image file
    
    Args:
        file_path: Path to image file
        max_size: Maximum file size in bytes
        
    Returns:
        Validation result dictionary
    """
    try:
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            return {"valid": False, "error": "File does not exist"}
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > max_size:
            return {"valid": False, "error": f"File size ({file_size} bytes) exceeds maximum ({max_size} bytes)"}
        
        # Check file extension
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        if file_path.suffix.lower() not in allowed_extensions:
            return {"valid": False, "error": f"Unsupported file extension: {file_path.suffix}"}
        
        # Try to open image
        from PIL import Image
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                format = img.format
                mode = img.mode
        except Exception as e:
            return {"valid": False, "error": f"Invalid image file: {str(e)}"}
        
        return {
            "valid": True,
            "file_size": file_size,
            "dimensions": (width, height),
            "format": format,
            "mode": mode
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}

def create_directory_structure(base_dir: str):
    """
    Create necessary directory structure
    
    Args:
        base_dir: Base directory path
    """
    directories = [
        "uploads",
        "logs",
        "models",
        "temp"
    ]
    
    base_path = Path(base_dir)
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)

def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """
    Clean up old files in a directory
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours
    """
    try:
        import time
        from pathlib import Path
        
        directory = Path(directory)
        if not directory.exists():
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned_count = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    cleaned_count += 1
        
        logging.info(f"Cleaned up {cleaned_count} old files from {directory}")
        
    except Exception as e:
        logging.error(f"Error cleaning up files: {str(e)}")

def get_system_info() -> Dict[str, Any]:
    """
    Get system information
    
    Returns:
        System information dictionary
    """
    import platform
    import psutil
    import torch
    
    try:
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            system_info["gpu_info"] = []
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    "device": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_cached": torch.cuda.memory_reserved(i)
                }
                system_info["gpu_info"].append(gpu_info)
        
        return system_info
        
    except Exception as e:
        return {"error": str(e)}

def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human readable format
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def estimate_processing_time(image_size: tuple, style: str = "professional") -> int:
    """
    Estimate processing time based on image size and style
    
    Args:
        image_size: Image dimensions (width, height)
        style: Enhancement style
        
    Returns:
        Estimated time in seconds
    """
    width, height = image_size
    pixels = width * height
    
    # Base time estimates (in seconds)
    base_times = {
        "professional": 30,
        "glamorous": 45,
        "casual": 25,
        "artistic": 60,
        "vintage": 40,
        "modern": 35
    }
    
    base_time = base_times.get(style, 30)
    
    # Adjust for image size
    if pixels > 1024 * 1024:  # > 1MP
        base_time *= 1.5
    elif pixels > 2048 * 2048:  # > 4MP
        base_time *= 2.0
    
    return int(base_time)

def save_processing_report(result: Dict[str, Any], output_path: str):
    """
    Save processing report to file
    
    Args:
        result: Processing result
        output_path: Output file path
    """
    try:
        report = {
            "timestamp": result.get("timestamp"),
            "success": result.get("success", False),
            "processing_time": result.get("processing_time", 0),
            "steps_completed": result.get("steps_completed", []),
            "face_analysis": result.get("face_analysis", {}),
            "enhancement_details": result.get("enhancement_details", {}),
            "validation_results": result.get("validation_results", {}),
            "error": result.get("error")
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
    except Exception as e:
        logging.error(f"Error saving processing report: {str(e)}")

def load_processing_report(report_path: str) -> Dict[str, Any]:
    """
    Load processing report from file
    
    Args:
        report_path: Path to report file
        
    Returns:
        Processing report dictionary
    """
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading processing report: {str(e)}")
        return {}

class PerformanceMonitor:
    """Simple performance monitoring utility"""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
    
    def start(self):
        """Start monitoring"""
        import time
        self.start_time = time.time()
    
    def checkpoint(self, name: str):
        """Add a checkpoint"""
        if self.start_time is None:
            self.start()
        
        import time
        self.checkpoints[name] = time.time() - self.start_time
    
    def get_elapsed(self) -> float:
        """Get total elapsed time"""
        if self.start_time is None:
            return 0.0
        
        import time
        return time.time() - self.start_time
    
    def get_report(self) -> Dict[str, float]:
        """Get performance report"""
        report = {
            "total_time": self.get_elapsed(),
            "checkpoints": self.checkpoints.copy()
        }
        
        # Calculate step times
        sorted_checkpoints = sorted(self.checkpoints.items(), key=lambda x: x[1])
        step_times = {}
        prev_time = 0
        
        for name, time_elapsed in sorted_checkpoints:
            step_times[name] = time_elapsed - prev_time
            prev_time = time_elapsed
        
        report["step_times"] = step_times
        return report 