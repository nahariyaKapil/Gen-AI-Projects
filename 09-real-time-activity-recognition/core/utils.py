import logging
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import time
import numpy as np
import cv2
import torch
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Set up handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    else:
        # Default log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_file = logs_dir / f"activity_recognition_{timestamp}.log"
        file_handler = logging.FileHandler(default_log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")
    
    return logger

def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    
    if config_path is None:
        config_path = "config.yaml"
    
    # Default configuration
    default_config = {
        "models": {
            "i3d": {
                "input_size": [224, 224],
                "num_frames": 16,
                "pretrained": True
            },
            "slowfast": {
                "input_size": [224, 224],
                "num_frames": 32,
                "pretrained": True
            },
            "vivit": {
                "input_size": [224, 224],
                "num_frames": 16,
                "pretrained": True
            }
        },
        "detection": {
            "yolo_model": "yolov8n",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
            "max_detections": 100
        },
        "processing": {
            "fps_limit": 30,
            "buffer_size": 16,
            "batch_size": 1,
            "num_workers": 4
        },
        "optimization": {
            "use_onnx": True,
            "use_tensorrt": False,
            "use_quantization": False,
            "enable_gpu": True
        },
        "ui": {
            "theme": "light",
            "display_fps": True,
            "show_confidence": True,
            "overlay_detections": True
        },
        "logging": {
            "level": "INFO",
            "file": None,
            "console": True
        }
    }
    
    try:
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                    loaded_config = yaml.safe_load(f)
                else:
                    loaded_config = json.load(f)
            
            # Merge with default config
            config = deep_merge(default_config, loaded_config)
            
            logger = logging.getLogger(__name__)
            logger.info(f"Configuration loaded from: {config_path}")
            
        else:
            config = default_config
            logger = logging.getLogger(__name__)
            logger.warning(f"Configuration file not found: {config_path}, using defaults")
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading configuration: {str(e)}, using defaults")
        config = default_config
    
    return config

def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def format_confidence(confidence: float) -> str:
    """
    Format confidence value for display
    
    Args:
        confidence: Confidence value (0-1)
        
    Returns:
        Formatted confidence string
    """
    return f"{confidence:.2f}"

def format_time(seconds: float) -> str:
    """
    Format time duration
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"

def calculate_fps(frame_times: List[float]) -> float:
    """
    Calculate FPS from frame times
    
    Args:
        frame_times: List of frame processing times
        
    Returns:
        FPS value
    """
    if not frame_times:
        return 0.0
    
    avg_time = np.mean(frame_times)
    return 1.0 / avg_time if avg_time > 0 else 0.0

def create_color_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Create color palette for visualization
    
    Args:
        num_colors: Number of colors to generate
        
    Returns:
        List of RGB color tuples
    """
    colors = []
    
    # Predefined colors for common classes
    predefined_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (255, 192, 203), # Pink
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (128, 0, 0),    # Maroon
        (0, 128, 128),  # Teal
        (192, 192, 192), # Silver
    ]
    
    # Use predefined colors first
    for i in range(min(num_colors, len(predefined_colors))):
        colors.append(predefined_colors[i])
    
    # Generate additional colors if needed
    for i in range(len(predefined_colors), num_colors):
        # Generate colors using HSV color space
        hue = (i * 137.5) % 360  # Golden angle approximation
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.8 + (i % 2) * 0.2
        
        # Convert HSV to RGB
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
        color = tuple(int(c * 255) for c in rgb)
        colors.append(color)
    
    return colors

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    try:
        if maintain_aspect_ratio:
            # Calculate scaling factor
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            scale = min(target_w / w, target_h / h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h))
            
            # Add padding if needed
            if new_w != target_w or new_h != target_h:
                # Create padded image
                padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
                
                # Calculate padding offsets
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                
                # Place resized image in center
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                return padded
            else:
                return resized
        else:
            # Direct resize without maintaining aspect ratio
            return cv2.resize(image, target_size)
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error resizing image: {str(e)}")
        return image

def save_results(results: Dict, output_path: str):
    """
    Save results to file
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = make_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error saving results: {str(e)}")

def make_serializable(obj: Any) -> Any:
    """
    Make object JSON serializable
    
    Args:
        obj: Object to make serializable
        
    Returns:
        Serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    else:
        return obj

def get_device_info() -> Dict:
    """
    Get device information
    
    Returns:
        Device information dictionary
    """
    info = {
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "cuda_devices": []
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        
        for i in range(torch.cuda.device_count()):
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(i),
                "memory_cached": torch.cuda.memory_reserved(i)
            }
            info["cuda_devices"].append(device_info)
    
    return info

def benchmark_function(func, *args, num_iterations: int = 100, **kwargs) -> Dict:
    """
    Benchmark a function
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        num_iterations: Number of iterations
        **kwargs: Function keyword arguments
        
    Returns:
        Benchmark results
    """
    try:
        times = []
        
        # Warm-up runs
        for _ in range(min(10, num_iterations)):
            func(*args, **kwargs)
        
        # Benchmark runs
        for _ in range(num_iterations):
            start_time = time.time()
            func(*args, **kwargs)
            times.append(time.time() - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        return {
            "function": func.__name__,
            "num_iterations": num_iterations,
            "average_time": avg_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "total_time": sum(times)
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error benchmarking function: {str(e)}")
        return {"error": str(e)}

def create_video_writer(output_path: str, fps: float, frame_size: Tuple[int, int],
                       codec: str = 'mp4v') -> cv2.VideoWriter:
    """
    Create video writer
    
    Args:
        output_path: Output video path
        fps: Frame rate
        frame_size: Frame size (width, height)
        codec: Video codec
        
    Returns:
        VideoWriter object
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, frame_size)
        
        if not writer.isOpened():
            raise ValueError(f"Could not create video writer for {output_path}")
        
        return writer
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error creating video writer: {str(e)}")
        raise

def validate_input_format(input_data: Any, expected_format: str) -> bool:
    """
    Validate input data format
    
    Args:
        input_data: Input data to validate
        expected_format: Expected format ('image', 'video', 'tensor', etc.)
        
    Returns:
        True if format is valid
    """
    try:
        if expected_format == 'image':
            return isinstance(input_data, np.ndarray) and len(input_data.shape) == 3
        elif expected_format == 'video':
            return isinstance(input_data, np.ndarray) and len(input_data.shape) == 4
        elif expected_format == 'tensor':
            return isinstance(input_data, torch.Tensor)
        elif expected_format == 'path':
            return isinstance(input_data, (str, Path)) and Path(input_data).exists()
        else:
            return False
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error validating input format: {str(e)}")
        return False

def get_memory_usage() -> Dict:
    """
    Get memory usage information
    
    Returns:
        Memory usage dictionary
    """
    import psutil
    
    try:
        # System memory
        memory = psutil.virtual_memory()
        
        info = {
            "system_memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free
            }
        }
        
        # GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = {}
            for i in range(torch.cuda.device_count()):
                gpu_memory[f"gpu_{i}"] = {
                    "total": torch.cuda.get_device_properties(i).total_memory,
                    "allocated": torch.cuda.memory_allocated(i),
                    "cached": torch.cuda.memory_reserved(i)
                }
            info["gpu_memory"] = gpu_memory
        
        return info
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting memory usage: {str(e)}")
        return {"error": str(e)}

def create_progress_bar(total: int, desc: str = "Processing") -> Any:
    """
    Create progress bar
    
    Args:
        total: Total number of items
        desc: Description
        
    Returns:
        Progress bar object
    """
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc)
    except ImportError:
        # Fallback simple progress tracker
        class SimpleProgress:
            def __init__(self, total, desc):
                self.total = total
                self.desc = desc
                self.current = 0
                
            def update(self, n=1):
                self.current += n
                percent = (self.current / self.total) * 100
                print(f"\r{self.desc}: {percent:.1f}% ({self.current}/{self.total})", end="")
                
            def close(self):
                print()  # New line
                
        return SimpleProgress(total, desc)

def cleanup_temp_files(temp_dir: str = "temp"):
    """
    Clean up temporary files
    
    Args:
        temp_dir: Temporary directory path
    """
    try:
        temp_path = Path(temp_dir)
        
        if temp_path.exists():
            import shutil
            shutil.rmtree(temp_path)
            
            logger = logging.getLogger(__name__)
            logger.info(f"Cleaned up temporary files in: {temp_dir}")
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error cleaning up temporary files: {str(e)}")

# Activity label mappings
ACTIVITY_LABELS = {
    "walking": "Walking",
    "running": "Running",
    "jumping": "Jumping",
    "sitting": "Sitting",
    "standing": "Standing",
    "dancing": "Dancing",
    "cooking": "Cooking",
    "eating": "Eating",
    "drinking": "Drinking",
    "reading": "Reading",
    "writing": "Writing",
    "typing": "Typing",
    "talking": "Talking",
    "singing": "Singing",
    "exercising": "Exercising",
    "stretching": "Stretching",
    "yoga": "Yoga",
    "playing_guitar": "Playing Guitar",
    "playing_piano": "Playing Piano",
    "clapping": "Clapping",
    "waving": "Waving",
    "pointing": "Pointing",
    "shaking_hands": "Shaking Hands",
    "hugging": "Hugging",
    "kissing": "Kissing",
    "sleeping": "Sleeping",
    "lying_down": "Lying Down",
    "getting_up": "Getting Up",
    "falling_down": "Falling Down",
    "climbing_stairs": "Climbing Stairs"
}

def get_activity_label(activity_key: str) -> str:
    """
    Get formatted activity label
    
    Args:
        activity_key: Activity key
        
    Returns:
        Formatted activity label
    """
    return ACTIVITY_LABELS.get(activity_key, activity_key.replace("_", " ").title()) 