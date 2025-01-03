import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Object detection using YOLOv8 for activity recognition overlay
    """
    
    def __init__(self, model_size: str = "yolov8n"):
        """
        Initialize object detector
        
        Args:
            model_size: YOLOv8 model size ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
        """
        self.model_size = model_size
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # COCO class names
        self.class_names = self._get_coco_classes()
        
        # Human-related classes from COCO dataset
        self.human_classes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe"
        }
        
        # Activity-relevant objects
        self.activity_objects = {
            "sports": [32, 33, 34, 35, 36, 37, 38, 39, 40],  # sports ball, kite, baseball bat, etc.
            "kitchen": [47, 48, 49, 50, 51, 52, 53, 54, 55],  # apple, sandwich, orange, etc.
            "furniture": [56, 57, 58, 59, 60, 61],  # chair, couch, potted plant, etc.
            "electronics": [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]  # tv, laptop, mouse, etc.
        }
        
        # Detection parameters
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.max_detections = 100
        
        logger.info(f"Object detector initialized with {model_size} on {self.device}")
    
    def _get_coco_classes(self) -> List[str]:
        """Get COCO class names"""
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
    
    def load_model(self) -> bool:
        """
        Load YOLOv8 model
        
        Returns:
            True if model loaded successfully
        """
        try:
            # Try to import ultralytics
            try:
                from ultralytics import YOLO
                self.model = YOLO(f"{self.model_size}.pt")
                self.model.to(self.device)
                logger.info(f"YOLOv8 model ({self.model_size}) loaded successfully")
                return True
                
            except ImportError:
                logger.warning("ultralytics not available, using dummy detector")
                self.model = self._create_dummy_detector()
                return True
                
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {str(e)}")
            self.model = self._create_dummy_detector()
            return False
    
    def _create_dummy_detector(self):
        """Create dummy detector for demonstration"""
        class DummyDetector:
            def __init__(self):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            def predict(self, image, conf=0.5, iou=0.45, max_det=100):
                # Return dummy detection results
                height, width = image.shape[:2]
                
                # Create some dummy detections
                detections = []
                
                # Add a dummy person detection
                if np.random.random() > 0.3:  # 70% chance of detecting a person
                    x1, y1 = width * 0.3, height * 0.2
                    x2, y2 = width * 0.7, height * 0.8
                    conf = 0.8 + np.random.random() * 0.15
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': 0,  # person
                        'class_name': 'person'
                    }
                    detections.append(detection)
                
                # Add some random object detections
                for _ in range(np.random.randint(0, 3)):
                    x1 = np.random.randint(0, width // 2)
                    y1 = np.random.randint(0, height // 2)
                    x2 = x1 + np.random.randint(50, 200)
                    y2 = y1 + np.random.randint(50, 200)
                    
                    # Keep within bounds
                    x2 = min(x2, width)
                    y2 = min(y2, height)
                    
                    class_id = np.random.randint(1, 10)
                    conf = 0.5 + np.random.random() * 0.3
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': class_id,
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
                    }
                    detections.append(detection)
                
                return [{'detections': detections}]
        
        return DummyDetector()
    
    def detect_objects(self, image: np.ndarray) -> Dict:
        """
        Detect objects in image
        
        Args:
            image: Input image
            
        Returns:
            Detection results
        """
        try:
            if self.model is None:
                if not self.load_model():
                    return {"error": "Failed to load model", "detections": []}
            
            start_time = time.time()
            
            # Run inference
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            inference_time = time.time() - start_time
            
            # Process results
            detections = self._process_results(results)
            
            # Filter for human-related detections
            human_detections = self._filter_human_detections(detections)
            
            return {
                "detections": detections,
                "human_detections": human_detections,
                "inference_time": inference_time,
                "num_detections": len(detections),
                "num_human_detections": len(human_detections),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}")
            return {
                "error": str(e),
                "detections": [],
                "success": False
            }
    
    def _process_results(self, results) -> List[Dict]:
        """Process detection results"""
        try:
            detections = []
            
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                # Real YOLOv8 results
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().item()
                    class_id = int(boxes.cls[i].cpu().item())
                    
                    detection = {
                        "bbox": bbox.tolist(),
                        "confidence": confidence,
                        "class": class_id,
                        "class_name": self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                    }
                    detections.append(detection)
            
            else:
                # Dummy detector results
                detections = results[0].get('detections', [])
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing detection results: {str(e)}")
            return []
    
    def _filter_human_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections for human-related objects"""
        human_detections = []
        
        for detection in detections:
            class_id = detection["class"]
            if class_id in self.human_classes or class_id == 0:  # person class
                human_detections.append(detection)
        
        return human_detections
    
    def draw_detections(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        Draw detection results on image
        
        Args:
            image: Input image
            detection_result: Detection results
            
        Returns:
            Image with drawn detections
        """
        try:
            if not detection_result.get("success", False):
                return image
            
            result_image = image.copy()
            detections = detection_result.get("detections", [])
            
            # Color map for different classes
            colors = [
                (255, 0, 0),    # Red for person
                (0, 255, 0),    # Green for vehicles
                (0, 0, 255),    # Blue for animals
                (255, 255, 0),  # Yellow for sports
                (255, 0, 255),  # Magenta for kitchen
                (0, 255, 255),  # Cyan for furniture
                (128, 0, 128),  # Purple for electronics
                (255, 165, 0),  # Orange for others
            ]
            
            for detection in detections:
                bbox = detection["bbox"]
                confidence = detection["confidence"]
                class_name = detection["class_name"]
                class_id = detection["class"]
                
                # Skip low confidence detections
                if confidence < self.conf_threshold:
                    continue
                
                # Get color based on class
                color = colors[class_id % len(colors)]
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(result_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            return image
    
    def get_activity_context(self, detection_result: Dict) -> Dict:
        """
        Get activity context from detections
        
        Args:
            detection_result: Detection results
            
        Returns:
            Activity context information
        """
        try:
            detections = detection_result.get("detections", [])
            
            context = {
                "has_person": False,
                "person_count": 0,
                "activity_objects": [],
                "scene_type": "unknown",
                "confidence_score": 0.0
            }
            
            object_counts = {}
            total_confidence = 0
            
            for detection in detections:
                class_id = detection["class"]
                class_name = detection["class_name"]
                confidence = detection["confidence"]
                
                # Count objects
                if class_name not in object_counts:
                    object_counts[class_name] = 0
                object_counts[class_name] += 1
                
                total_confidence += confidence
                
                # Check for person
                if class_id == 0:  # person
                    context["has_person"] = True
                    context["person_count"] += 1
                
                # Check for activity-relevant objects
                for activity_type, class_ids in self.activity_objects.items():
                    if class_id in class_ids:
                        context["activity_objects"].append({
                            "type": activity_type,
                            "object": class_name,
                            "confidence": confidence
                        })
            
            # Determine scene type
            context["scene_type"] = self._determine_scene_type(object_counts)
            
            # Calculate average confidence
            if len(detections) > 0:
                context["confidence_score"] = total_confidence / len(detections)
            
            context["object_counts"] = object_counts
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting activity context: {str(e)}")
            return {"error": str(e)}
    
    def _determine_scene_type(self, object_counts: Dict) -> str:
        """Determine scene type from object counts"""
        try:
            # Define scene indicators
            scene_indicators = {
                "kitchen": ["bowl", "cup", "fork", "knife", "spoon", "apple", "banana", "sandwich"],
                "living_room": ["couch", "tv", "chair", "remote", "book"],
                "office": ["laptop", "keyboard", "mouse", "chair", "book"],
                "outdoor": ["bicycle", "car", "tree", "bench", "bird"],
                "sports": ["sports ball", "tennis racket", "skateboard", "surfboard"],
                "bedroom": ["bed", "pillow", "clock"],
                "bathroom": ["toilet", "sink", "toothbrush"]
            }
            
            scene_scores = {}
            
            for scene, indicators in scene_indicators.items():
                score = 0
                for indicator in indicators:
                    if indicator in object_counts:
                        score += object_counts[indicator]
                scene_scores[scene] = score
            
            # Return scene with highest score
            if scene_scores:
                best_scene = max(scene_scores, key=scene_scores.get)
                if scene_scores[best_scene] > 0:
                    return best_scene
            
            return "general"
            
        except Exception as e:
            logger.error(f"Error determining scene type: {str(e)}")
            return "unknown"
    
    def set_detection_parameters(self, conf_threshold: float = None, 
                                iou_threshold: float = None, 
                                max_detections: int = None):
        """
        Set detection parameters
        
        Args:
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections
        """
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        if max_detections is not None:
            self.max_detections = max_detections
        
        logger.info(f"Detection parameters updated: conf={self.conf_threshold}, "
                   f"iou={self.iou_threshold}, max_det={self.max_detections}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_size": self.model_size,
            "device": str(self.device),
            "num_classes": len(self.class_names),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "model_loaded": self.model is not None
        }
    
    def benchmark_detection(self, image: np.ndarray, num_iterations: int = 50) -> Dict:
        """
        Benchmark detection performance
        
        Args:
            image: Test image
            num_iterations: Number of iterations
            
        Returns:
            Benchmark results
        """
        try:
            if self.model is None:
                return {"error": "Model not loaded"}
            
            times = []
            
            # Warm-up runs
            for _ in range(5):
                self.detect_objects(image)
            
            # Benchmark runs
            for _ in range(num_iterations):
                start_time = time.time()
                self.detect_objects(image)
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / avg_time
            
            return {
                "model_size": self.model_size,
                "num_iterations": num_iterations,
                "average_time": avg_time,
                "std_time": std_time,
                "fps": fps,
                "device": str(self.device)
            }
            
        except Exception as e:
            logger.error(f"Error benchmarking detection: {str(e)}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Object detector cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.class_names.copy()
    
    def get_human_classes(self) -> Dict:
        """Get human-related classes"""
        return self.human_classes.copy()
    
    def get_activity_objects(self) -> Dict:
        """Get activity-relevant objects"""
        return self.activity_objects.copy() 