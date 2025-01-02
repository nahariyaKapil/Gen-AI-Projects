import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Face detection and landmark extraction using MediaPipe
    """
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
        
        # Initialize face mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self._loaded = True
        logger.info("Face detector initialized successfully")
    
    def is_loaded(self) -> bool:
        """Check if the face detector is loaded"""
        return self._loaded
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        Detect faces in an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face detection results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detection.process(image_rgb)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    # Extract bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    bbox = {
                        'x': int(bboxC.xmin * iw),
                        'y': int(bboxC.ymin * ih),
                        'width': int(bboxC.width * iw),
                        'height': int(bboxC.height * ih)
                    }
                    
                    # Extract key points
                    keypoints = []
                    for keypoint in detection.location_data.relative_keypoints:
                        keypoints.append({
                            'x': int(keypoint.x * iw),
                            'y': int(keypoint.y * ih)
                        })
                    
                    face_data = {
                        'bbox': bbox,
                        'keypoints': keypoints,
                        'confidence': detection.score[0],
                        'relative_bbox': {
                            'xmin': bboxC.xmin,
                            'ymin': bboxC.ymin,
                            'width': bboxC.width,
                            'height': bboxC.height
                        }
                    }
                    faces.append(face_data)
            
            logger.info(f"Detected {len(faces)} faces in {image_path}")
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            raise
    
    def extract_face_landmarks(self, image_path: str) -> Dict:
        """
        Extract detailed face landmarks using MediaPipe Face Mesh
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing face landmarks and analysis
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with face mesh
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return {"landmarks": [], "face_found": False}
            
            # Extract landmarks for the first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to pixel coordinates
            ih, iw, _ = image.shape
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append({
                    'x': int(landmark.x * iw),
                    'y': int(landmark.y * ih),
                    'z': landmark.z
                })
            
            # Extract key facial features
            features = self._extract_facial_features(landmarks, iw, ih)
            
            return {
                "landmarks": landmarks,
                "face_found": True,
                "features": features,
                "image_shape": (ih, iw)
            }
            
        except Exception as e:
            logger.error(f"Error extracting face landmarks: {str(e)}")
            raise
    
    def _extract_facial_features(self, landmarks: List[Dict], width: int, height: int) -> Dict:
        """
        Extract key facial features from landmarks
        
        Args:
            landmarks: List of face landmarks
            width: Image width
            height: Image height
            
        Returns:
            Dictionary of facial features
        """
        try:
            # Key landmark indices for MediaPipe Face Mesh
            LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            NOSE = [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 237, 238, 239, 240, 241, 242]
            MOUTH = [0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 0, 269, 270, 267, 271, 272]
            FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            
            # Calculate centers
            left_eye_center = self._calculate_center([landmarks[i] for i in LEFT_EYE])
            right_eye_center = self._calculate_center([landmarks[i] for i in RIGHT_EYE])
            nose_center = self._calculate_center([landmarks[i] for i in NOSE])
            mouth_center = self._calculate_center([landmarks[i] for i in MOUTH])
            
            # Calculate face angle
            eye_center_x = (left_eye_center['x'] + right_eye_center['x']) / 2
            eye_center_y = (left_eye_center['y'] + right_eye_center['y']) / 2
            
            angle = np.arctan2(
                right_eye_center['y'] - left_eye_center['y'],
                right_eye_center['x'] - left_eye_center['x']
            ) * 180 / np.pi
            
            # Calculate face dimensions
            face_width = abs(right_eye_center['x'] - left_eye_center['x']) * 2.5
            face_height = abs(mouth_center['y'] - eye_center_y) * 2.5
            
            # Face quality metrics
            symmetry_score = self._calculate_symmetry_score(landmarks)
            lighting_score = self._calculate_lighting_score(landmarks)
            
            return {
                "left_eye_center": left_eye_center,
                "right_eye_center": right_eye_center,
                "nose_center": nose_center,
                "mouth_center": mouth_center,
                "face_angle": angle,
                "face_width": face_width,
                "face_height": face_height,
                "symmetry_score": symmetry_score,
                "lighting_score": lighting_score,
                "eye_distance": np.sqrt(
                    (right_eye_center['x'] - left_eye_center['x'])**2 +
                    (right_eye_center['y'] - left_eye_center['y'])**2
                )
            }
            
        except Exception as e:
            logger.error(f"Error extracting facial features: {str(e)}")
            return {}
    
    def _calculate_center(self, points: List[Dict]) -> Dict:
        """Calculate center point of a list of landmarks"""
        if not points:
            return {'x': 0, 'y': 0}
        
        x = sum(p['x'] for p in points) / len(points)
        y = sum(p['y'] for p in points) / len(points)
        return {'x': x, 'y': y}
    
    def _calculate_symmetry_score(self, landmarks: List[Dict]) -> float:
        """
        Calculate facial symmetry score (0-1, higher is more symmetric)
        """
        try:
            # Simple symmetry calculation based on key points
            # This is a simplified version - in practice, you'd use more sophisticated methods
            center_x = sum(p['x'] for p in landmarks) / len(landmarks)
            
            # Calculate variance of points from center
            left_points = [p for p in landmarks if p['x'] < center_x]
            right_points = [p for p in landmarks if p['x'] > center_x]
            
            if not left_points or not right_points:
                return 0.5
            
            # Simple symmetry metric
            symmetry = 1.0 - abs(len(left_points) - len(right_points)) / len(landmarks)
            return max(0.0, min(1.0, symmetry))
            
        except Exception:
            return 0.5
    
    def _calculate_lighting_score(self, landmarks: List[Dict]) -> float:
        """
        Calculate lighting quality score (0-1, higher is better lighting)
        """
        try:
            # This is a placeholder - in practice, you'd analyze image brightness,
            # contrast, and shadow patterns
            return 0.75  # Default good lighting score
            
        except Exception:
            return 0.5
    
    def crop_face(self, image_path: str, face_bbox: Dict, padding: float = 0.2) -> np.ndarray:
        """
        Crop face from image with padding
        
        Args:
            image_path: Path to the image file
            face_bbox: Face bounding box
            padding: Padding factor (0.2 = 20% padding)
            
        Returns:
            Cropped face image as numpy array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            ih, iw, _ = image.shape
            
            # Calculate crop region with padding
            x = max(0, int(face_bbox['x'] - face_bbox['width'] * padding))
            y = max(0, int(face_bbox['y'] - face_bbox['height'] * padding))
            w = min(iw - x, int(face_bbox['width'] * (1 + 2 * padding)))
            h = min(ih - y, int(face_bbox['height'] * (1 + 2 * padding)))
            
            # Crop face
            cropped_face = image[y:y+h, x:x+w]
            
            return cropped_face
            
        except Exception as e:
            logger.error(f"Error cropping face: {str(e)}")
            raise
    
    def analyze_face_quality(self, image_path: str) -> Dict:
        """
        Analyze face quality for enhancement suitability
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Detect faces
            faces = self.detect_faces(image_path)
            if not faces:
                return {
                    "quality_score": 0.0,
                    "issues": ["No face detected"],
                    "recommendations": ["Ensure face is clearly visible in the image"]
                }
            
            # Extract landmarks
            landmarks_data = self.extract_face_landmarks(image_path)
            
            # Calculate quality metrics
            face = faces[0]  # Use first face
            quality_score = face['confidence']
            
            issues = []
            recommendations = []
            
            # Check face size
            if face['bbox']['width'] < 100 or face['bbox']['height'] < 100:
                issues.append("Face too small")
                recommendations.append("Use a higher resolution image or crop closer to face")
                quality_score *= 0.8
            
            # Check face angle
            if landmarks_data.get('face_found'):
                features = landmarks_data['features']
                if abs(features.get('face_angle', 0)) > 15:
                    issues.append("Face angle too extreme")
                    recommendations.append("Use a more frontal view of the face")
                    quality_score *= 0.9
                
                # Check symmetry
                if features.get('symmetry_score', 0) < 0.6:
                    issues.append("Face partially occluded or asymmetric")
                    recommendations.append("Ensure face is fully visible and not obstructed")
                    quality_score *= 0.85
            
            # Overall quality assessment
            if quality_score >= 0.8:
                quality_rating = "Excellent"
            elif quality_score >= 0.6:
                quality_rating = "Good"
            elif quality_score >= 0.4:
                quality_rating = "Fair"
            else:
                quality_rating = "Poor"
            
            return {
                "quality_score": quality_score,
                "quality_rating": quality_rating,
                "face_count": len(faces),
                "primary_face": face,
                "landmarks": landmarks_data,
                "issues": issues,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing face quality: {str(e)}")
            return {
                "quality_score": 0.0,
                "issues": [f"Analysis error: {str(e)}"],
                "recommendations": ["Please try with a different image"]
            } 