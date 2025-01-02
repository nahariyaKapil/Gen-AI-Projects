import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class IdentityPreservor:
    """
    Identity preservation using face encodings and similarity measures
    """
    
    def __init__(self, model_name: str = "hog"):
        """
        Initialize the identity preservor
        
        Args:
            model_name: Face recognition model ("hog" or "cnn")
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.similarity_threshold = 0.6
        self._loaded = True
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Identity preservor initialized with model: {model_name}")
    
    def is_loaded(self) -> bool:
        """Check if the identity preservor is loaded"""
        return self._loaded
    
    def extract_face_encoding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract face encoding from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Face encoding as numpy array or None if no face found
        """
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image, model=self.model_name)
            
            if not face_locations:
                logger.warning(f"No face found in {image_path}")
                return None
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                logger.warning(f"Could not generate face encoding for {image_path}")
                return None
            
            # Return the first face encoding
            encoding = face_encodings[0]
            logger.info(f"Successfully extracted face encoding from {image_path}")
            return encoding
            
        except Exception as e:
            logger.error(f"Error extracting face encoding: {str(e)}")
            return None
    
    def calculate_identity_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Calculate identity similarity between two face encodings
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            # Calculate cosine similarity
            similarity = cosine_similarity([encoding1], [encoding2])[0][0]
            
            # Convert to 0-1 scale (cosine similarity is -1 to 1)
            similarity_score = (similarity + 1) / 2
            
            return similarity_score
            
        except Exception as e:
            logger.error(f"Error calculating identity similarity: {str(e)}")
            return 0.0
    
    def validate_identity_preservation(self, original_path: str, enhanced_path: str) -> Dict:
        """
        Validate that identity is preserved in enhanced image
        
        Args:
            original_path: Path to original image
            enhanced_path: Path to enhanced image
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Extract encodings
            original_encoding = self.extract_face_encoding(original_path)
            enhanced_encoding = self.extract_face_encoding(enhanced_path)
            
            if original_encoding is None:
                return {
                    "identity_preserved": False,
                    "similarity_score": 0.0,
                    "issue": "Could not extract face encoding from original image"
                }
            
            if enhanced_encoding is None:
                return {
                    "identity_preserved": False,
                    "similarity_score": 0.0,
                    "issue": "Could not extract face encoding from enhanced image"
                }
            
            # Calculate similarity
            similarity = self.calculate_identity_similarity(original_encoding, enhanced_encoding)
            
            # Determine if identity is preserved
            identity_preserved = similarity >= self.similarity_threshold
            
            # Generate feedback
            if identity_preserved:
                feedback = "Identity well preserved"
            elif similarity >= 0.4:
                feedback = "Identity partially preserved - minor variations"
            else:
                feedback = "Identity significantly altered - major changes detected"
            
            return {
                "identity_preserved": identity_preserved,
                "similarity_score": similarity,
                "threshold": self.similarity_threshold,
                "feedback": feedback,
                "original_encoding": original_encoding,
                "enhanced_encoding": enhanced_encoding
            }
            
        except Exception as e:
            logger.error(f"Error validating identity preservation: {str(e)}")
            return {
                "identity_preserved": False,
                "similarity_score": 0.0,
                "issue": f"Validation error: {str(e)}"
            }
    
    def create_identity_mask(self, image_path: str, face_bbox: Dict) -> np.ndarray:
        """
        Create identity-preserving mask for selective enhancement
        
        Args:
            image_path: Path to the image file
            face_bbox: Face bounding box
            
        Returns:
            Identity mask as numpy array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            h, w = image.shape[:2]
            
            # Create base mask
            mask = np.zeros((h, w), dtype=np.float32)
            
            # Define regions with different preservation levels
            # Eyes and nose - high preservation
            eye_nose_region = self._get_eye_nose_region(face_bbox)
            mask[eye_nose_region[1]:eye_nose_region[3], eye_nose_region[0]:eye_nose_region[2]] = 1.0
            
            # Mouth - medium preservation
            mouth_region = self._get_mouth_region(face_bbox)
            mask[mouth_region[1]:mouth_region[3], mouth_region[0]:mouth_region[2]] = 0.7
            
            # Face contour - low preservation (allow more modification)
            face_contour = self._get_face_contour_region(face_bbox)
            mask[face_contour[1]:face_contour[3], face_contour[0]:face_contour[2]] = 0.3
            
            # Apply Gaussian blur for smooth transitions
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error creating identity mask: {str(e)}")
            # Return default mask
            image = cv2.imread(image_path)
            if image is not None:
                return np.ones(image.shape[:2], dtype=np.float32) * 0.5
            return np.ones((512, 512), dtype=np.float32) * 0.5
    
    def _get_eye_nose_region(self, face_bbox: Dict) -> Tuple[int, int, int, int]:
        """Get eye and nose region coordinates"""
        x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['width'], face_bbox['height']
        
        # Eyes and nose region (upper 60% of face)
        region_x = x + int(w * 0.1)
        region_y = y + int(h * 0.1)
        region_w = int(w * 0.8)
        region_h = int(h * 0.6)
        
        return (region_x, region_y, region_x + region_w, region_y + region_h)
    
    def _get_mouth_region(self, face_bbox: Dict) -> Tuple[int, int, int, int]:
        """Get mouth region coordinates"""
        x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['width'], face_bbox['height']
        
        # Mouth region (lower 30% of face)
        region_x = x + int(w * 0.2)
        region_y = y + int(h * 0.6)
        region_w = int(w * 0.6)
        region_h = int(h * 0.3)
        
        return (region_x, region_y, region_x + region_w, region_y + region_h)
    
    def _get_face_contour_region(self, face_bbox: Dict) -> Tuple[int, int, int, int]:
        """Get face contour region coordinates"""
        x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['width'], face_bbox['height']
        
        # Expand bbox slightly for contour
        padding = 0.1
        region_x = max(0, x - int(w * padding))
        region_y = max(0, y - int(h * padding))
        region_w = int(w * (1 + 2 * padding))
        region_h = int(h * (1 + 2 * padding))
        
        return (region_x, region_y, region_x + region_w, region_y + region_h)
    
    def adaptive_enhancement_weights(self, original_path: str, enhanced_path: str, 
                                   target_similarity: float = 0.8) -> Dict:
        """
        Calculate adaptive enhancement weights based on identity preservation
        
        Args:
            original_path: Path to original image
            enhanced_path: Path to enhanced image
            target_similarity: Target similarity score
            
        Returns:
            Dictionary with enhancement weights and recommendations
        """
        try:
            # Get identity validation
            validation = self.validate_identity_preservation(original_path, enhanced_path)
            
            if not validation.get("identity_preserved", False):
                current_similarity = validation.get("similarity_score", 0.0)
                
                # Calculate adjustment factor
                if current_similarity < target_similarity:
                    # Reduce enhancement strength
                    adjustment_factor = current_similarity / target_similarity
                    
                    return {
                        "enhancement_weight": min(0.9, adjustment_factor),
                        "identity_weight": 1.0 - adjustment_factor,
                        "recommendation": "Reduce enhancement strength to preserve identity",
                        "current_similarity": current_similarity,
                        "target_similarity": target_similarity
                    }
            
            # Identity is well preserved
            return {
                "enhancement_weight": 1.0,
                "identity_weight": 0.0,
                "recommendation": "Identity well preserved - full enhancement can be applied",
                "current_similarity": validation.get("similarity_score", 1.0),
                "target_similarity": target_similarity
            }
            
        except Exception as e:
            logger.error(f"Error calculating adaptive weights: {str(e)}")
            return {
                "enhancement_weight": 0.5,
                "identity_weight": 0.5,
                "recommendation": f"Error occurred: {str(e)} - using balanced weights",
                "current_similarity": 0.5,
                "target_similarity": target_similarity
            }
    
    def generate_identity_report(self, original_path: str, enhanced_path: str) -> Dict:
        """
        Generate comprehensive identity preservation report
        
        Args:
            original_path: Path to original image
            enhanced_path: Path to enhanced image
            
        Returns:
            Detailed identity preservation report
        """
        try:
            # Get validation results
            validation = self.validate_identity_preservation(original_path, enhanced_path)
            
            # Calculate detailed metrics
            similarity_score = validation.get("similarity_score", 0.0)
            identity_preserved = validation.get("identity_preserved", False)
            
            # Generate quality assessment
            if similarity_score >= 0.9:
                quality_rating = "Excellent"
                quality_desc = "Identity perfectly preserved"
            elif similarity_score >= 0.8:
                quality_rating = "Very Good"
                quality_desc = "Identity well preserved with minor variations"
            elif similarity_score >= 0.6:
                quality_rating = "Good"
                quality_desc = "Identity adequately preserved"
            elif similarity_score >= 0.4:
                quality_rating = "Fair"
                quality_desc = "Identity partially preserved with noticeable changes"
            else:
                quality_rating = "Poor"
                quality_desc = "Identity significantly altered"
            
            # Generate recommendations
            recommendations = []
            if similarity_score < 0.6:
                recommendations.append("Consider reducing enhancement strength")
                recommendations.append("Focus on lighting and color adjustments rather than structural changes")
            
            if similarity_score < 0.4:
                recommendations.append("Use identity-preserving masks for selective enhancement")
                recommendations.append("Apply enhancement in multiple passes with validation")
            
            return {
                "identity_preserved": identity_preserved,
                "similarity_score": similarity_score,
                "quality_rating": quality_rating,
                "quality_description": quality_desc,
                "threshold_used": self.similarity_threshold,
                "recommendations": recommendations,
                "detailed_metrics": {
                    "face_recognition_model": self.model_name,
                    "similarity_threshold": self.similarity_threshold,
                    "pass_threshold": identity_preserved
                },
                "validation_details": validation
            }
            
        except Exception as e:
            logger.error(f"Error generating identity report: {str(e)}")
            return {
                "identity_preserved": False,
                "similarity_score": 0.0,
                "quality_rating": "Error",
                "quality_description": f"Could not generate report: {str(e)}",
                "recommendations": ["Please try again with valid images"]
            }
    
    def set_similarity_threshold(self, threshold: float):
        """
        Set the similarity threshold for identity preservation
        
        Args:
            threshold: Similarity threshold (0-1)
        """
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
            logger.info(f"Similarity threshold set to {threshold}")
        else:
            logger.warning(f"Invalid threshold {threshold}, must be between 0 and 1")
    
    def compare_multiple_enhancements(self, original_path: str, 
                                    enhanced_paths: List[str]) -> Dict:
        """
        Compare multiple enhancement results for identity preservation
        
        Args:
            original_path: Path to original image
            enhanced_paths: List of paths to enhanced images
            
        Returns:
            Comparison results with rankings
        """
        try:
            results = []
            
            for i, enhanced_path in enumerate(enhanced_paths):
                validation = self.validate_identity_preservation(original_path, enhanced_path)
                results.append({
                    "index": i,
                    "path": enhanced_path,
                    "similarity_score": validation.get("similarity_score", 0.0),
                    "identity_preserved": validation.get("identity_preserved", False),
                    "validation": validation
                })
            
            # Sort by similarity score
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Add rankings
            for i, result in enumerate(results):
                result["rank"] = i + 1
            
            return {
                "total_comparisons": len(enhanced_paths),
                "best_result": results[0] if results else None,
                "worst_result": results[-1] if results else None,
                "all_results": results,
                "average_similarity": sum(r["similarity_score"] for r in results) / len(results) if results else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error comparing enhancements: {str(e)}")
            return {
                "total_comparisons": 0,
                "error": str(e)
            } 