import os
import logging
import asyncio
from typing import Dict, List, Optional, Union, Any
from PIL import Image, ImageEnhance, ImageFilter
import shutil
from pathlib import Path
import hashlib
import json
import numpy as np
import io

# Try to import cv2, fallback if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Try to import face_recognition, fallback if not available
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Process and manage training images for avatar generation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.webp']
        self.max_image_size = config.get("max_image_size", 1024)
        self.min_image_size = config.get("min_image_size", 256)
        self.target_size = config.get("target_size", 512)
        
        logger.info("Image Processor initialized")
    
    def process_training_images(self, files: List[Any], user_id: str) -> List[str]:
        """Process uploaded training images (works with Streamlit UploadedFile)"""
        try:
            logger.info(f"Processing {len(files)} training images for user {user_id}")
            
            # Create user directory
            user_dir = Path("uploads") / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear existing images
            for existing_file in user_dir.glob("*"):
                existing_file.unlink()
            
            processed_paths = []
            face_encodings = []
            consistency_score = 1.0
            
            for i, file in enumerate(files):
                filename = getattr(file, 'name', f'image_{i}')
                logger.info(f"Processing image {i+1}/{len(files)}: {filename}")
                
                # Validate file
                if not self._is_valid_image_streamlit(file):
                    logger.warning(f"Skipping invalid image: {filename}")
                    continue
                
                # Save temporary file
                temp_path = user_dir / f"temp_{i}_{filename}"
                with open(temp_path, "wb") as buffer:
                    # Handle Streamlit UploadedFile
                    if hasattr(file, 'read'):
                        content = file.read()
                        file.seek(0)  # Reset file pointer
                    else:
                        content = file
                    buffer.write(content)
                
                # Process image
                try:
                    processed_path = self._process_single_image(temp_path, user_id, i)
                    if processed_path:
                        processed_paths.append(processed_path)
                        
                        # Extract face encoding for consistency check
                        if FACE_RECOGNITION_AVAILABLE:
                            face_encoding = self._extract_face_encoding(processed_path)
                            if face_encoding is not None:
                                face_encodings.append(face_encoding)
                    
                except Exception as e:
                    logger.error(f"Error processing image {filename}: {str(e)}")
                
                # Remove temporary file
                if temp_path.exists():
                    temp_path.unlink()
            
            # Validate face consistency
            if len(face_encodings) > 1 and FACE_RECOGNITION_AVAILABLE:
                consistency_score = self._check_face_consistency(face_encodings)
                logger.info(f"Face consistency score: {consistency_score:.2f}")
                
                if consistency_score < 0.6:
                    logger.warning("Low face consistency detected. Training may not be optimal.")
            
            # Save metadata
            metadata = {
                "user_id": user_id,
                "total_images": len(processed_paths),
                "face_consistency": consistency_score,
                "processed_at": str(Path().cwd()),
                "image_paths": processed_paths
            }
            
            metadata_path = user_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully processed {len(processed_paths)} images for user {user_id}")
            return processed_paths
            
        except Exception as e:
            logger.error(f"Error processing training images: {str(e)}")
            raise
    
    def _process_single_image(self, image_path: Path, user_id: str, index: int) -> Optional[str]:
        """Process a single training image"""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Validate image dimensions
            if image.size[0] < self.min_image_size or image.size[1] < self.min_image_size:
                logger.warning(f"Image too small: {image.size}")
                return None
            
            # Detect and crop face
            face_image = self._detect_and_crop_face(image)
            if face_image is None:
                logger.warning(f"No face detected in image {index}")
                # Use center crop as fallback
                face_image = self._center_crop(image)
            
            # Resize to target size
            face_image = face_image.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
            
            # Apply image enhancements
            face_image = self._enhance_image(face_image)
            
            # Generate output filename
            file_hash = hashlib.md5(str(image_path).encode()).hexdigest()[:8]
            output_filename = f"training_{index:03d}_{file_hash}.jpg"
            output_path = Path("uploads") / user_id / output_filename
            
            # Save processed image
            face_image.save(output_path, "JPEG", quality=95)
            
            logger.info(f"Processed image saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error processing single image: {str(e)}")
            return None
    
    def _is_valid_image_streamlit(self, file: Any) -> bool:
        """Validate if uploaded file is a valid image (Streamlit version)"""
        try:
            # Check file extension
            filename = getattr(file, 'name', '')
            if filename:
                ext = Path(filename).suffix.lower()
                if ext not in self.supported_formats:
                    return False
            
            # Check content type
            content_type = getattr(file, 'type', '')
            if content_type and not content_type.startswith("image/"):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _detect_and_crop_face(self, image: Image.Image) -> Optional[Image.Image]:
        """Detect face and crop image around it"""
        try:
            if not FACE_RECOGNITION_AVAILABLE:
                logger.warning("Face recognition not available, skipping face detection")
                return None
            
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(img_array)
            
            if len(face_locations) == 0:
                return None
            
            # Use the first (largest) face
            top, right, bottom, left = face_locations[0]
            
            # Expand the crop area
            height, width = img_array.shape[:2]
            
            # Calculate expansion
            face_width = right - left
            face_height = bottom - top
            expansion = max(face_width, face_height) * 0.3  # 30% expansion
            
            # Calculate crop coordinates
            crop_left = max(0, left - int(expansion))
            crop_top = max(0, top - int(expansion))
            crop_right = min(width, right + int(expansion))
            crop_bottom = min(height, bottom + int(expansion))
            
            # Crop the image
            cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
            
            return cropped
            
        except Exception as e:
            logger.error(f"Error detecting face: {str(e)}")
            return None
    
    def _center_crop(self, image: Image.Image) -> Image.Image:
        """Center crop image to square"""
        try:
            width, height = image.size
            
            # Calculate crop dimensions
            crop_size = min(width, height)
            
            # Calculate crop coordinates
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            
            return image.crop((left, top, right, bottom))
            
        except Exception as e:
            logger.error(f"Error center cropping: {str(e)}")
            return image
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancements"""
        try:
            # Slightly enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # Slightly enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Slightly reduce noise
            image = image.filter(ImageFilter.SMOOTH_MORE)
            
            return image
            
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return image
    
    def _extract_face_encoding(self, image_path: str) -> Optional[np.ndarray]:
        """Extract face encoding for consistency checking"""
        try:
            if not FACE_RECOGNITION_AVAILABLE:
                return None
            
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                return encodings[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting face encoding: {str(e)}")
            return None
    
    def _check_face_consistency(self, face_encodings: List[np.ndarray]) -> float:
        """Check consistency between face encodings"""
        try:
            if not FACE_RECOGNITION_AVAILABLE or len(face_encodings) < 2:
                return 1.0
            
            # Calculate pairwise similarities
            similarities = []
            
            for i in range(len(face_encodings)):
                for j in range(i + 1, len(face_encodings)):
                    # Calculate face distance (lower is more similar)
                    distance = face_recognition.face_distance([face_encodings[i]], face_encodings[j])[0]
                    # Convert to similarity (0-1 scale)
                    similarity = 1 - distance
                    similarities.append(similarity)
            
            # Return average similarity
            return np.mean(similarities)
            
        except Exception as e:
            logger.error(f"Error checking face consistency: {str(e)}")
            return 0.5
    
    def clear_user_data(self, user_id: str):
        """Clear all data for a user"""
        try:
            # Clear uploads
            uploads_dir = Path("uploads") / user_id
            if uploads_dir.exists():
                shutil.rmtree(uploads_dir)
            
            # Clear outputs
            outputs_dir = Path("outputs") / user_id
            if outputs_dir.exists():
                shutil.rmtree(outputs_dir)
            
            # Clear models
            models_dir = Path("models") / user_id
            if models_dir.exists():
                shutil.rmtree(models_dir)
            
            logger.info(f"Cleared all data for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error clearing user data: {str(e)}")
            raise
    
    def get_training_stats(self, user_id: str) -> Dict:
        """Get training image statistics"""
        try:
            user_dir = Path("uploads") / user_id
            
            if not user_dir.exists():
                return {"total_images": 0, "metadata": None}
            
            # Count images
            image_files = []
            for ext in self.supported_formats:
                image_files.extend(user_dir.glob(f"*{ext}"))
            
            total_images = len(image_files)
            
            # Load metadata if exists
            metadata_path = user_dir / "metadata.json"
            metadata = None
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            
            return {
                "total_images": total_images,
                "metadata": metadata,
                "image_files": [str(f) for f in image_files]
            }
            
        except Exception as e:
            logger.error(f"Error getting training stats: {str(e)}")
            return {"total_images": 0, "metadata": None}
    
    def validate_training_set(self, user_id: str) -> Dict:
        """Validate training set quality"""
        try:
            stats = self.get_training_stats(user_id)
            
            validation_result = {
                "is_valid": True,
                "warnings": [],
                "errors": [],
                "recommendations": []
            }
            
            # Check minimum number of images
            if stats["total_images"] < 5:
                validation_result["errors"].append(
                    f"Insufficient training images: {stats['total_images']}/5 minimum"
                )
                validation_result["is_valid"] = False
            
            # Check face consistency
            if stats["metadata"]:
                face_consistency = stats["metadata"].get("face_consistency", 0)
                if face_consistency < 0.6:
                    validation_result["warnings"].append(
                        f"Low face consistency: {face_consistency:.2f}. Consider using more similar images."
                    )
                elif face_consistency < 0.8:
                    validation_result["recommendations"].append(
                        "Good face consistency. Training should work well."
                    )
            
            # Check optimal number of images
            if stats["total_images"] > 20:
                validation_result["warnings"].append(
                    "Many training images. Consider using 10-15 of the best quality images."
                )
            elif stats["total_images"] < 10:
                validation_result["recommendations"].append(
                    "Consider adding more training images (10-15 recommended) for better results."
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating training set: {str(e)}")
            return {
                "is_valid": False,
                "warnings": [],
                "errors": [f"Validation error: {str(e)}"],
                "recommendations": []
            } 