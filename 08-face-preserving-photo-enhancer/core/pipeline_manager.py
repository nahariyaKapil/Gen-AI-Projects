import asyncio
import time
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

from .face_detector import FaceDetector
from .identity_preservor import IdentityPreservor
from .style_enhancer import StyleEnhancer

logger = logging.getLogger(__name__)

class PipelineManager:
    """
    Pipeline manager that orchestrates the entire face enhancement process
    """
    
    def __init__(self, face_detector: FaceDetector, identity_preservor: IdentityPreservor, 
                 style_enhancer: StyleEnhancer):
        """
        Initialize the pipeline manager
        
        Args:
            face_detector: Face detection component
            identity_preservor: Identity preservation component
            style_enhancer: Style enhancement component
        """
        self.face_detector = face_detector
        self.identity_preservor = identity_preservor
        self.style_enhancer = style_enhancer
        self.processing_steps = [
            "face_analysis",
            "identity_extraction",
            "style_enhancement",
            "identity_validation",
            "post_processing"
        ]
        
        logger.info("Pipeline manager initialized")
    
    def is_ready(self) -> bool:
        """Check if all components are ready"""
        return (
            self.face_detector.is_loaded() and 
            self.identity_preservor.is_loaded() and 
            self.style_enhancer.is_loaded()
        )
    
    async def process_image(self, image_path: str, style: str = "professional",
                           enhancement_level: float = 0.8, preserve_identity: bool = True,
                           output_format: str = "jpeg", resolution: str = "1024x1024",
                           additional_prompts: List[str] = None, task_id: str = None) -> Dict:
        """
        Process image through the complete enhancement pipeline
        
        Args:
            image_path: Path to input image
            style: Enhancement style
            enhancement_level: Enhancement strength (0-1)
            preserve_identity: Whether to preserve identity
            output_format: Output format
            resolution: Output resolution
            additional_prompts: Additional prompts
            task_id: Task ID for progress tracking
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            if not self.is_ready():
                raise RuntimeError("Pipeline components not ready")
            
            # Initialize result dictionary
            result = {
                "success": False,
                "processing_time": 0,
                "steps_completed": [],
                "face_analysis": None,
                "identity_data": None,
                "enhancement_details": None,
                "output_path": None,
                "validation_results": None
            }
            
            # Step 1: Face Analysis
            logger.info(f"Starting face analysis for {image_path}")
            if task_id:
                self._update_progress(task_id, 0, "Analyzing face...")
            
            face_analysis = await self._analyze_face(image_path)
            result["face_analysis"] = face_analysis
            result["steps_completed"].append("face_analysis")
            
            if not face_analysis.get("face_found", False):
                raise ValueError("No suitable face found in image")
            
            # Step 2: Identity Extraction
            logger.info("Extracting identity features")
            if task_id:
                self._update_progress(task_id, 20, "Extracting identity features...")
            
            identity_data = await self._extract_identity(image_path)
            result["identity_data"] = identity_data
            result["steps_completed"].append("identity_extraction")
            
            # Step 3: Style Enhancement
            logger.info(f"Applying style enhancement: {style}")
            if task_id:
                self._update_progress(task_id, 40, f"Applying {style} style...")
            
            enhancement_result = await self._apply_enhancement(
                image_path=image_path,
                style=style,
                enhancement_level=enhancement_level,
                resolution=resolution,
                additional_prompts=additional_prompts,
                face_analysis=face_analysis
            )
            result["enhancement_details"] = enhancement_result
            result["steps_completed"].append("style_enhancement")
            
            # Step 4: Identity Validation (if enabled)
            if preserve_identity:
                logger.info("Validating identity preservation")
                if task_id:
                    self._update_progress(task_id, 70, "Validating identity preservation...")
                
                validation_result = await self._validate_identity(
                    original_path=image_path,
                    enhanced_image=enhancement_result["enhanced_image"],
                    task_id=task_id
                )
                result["validation_results"] = validation_result
                result["steps_completed"].append("identity_validation")
                
                # Apply adaptive adjustments if needed
                if not validation_result.get("identity_preserved", True):
                    logger.info("Applying adaptive identity correction")
                    if task_id:
                        self._update_progress(task_id, 80, "Applying identity correction...")
                    
                    enhancement_result = await self._apply_adaptive_correction(
                        original_path=image_path,
                        enhanced_image=enhancement_result["enhanced_image"],
                        validation_result=validation_result,
                        enhancement_params={
                            "style": style,
                            "enhancement_level": enhancement_level,
                            "resolution": resolution,
                            "additional_prompts": additional_prompts
                        }
                    )
                    
                    # Re-validate
                    validation_result = await self._validate_identity(
                        original_path=image_path,
                        enhanced_image=enhancement_result["enhanced_image"],
                        task_id=task_id
                    )
                    result["validation_results"] = validation_result
            
            # Step 5: Post-processing and Save
            logger.info("Finalizing output")
            if task_id:
                self._update_progress(task_id, 90, "Finalizing output...")
            
            output_path = await self._finalize_output(
                enhanced_image=enhancement_result["enhanced_image"],
                output_format=output_format,
                task_id=task_id
            )
            result["output_path"] = output_path
            result["steps_completed"].append("post_processing")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            result["success"] = True
            
            logger.info(f"Pipeline completed successfully in {processing_time:.2f} seconds")
            
            if task_id:
                self._update_progress(task_id, 100, "Enhancement completed!")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            result["error"] = str(e)
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    async def _analyze_face(self, image_path: str) -> Dict:
        """Analyze face in the image"""
        try:
            # Run face analysis
            face_quality = self.face_detector.analyze_face_quality(image_path)
            
            # Extract detailed landmarks
            landmarks = self.face_detector.extract_face_landmarks(image_path)
            
            return {
                "face_found": face_quality["quality_score"] > 0.0,
                "quality_analysis": face_quality,
                "landmarks": landmarks,
                "enhancement_suitability": self._assess_enhancement_suitability(face_quality)
            }
            
        except Exception as e:
            logger.error(f"Face analysis failed: {str(e)}")
            return {"face_found": False, "error": str(e)}
    
    async def _extract_identity(self, image_path: str) -> Dict:
        """Extract identity features"""
        try:
            # Extract face encoding
            encoding = self.identity_preservor.extract_face_encoding(image_path)
            
            if encoding is None:
                return {"identity_extracted": False, "error": "Could not extract face encoding"}
            
            return {
                "identity_extracted": True,
                "encoding": encoding,
                "encoding_shape": encoding.shape
            }
            
        except Exception as e:
            logger.error(f"Identity extraction failed: {str(e)}")
            return {"identity_extracted": False, "error": str(e)}
    
    async def _apply_enhancement(self, image_path: str, style: str, enhancement_level: float,
                                resolution: str, additional_prompts: List[str],
                                face_analysis: Dict) -> Dict:
        """Apply style enhancement"""
        try:
            # Adjust enhancement level based on face quality
            adjusted_level = self._adjust_enhancement_level(
                enhancement_level, face_analysis.get("quality_analysis", {})
            )
            
            # Apply enhancement
            enhancement_result = self.style_enhancer.enhance_image(
                image_path=image_path,
                style=style,
                enhancement_level=adjusted_level,
                resolution=resolution,
                additional_prompts=additional_prompts
            )
            
            return {
                "success": True,
                "enhanced_image": enhancement_result["enhanced_image"],
                "original_enhancement_level": enhancement_level,
                "adjusted_enhancement_level": adjusted_level,
                "style_config": enhancement_result
            }
            
        except Exception as e:
            logger.error(f"Enhancement failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _validate_identity(self, original_path: str, enhanced_image: Image.Image,
                                task_id: str = None) -> Dict:
        """Validate identity preservation"""
        try:
            # Save enhanced image temporarily for validation
            temp_path = f"temp_enhanced_{task_id}.jpg" if task_id else "temp_enhanced.jpg"
            enhanced_image.save(temp_path)
            
            # Validate identity
            validation = self.identity_preservor.validate_identity_preservation(
                original_path=original_path,
                enhanced_path=temp_path
            )
            
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
            
            return validation
            
        except Exception as e:
            logger.error(f"Identity validation failed: {str(e)}")
            return {"identity_preserved": False, "error": str(e)}
    
    async def _apply_adaptive_correction(self, original_path: str, enhanced_image: Image.Image,
                                       validation_result: Dict, enhancement_params: Dict) -> Dict:
        """Apply adaptive correction based on identity validation"""
        try:
            # Get adaptive weights
            weights = self.identity_preservor.adaptive_enhancement_weights(
                original_path=original_path,
                enhanced_path="temp_enhanced.jpg",  # This would be the enhanced image path
                target_similarity=0.7
            )
            
            # Reduce enhancement strength
            corrected_level = enhancement_params["enhancement_level"] * weights["enhancement_weight"]
            
            # Re-apply enhancement with corrected parameters
            correction_result = self.style_enhancer.enhance_image(
                image_path=original_path,
                style=enhancement_params["style"],
                enhancement_level=corrected_level,
                resolution=enhancement_params["resolution"],
                additional_prompts=enhancement_params["additional_prompts"]
            )
            
            return {
                "success": True,
                "enhanced_image": correction_result["enhanced_image"],
                "correction_applied": True,
                "original_level": enhancement_params["enhancement_level"],
                "corrected_level": corrected_level,
                "weights_used": weights
            }
            
        except Exception as e:
            logger.error(f"Adaptive correction failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _finalize_output(self, enhanced_image: Image.Image, output_format: str,
                              task_id: str = None) -> str:
        """Finalize and save output image"""
        try:
            # Generate output filename
            if task_id:
                output_path = f"uploads/{task_id}/enhanced.{output_format}"
            else:
                output_path = f"enhanced_output.{output_format}"
            
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image
            if output_format.lower() == 'jpeg':
                enhanced_image.save(output_path, 'JPEG', quality=95, optimize=True)
            elif output_format.lower() == 'png':
                enhanced_image.save(output_path, 'PNG', optimize=True)
            else:
                enhanced_image.save(output_path, output_format.upper())
            
            return output_path
            
        except Exception as e:
            logger.error(f"Output finalization failed: {str(e)}")
            raise
    
    def _assess_enhancement_suitability(self, face_quality: Dict) -> Dict:
        """Assess how suitable the image is for enhancement"""
        quality_score = face_quality.get("quality_score", 0.0)
        
        if quality_score >= 0.8:
            suitability = "Excellent"
            recommendation = "Perfect for all enhancement styles"
        elif quality_score >= 0.6:
            suitability = "Good"
            recommendation = "Suitable for most enhancement styles"
        elif quality_score >= 0.4:
            suitability = "Fair"
            recommendation = "Use lower enhancement levels"
        else:
            suitability = "Poor"
            recommendation = "Consider using a different image"
        
        return {
            "suitability": suitability,
            "recommendation": recommendation,
            "suggested_enhancement_level": min(1.0, max(0.2, quality_score))
        }
    
    def _adjust_enhancement_level(self, enhancement_level: float, face_quality: Dict) -> float:
        """Adjust enhancement level based on face quality"""
        quality_score = face_quality.get("quality_score", 0.5)
        
        # Reduce enhancement level for lower quality images
        if quality_score < 0.6:
            adjustment_factor = max(0.5, quality_score)
            adjusted_level = enhancement_level * adjustment_factor
            logger.info(f"Adjusted enhancement level from {enhancement_level} to {adjusted_level} due to face quality")
            return adjusted_level
        
        return enhancement_level
    
    def _update_progress(self, task_id: str, progress: int, message: str):
        """Update progress for a task"""
        # This would typically update a database or cache
        # For now, we'll just log the progress
        logger.info(f"Task {task_id}: {progress}% - {message}")
    
    async def process_batch(self, image_paths: List[str], style: str = "professional",
                           enhancement_level: float = 0.8, preserve_identity: bool = True,
                           output_format: str = "jpeg", resolution: str = "1024x1024") -> List[Dict]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image paths
            style: Enhancement style
            enhancement_level: Enhancement strength
            preserve_identity: Whether to preserve identity
            output_format: Output format
            resolution: Output resolution
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing batch image {i+1}/{len(image_paths)}: {image_path}")
                
                result = await self.process_image(
                    image_path=image_path,
                    style=style,
                    enhancement_level=enhancement_level,
                    preserve_identity=preserve_identity,
                    output_format=output_format,
                    resolution=resolution,
                    task_id=f"batch_{i}"
                )
                
                results.append({
                    "index": i,
                    "image_path": image_path,
                    "success": True,
                    "result": result
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "image_path": image_path,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        return {
            "ready": self.is_ready(),
            "components": {
                "face_detector": self.face_detector.is_loaded(),
                "identity_preservor": self.identity_preservor.is_loaded(),
                "style_enhancer": self.style_enhancer.is_loaded()
            },
            "supported_styles": self.style_enhancer.get_available_styles(),
            "processing_steps": self.processing_steps
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.style_enhancer.cleanup_memory()
            logger.info("Pipeline cleanup completed")
        except Exception as e:
            logger.error(f"Pipeline cleanup failed: {str(e)}") 