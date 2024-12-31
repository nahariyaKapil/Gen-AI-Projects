import logging
from typing import Dict, List, Optional, Any, Callable
import asyncio
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class PipelineManager:
    """Manages complete vision-language processing pipelines"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Pipeline definitions
        self.pipeline_definitions = {
            "caption_to_image": {
                "steps": ["caption", "enhance", "generate"],
                "description": "Caption image → enhance caption → generate new image"
            },
            "image_to_variations": {
                "steps": ["caption", "variations", "generate"],
                "description": "Caption image → generate variations → create images"
            },
            "video_summary": {
                "steps": ["extract", "caption", "summarize"],
                "description": "Process video → extract frames → generate summary"
            },
            "style_transfer": {
                "steps": ["caption", "style", "generate"],
                "description": "Caption image → apply style → generate styled image"
            }
        }
        
        logger.info("Pipeline Manager initialized")
    
    async def run_pipeline(
        self,
        file_path: str,
        job_id: str,
        pipeline_type: str,
        config_params: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ):
        """Run complete pipeline"""
        try:
            logger.info(f"Starting pipeline {pipeline_type} for job {job_id}")
            
            if pipeline_type not in self.pipeline_definitions:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")
            
            pipeline_def = self.pipeline_definitions[pipeline_type]
            steps = pipeline_def["steps"]
            
            if progress_callback:
                progress_callback({
                    "status": "starting",
                    "progress": 0,
                    "pipeline_type": pipeline_type,
                    "total_steps": len(steps),
                    "current_step": 0
                })
            
            # Execute pipeline based on type
            if pipeline_type == "caption_to_image":
                result = await self._run_caption_to_image_pipeline(file_path, job_id, config_params, progress_callback)
            elif pipeline_type == "image_to_variations":
                result = await self._run_image_to_variations_pipeline(file_path, job_id, config_params, progress_callback)
            elif pipeline_type == "video_summary":
                result = await self._run_video_summary_pipeline(file_path, job_id, config_params, progress_callback)
            elif pipeline_type == "style_transfer":
                result = await self._run_style_transfer_pipeline(file_path, job_id, config_params, progress_callback)
            else:
                raise ValueError(f"Pipeline execution not implemented for: {pipeline_type}")
            
            # Save final results
            result_path = await self._save_pipeline_results(job_id, pipeline_type, result)
            
            if progress_callback:
                progress_callback({
                    "status": "completed",
                    "progress": 100,
                    "pipeline_type": pipeline_type,
                    "result_path": result_path,
                    "result": result
                })
            
            logger.info(f"Pipeline {pipeline_type} completed for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}")
            if progress_callback:
                progress_callback({
                    "status": "failed",
                    "progress": 0,
                    "error": str(e)
                })
            raise
    
    async def _run_caption_to_image_pipeline(
        self,
        file_path: str,
        job_id: str,
        config_params: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run caption → enhance → generate pipeline"""
        try:
            result = {"steps": []}
            
            # Step 1: Caption image
            if progress_callback:
                progress_callback({
                    "status": "captioning",
                    "progress": 20,
                    "current_step": 1,
                    "step_name": "caption"
                })
            
            # Simulate captioning (would use VisionCaptioner)
            await asyncio.sleep(1)
            caption = "A beautiful landscape with mountains and trees"
            
            result["steps"].append({
                "step": "caption",
                "result": caption,
                "confidence": 0.85
            })
            
            # Step 2: Enhance caption
            if progress_callback:
                progress_callback({
                    "status": "enhancing",
                    "progress": 50,
                    "current_step": 2,
                    "step_name": "enhance"
                })
            
            # Simulate enhancement (would use LanguageProcessor)
            await asyncio.sleep(1)
            enhanced_caption = f"Enhanced: {caption} with vivid colors and dramatic lighting"
            
            result["steps"].append({
                "step": "enhance",
                "original": caption,
                "enhanced": enhanced_caption
            })
            
            # Step 3: Generate new image
            if progress_callback:
                progress_callback({
                    "status": "generating",
                    "progress": 80,
                    "current_step": 3,
                    "step_name": "generate"
                })
            
            # Simulate generation (would use ImageEditor)
            await asyncio.sleep(2)
            generated_image_path = f"outputs/generated_{job_id}.png"
            
            result["steps"].append({
                "step": "generate",
                "prompt": enhanced_caption,
                "generated_image": generated_image_path
            })
            
            result["final_output"] = {
                "original_image": file_path,
                "caption": caption,
                "enhanced_caption": enhanced_caption,
                "generated_image": generated_image_path
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in caption-to-image pipeline: {str(e)}")
            raise
    
    async def _run_image_to_variations_pipeline(
        self,
        file_path: str,
        job_id: str,
        config_params: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run caption → variations → generate pipeline"""
        try:
            result = {"steps": []}
            
            # Step 1: Caption image
            if progress_callback:
                progress_callback({
                    "status": "captioning",
                    "progress": 25,
                    "current_step": 1,
                    "step_name": "caption"
                })
            
            await asyncio.sleep(1)
            caption = "A scenic mountain landscape"
            
            result["steps"].append({
                "step": "caption",
                "result": caption
            })
            
            # Step 2: Generate variations
            if progress_callback:
                progress_callback({
                    "status": "generating_variations",
                    "progress": 50,
                    "current_step": 2,
                    "step_name": "variations"
                })
            
            await asyncio.sleep(1)
            variations = [
                "A majestic mountain range with snow-capped peaks",
                "Rolling hills with lush green vegetation",
                "A peaceful valley surrounded by towering mountains"
            ]
            
            result["steps"].append({
                "step": "variations",
                "original": caption,
                "variations": variations
            })
            
            # Step 3: Generate images for variations
            if progress_callback:
                progress_callback({
                    "status": "generating_images",
                    "progress": 75,
                    "current_step": 3,
                    "step_name": "generate"
                })
            
            await asyncio.sleep(2)
            generated_images = []
            
            for i, variation in enumerate(variations):
                image_path = f"outputs/variation_{i}_{job_id}.png"
                generated_images.append({
                    "variation": variation,
                    "image_path": image_path
                })
            
            result["steps"].append({
                "step": "generate",
                "generated_images": generated_images
            })
            
            result["final_output"] = {
                "original_image": file_path,
                "caption": caption,
                "variations": variations,
                "generated_images": generated_images
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in image-to-variations pipeline: {str(e)}")
            raise
    
    async def _run_video_summary_pipeline(
        self,
        file_path: str,
        job_id: str,
        config_params: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run video → extract → caption → summarize pipeline"""
        try:
            result = {"steps": []}
            
            # Step 1: Extract frames
            if progress_callback:
                progress_callback({
                    "status": "extracting_frames",
                    "progress": 30,
                    "current_step": 1,
                    "step_name": "extract"
                })
            
            await asyncio.sleep(2)
            extracted_frames = [
                {"frame_index": 0, "timestamp": 0.0, "frame_path": f"temp/{job_id}/frame_0.jpg"},
                {"frame_index": 1, "timestamp": 5.0, "frame_path": f"temp/{job_id}/frame_1.jpg"},
                {"frame_index": 2, "timestamp": 10.0, "frame_path": f"temp/{job_id}/frame_2.jpg"}
            ]
            
            result["steps"].append({
                "step": "extract",
                "extracted_frames": len(extracted_frames),
                "frames": extracted_frames
            })
            
            # Step 2: Caption frames
            if progress_callback:
                progress_callback({
                    "status": "captioning_frames",
                    "progress": 60,
                    "current_step": 2,
                    "step_name": "caption"
                })
            
            await asyncio.sleep(2)
            frame_captions = [
                {"frame_index": 0, "caption": "Opening scene with title card"},
                {"frame_index": 1, "caption": "Person walking in a park"},
                {"frame_index": 2, "caption": "Close-up of flowers in bloom"}
            ]
            
            result["steps"].append({
                "step": "caption",
                "frame_captions": frame_captions
            })
            
            # Step 3: Generate summary
            if progress_callback:
                progress_callback({
                    "status": "generating_summary",
                    "progress": 90,
                    "current_step": 3,
                    "step_name": "summarize"
                })
            
            await asyncio.sleep(1)
            summary = {
                "title": "Nature Walk Video",
                "duration": "10 seconds",
                "key_scenes": [
                    {"timestamp": 0.0, "description": "Opening scene"},
                    {"timestamp": 5.0, "description": "Main activity"},
                    {"timestamp": 10.0, "description": "Ending scene"}
                ],
                "themes": ["nature", "walking", "flowers"],
                "overall_summary": "A short video showing a peaceful walk in nature with focus on natural elements."
            }
            
            result["steps"].append({
                "step": "summarize",
                "summary": summary
            })
            
            result["final_output"] = {
                "original_video": file_path,
                "extracted_frames": len(extracted_frames),
                "frame_captions": frame_captions,
                "summary": summary
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in video summary pipeline: {str(e)}")
            raise
    
    async def _run_style_transfer_pipeline(
        self,
        file_path: str,
        job_id: str,
        config_params: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run caption → style → generate pipeline"""
        try:
            result = {"steps": []}
            target_style = config_params.get("target_style", "artistic")
            
            # Step 1: Caption image
            if progress_callback:
                progress_callback({
                    "status": "captioning",
                    "progress": 30,
                    "current_step": 1,
                    "step_name": "caption"
                })
            
            await asyncio.sleep(1)
            caption = "A cityscape with buildings and streets"
            
            result["steps"].append({
                "step": "caption",
                "result": caption
            })
            
            # Step 2: Apply style transformation
            if progress_callback:
                progress_callback({
                    "status": "applying_style",
                    "progress": 60,
                    "current_step": 2,
                    "step_name": "style"
                })
            
            await asyncio.sleep(1)
            styled_prompt = f"{caption} in {target_style} style"
            
            result["steps"].append({
                "step": "style",
                "original_caption": caption,
                "styled_prompt": styled_prompt,
                "target_style": target_style
            })
            
            # Step 3: Generate styled image
            if progress_callback:
                progress_callback({
                    "status": "generating_styled_image",
                    "progress": 90,
                    "current_step": 3,
                    "step_name": "generate"
                })
            
            await asyncio.sleep(2)
            styled_image_path = f"outputs/styled_{target_style}_{job_id}.png"
            
            result["steps"].append({
                "step": "generate",
                "styled_prompt": styled_prompt,
                "styled_image": styled_image_path
            })
            
            result["final_output"] = {
                "original_image": file_path,
                "caption": caption,
                "target_style": target_style,
                "styled_prompt": styled_prompt,
                "styled_image": styled_image_path
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in style transfer pipeline: {str(e)}")
            raise
    
    async def _save_pipeline_results(
        self,
        job_id: str,
        pipeline_type: str,
        result: Dict[str, Any]
    ) -> str:
        """Save complete pipeline results"""
        try:
            # Create results directory
            results_dir = Path("outputs") / "pipelines"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            result["metadata"] = {
                "job_id": job_id,
                "pipeline_type": pipeline_type,
                "completed_at": time.time(),
                "total_steps": len(result["steps"])
            }
            
            # Save results
            results_file = results_dir / f"pipeline_{job_id}.json"
            with open(results_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            
            logger.info(f"Pipeline results saved: {results_file}")
            return str(results_file)
            
        except Exception as e:
            logger.error(f"Error saving pipeline results: {str(e)}")
            raise
    
    def get_available_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available pipeline types"""
        return self.pipeline_definitions.copy()
    
    def validate_pipeline_config(
        self,
        pipeline_type: str,
        config_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate pipeline configuration"""
        try:
            validation_result = {
                "is_valid": True,
                "warnings": [],
                "errors": [],
                "recommendations": []
            }
            
            # Check if pipeline type exists
            if pipeline_type not in self.pipeline_definitions:
                validation_result["errors"].append(f"Unknown pipeline type: {pipeline_type}")
                validation_result["is_valid"] = False
                return validation_result
            
            # Validate config for specific pipeline types
            if pipeline_type == "style_transfer":
                if "target_style" not in config_params:
                    validation_result["warnings"].append("No target style specified, will use 'artistic'")
                    config_params["target_style"] = "artistic"
                
                valid_styles = ["artistic", "photorealistic", "cartoon", "sketch", "watercolor"]
                if config_params.get("target_style") not in valid_styles:
                    validation_result["recommendations"].append(
                        f"Consider using one of the supported styles: {', '.join(valid_styles)}"
                    )
            
            elif pipeline_type == "video_summary":
                frame_interval = config_params.get("frame_interval", 30)
                if frame_interval < 10:
                    validation_result["warnings"].append("Very low frame interval may result in slow processing")
                elif frame_interval > 300:
                    validation_result["warnings"].append("High frame interval may miss important scenes")
            
            elif pipeline_type == "image_to_variations":
                num_variations = config_params.get("num_variations", 3)
                if num_variations > 5:
                    validation_result["warnings"].append("Many variations may take longer to process")
                elif num_variations < 2:
                    validation_result["recommendations"].append("Consider generating at least 2 variations")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating pipeline config: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "recommendations": []
            }
    
    def get_pipeline_progress_info(self, pipeline_type: str) -> Dict[str, Any]:
        """Get information about pipeline steps and expected progress"""
        if pipeline_type not in self.pipeline_definitions:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        pipeline_def = self.pipeline_definitions[pipeline_type]
        
        # Define expected durations for each step (in seconds)
        step_durations = {
            "caption": 2,
            "enhance": 1,
            "generate": 5,
            "variations": 2,
            "extract": 10,
            "summarize": 3,
            "style": 1
        }
        
        steps_info = []
        total_duration = 0
        
        for step in pipeline_def["steps"]:
            duration = step_durations.get(step, 2)
            total_duration += duration
            
            steps_info.append({
                "step": step,
                "estimated_duration": duration,
                "description": self._get_step_description(step)
            })
        
        return {
            "pipeline_type": pipeline_type,
            "description": pipeline_def["description"],
            "steps": steps_info,
            "total_steps": len(pipeline_def["steps"]),
            "estimated_total_duration": total_duration
        }
    
    def _get_step_description(self, step: str) -> str:
        """Get description for a pipeline step"""
        descriptions = {
            "caption": "Generate image caption using vision models",
            "enhance": "Enhance caption with language models",
            "generate": "Generate images using diffusion models",
            "variations": "Create caption variations",
            "extract": "Extract frames from video",
            "summarize": "Generate summary from captions",
            "style": "Apply style transformation to prompts"
        }
        
        return descriptions.get(step, f"Execute {step} step")
    
    def cleanup_pipeline_files(self, job_id: str):
        """Clean up temporary files for a pipeline job"""
        try:
            # Clean up temp files
            temp_dir = Path("temp") / job_id
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp files for pipeline job {job_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up pipeline files: {str(e)}")
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get statistics about pipeline usage"""
        try:
            results_dir = Path("outputs") / "pipelines"
            
            if not results_dir.exists():
                return {"total_pipelines": 0, "pipeline_types": {}}
            
            # Count pipeline files
            pipeline_files = list(results_dir.glob("pipeline_*.json"))
            total_pipelines = len(pipeline_files)
            
            # Count by type
            pipeline_types = {}
            
            for file in pipeline_files:
                try:
                    with open(file, "r") as f:
                        data = json.load(f)
                        pipeline_type = data.get("metadata", {}).get("pipeline_type", "unknown")
                        pipeline_types[pipeline_type] = pipeline_types.get(pipeline_type, 0) + 1
                except:
                    continue
            
            return {
                "total_pipelines": total_pipelines,
                "pipeline_types": pipeline_types,
                "available_types": list(self.pipeline_definitions.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline statistics: {str(e)}")
            return {"total_pipelines": 0, "pipeline_types": {}} 