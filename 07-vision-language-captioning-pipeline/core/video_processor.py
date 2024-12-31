import cv2
import logging
from typing import Dict, List, Optional, Any, Callable
import time
import asyncio
from pathlib import Path
import json
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Video processing for frame extraction and captioning"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_frame_size = config.get("max_frame_size", (720, 480))
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        logger.info("Video Processor initialized")
    
    async def process_video(
        self,
        video_path: str,
        job_id: str,
        frame_interval: int = 30,
        model_type: str = "blip",
        progress_callback: Optional[Callable] = None
    ):
        """Process video for frame-by-frame analysis"""
        try:
            logger.info(f"Starting video processing for job {job_id}")
            
            # Update progress
            if progress_callback:
                progress_callback({"status": "analyzing_video", "progress": 5})
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video stats: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
            
            if progress_callback:
                progress_callback({
                    "status": "extracting_frames",
                    "progress": 10,
                    "total_frames": total_frames,
                    "fps": fps,
                    "duration": duration
                })
            
            # Extract frames at intervals
            frame_data = []
            current_frame = 0
            extracted_count = 0
            
            while current_frame < total_frames:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame if too large
                if frame_rgb.shape[1] > self.max_frame_size[0] or frame_rgb.shape[0] > self.max_frame_size[1]:
                    frame_rgb = self._resize_frame(frame_rgb, self.max_frame_size)
                
                # Save frame
                frame_info = await self._save_frame(frame_rgb, job_id, extracted_count, current_frame, fps)
                frame_data.append(frame_info)
                
                extracted_count += 1
                current_frame += frame_interval
                
                # Update progress
                progress_pct = 10 + (current_frame / total_frames) * 30
                if progress_callback:
                    progress_callback({
                        "status": "extracting_frames",
                        "progress": int(progress_pct),
                        "extracted_frames": extracted_count,
                        "current_frame": current_frame
                    })
                
                # Allow other async tasks to run
                await asyncio.sleep(0.01)
            
            cap.release()
            
            # Caption frames (this would use the vision captioner)
            if progress_callback:
                progress_callback({"status": "captioning_frames", "progress": 40})
            
            captions = await self._caption_frames(frame_data, model_type, progress_callback)
            
            # Generate summary
            if progress_callback:
                progress_callback({"status": "generating_summary", "progress": 80})
            
            summary = await self._generate_video_summary(captions, frame_data)
            
            # Save results
            result_path = await self._save_results(job_id, {
                "video_path": video_path,
                "total_frames": total_frames,
                "extracted_frames": len(frame_data),
                "fps": fps,
                "duration": duration,
                "frame_data": frame_data,
                "captions": captions,
                "summary": summary
            })
            
            if progress_callback:
                progress_callback({
                    "status": "completed",
                    "progress": 100,
                    "result_path": result_path,
                    "extracted_frames": len(frame_data),
                    "summary": summary
                })
            
            logger.info(f"Video processing completed for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            if progress_callback:
                progress_callback({
                    "status": "failed",
                    "progress": 0,
                    "error": str(e)
                })
            raise
    
    def _resize_frame(self, frame: np.ndarray, max_size: tuple) -> np.ndarray:
        """Resize frame while maintaining aspect ratio"""
        try:
            height, width = frame.shape[:2]
            max_width, max_height = max_size
            
            # Calculate scaling factor
            scale = min(max_width / width, max_height / height)
            
            if scale < 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error resizing frame: {str(e)}")
            return frame
    
    async def _save_frame(
        self,
        frame: np.ndarray,
        job_id: str,
        frame_index: int,
        frame_number: int,
        fps: float
    ) -> Dict[str, Any]:
        """Save frame to disk and return frame info"""
        try:
            # Create output directory
            frames_dir = Path("temp") / job_id / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to PIL Image and save
            image = Image.fromarray(frame)
            frame_filename = f"frame_{frame_index:06d}.jpg"
            frame_path = frames_dir / frame_filename
            
            image.save(frame_path, "JPEG", quality=85)
            
            # Calculate timestamp
            timestamp = frame_number / fps if fps > 0 else 0
            
            frame_info = {
                "frame_index": frame_index,
                "frame_number": frame_number,
                "timestamp": timestamp,
                "frame_path": str(frame_path),
                "width": frame.shape[1],
                "height": frame.shape[0]
            }
            
            return frame_info
            
        except Exception as e:
            logger.error(f"Error saving frame: {str(e)}")
            raise
    
    async def _caption_frames(
        self,
        frame_data: List[Dict],
        model_type: str,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Caption all extracted frames"""
        try:
            # This would typically use the VisionCaptioner
            # For now, we'll simulate the process
            captions = []
            
            for i, frame_info in enumerate(frame_data):
                # Simulate captioning process
                await asyncio.sleep(0.1)  # Simulate processing time
                
                # Mock caption (in real implementation, would use VisionCaptioner)
                caption = {
                    "frame_index": frame_info["frame_index"],
                    "timestamp": frame_info["timestamp"],
                    "caption": f"Frame at {frame_info['timestamp']:.2f}s",
                    "confidence": 0.8,
                    "model_type": model_type
                }
                
                captions.append(caption)
                
                # Update progress
                if progress_callback:
                    progress_pct = 40 + (i / len(frame_data)) * 35
                    progress_callback({
                        "status": "captioning_frames",
                        "progress": int(progress_pct),
                        "captioned_frames": i + 1,
                        "total_frames": len(frame_data)
                    })
            
            return captions
            
        except Exception as e:
            logger.error(f"Error captioning frames: {str(e)}")
            raise
    
    async def _generate_video_summary(
        self,
        captions: List[Dict],
        frame_data: List[Dict]
    ) -> Dict[str, Any]:
        """Generate summary of video content"""
        try:
            # Extract caption texts
            caption_texts = [c["caption"] for c in captions]
            
            # Calculate basic statistics
            total_duration = frame_data[-1]["timestamp"] if frame_data else 0
            frame_count = len(frame_data)
            avg_confidence = sum(c["confidence"] for c in captions) / len(captions) if captions else 0
            
            # Generate simple summary (in real implementation, would use LanguageProcessor)
            summary_text = f"Video contains {frame_count} key frames over {total_duration:.2f} seconds."
            
            # Identify key moments (frames with significant changes)
            key_moments = []
            for i, caption in enumerate(captions):
                if i == 0 or i == len(captions) - 1:  # First and last frames
                    key_moments.append({
                        "timestamp": caption["timestamp"],
                        "description": caption["caption"],
                        "importance": "high"
                    })
                elif i % (len(captions) // 5) == 0:  # Every 5th frame
                    key_moments.append({
                        "timestamp": caption["timestamp"],
                        "description": caption["caption"],
                        "importance": "medium"
                    })
            
            summary = {
                "summary_text": summary_text,
                "total_duration": total_duration,
                "frame_count": frame_count,
                "avg_confidence": avg_confidence,
                "key_moments": key_moments,
                "themes": self._extract_themes(caption_texts),
                "generated_at": time.time()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating video summary: {str(e)}")
            raise
    
    def _extract_themes(self, caption_texts: List[str]) -> List[str]:
        """Extract common themes from captions"""
        try:
            # Simple theme extraction (in real implementation, would use NLP)
            common_words = ["person", "people", "car", "building", "nature", "animal", "food", "indoor", "outdoor"]
            themes = []
            
            text_combined = " ".join(caption_texts).lower()
            
            for word in common_words:
                if word in text_combined:
                    themes.append(word)
            
            return themes[:5]  # Return top 5 themes
            
        except Exception as e:
            logger.error(f"Error extracting themes: {str(e)}")
            return []
    
    async def _save_results(self, job_id: str, results: Dict[str, Any]) -> str:
        """Save processing results to file"""
        try:
            # Create results directory
            results_dir = Path("temp") / job_id
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results as JSON
            results_path = results_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_path}")
            return str(results_path)
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get basic information about a video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            info = {
                "file_path": video_path,
                "total_frames": total_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "duration": duration,
                "format": Path(video_path).suffix.lower(),
                "size_mb": Path(video_path).stat().st_size / (1024 * 1024)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            raise
    
    def validate_video(self, video_path: str) -> Dict[str, Any]:
        """Validate video file and return recommendations"""
        try:
            validation_result = {
                "is_valid": True,
                "warnings": [],
                "errors": [],
                "recommendations": []
            }
            
            # Check if file exists
            if not Path(video_path).exists():
                validation_result["errors"].append("Video file not found")
                validation_result["is_valid"] = False
                return validation_result
            
            # Check file format
            file_ext = Path(video_path).suffix.lower()
            if file_ext not in self.supported_formats:
                validation_result["warnings"].append(f"Unsupported format: {file_ext}")
            
            # Get video info
            try:
                info = self.get_video_info(video_path)
                
                # Check duration
                if info["duration"] > 600:  # 10 minutes
                    validation_result["warnings"].append(
                        f"Long video ({info['duration']:.1f}s). Processing may take a while."
                    )
                elif info["duration"] < 1:
                    validation_result["warnings"].append("Very short video")
                
                # Check resolution
                if info["width"] > 1920 or info["height"] > 1080:
                    validation_result["recommendations"].append(
                        "High resolution video. Consider reducing resolution for faster processing."
                    )
                
                # Check file size
                if info["size_mb"] > 100:
                    validation_result["warnings"].append(
                        f"Large file size ({info['size_mb']:.1f} MB). Upload may take time."
                    )
                
                # Check FPS
                if info["fps"] > 60:
                    validation_result["recommendations"].append(
                        "High FPS video. Consider increasing frame interval for efficiency."
                    )
                elif info["fps"] < 10:
                    validation_result["warnings"].append("Low FPS video")
                
            except Exception as e:
                validation_result["errors"].append(f"Could not analyze video: {str(e)}")
                validation_result["is_valid"] = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating video: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "recommendations": []
            }
    
    async def extract_thumbnail(self, video_path: str, timestamp: float = 0) -> str:
        """Extract thumbnail from video at specific timestamp"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Set position to timestamp
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError("Could not extract frame from video")
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to thumbnail size
            thumbnail_size = (320, 240)
            frame_rgb = self._resize_frame(frame_rgb, thumbnail_size)
            
            # Save thumbnail
            thumbnail_dir = Path("temp") / "thumbnails"
            thumbnail_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp_str = int(time.time())
            thumbnail_filename = f"thumbnail_{timestamp_str}.jpg"
            thumbnail_path = thumbnail_dir / thumbnail_filename
            
            image = Image.fromarray(frame_rgb)
            image.save(thumbnail_path, "JPEG", quality=85)
            
            logger.info(f"Thumbnail extracted: {thumbnail_path}")
            return str(thumbnail_path)
            
        except Exception as e:
            logger.error(f"Error extracting thumbnail: {str(e)}")
            raise
    
    def cleanup_job_files(self, job_id: str):
        """Clean up temporary files for a job"""
        try:
            job_dir = Path("temp") / job_id
            if job_dir.exists():
                import shutil
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up files for job {job_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up job files: {str(e)}")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported video formats"""
        return self.supported_formats.copy() 