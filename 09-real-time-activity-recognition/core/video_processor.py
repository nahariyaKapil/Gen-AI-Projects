import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Video processing utility for real-time activity recognition
    """
    
    def __init__(self, frame_size: Tuple[int, int] = (224, 224)):
        """
        Initialize video processor
        
        Args:
            frame_size: Target frame size for model input
        """
        self.frame_size = frame_size
        self.frame_buffer = []
        self.buffer_size = 16  # Number of frames for temporal analysis
        self.fps_tracker = []
        
        logger.info(f"Video processor initialized with frame size: {frame_size}")
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single frame for model input
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Preprocessed frame tensor
        """
        try:
            # Resize frame
            resized = cv2.resize(frame, self.frame_size)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb_frame.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {str(e)}")
            raise
    
    def preprocess_video_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a sequence of frames for temporal models
        
        Args:
            frames: List of frame arrays
            
        Returns:
            Preprocessed video clip tensor [batch, channels, frames, height, width]
        """
        try:
            if len(frames) == 0:
                raise ValueError("No frames provided")
            
            # Process each frame
            processed_frames = []
            for frame in frames:
                # Resize frame
                resized = cv2.resize(frame, self.frame_size)
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1]
                normalized = rgb_frame.astype(np.float32) / 255.0
                
                processed_frames.append(normalized)
            
            # Stack frames and convert to tensor
            video_array = np.stack(processed_frames, axis=0)  # [frames, height, width, channels]
            
            # Rearrange dimensions to [channels, frames, height, width] and add batch dimension
            video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).unsqueeze(0)
            
            return video_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing video clip: {str(e)}")
            raise
    
    def update_frame_buffer(self, frame: np.ndarray) -> bool:
        """
        Update frame buffer with new frame
        
        Args:
            frame: New frame to add
            
        Returns:
            True if buffer is full and ready for processing
        """
        try:
            self.frame_buffer.append(frame.copy())
            
            # Maintain buffer size
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
            
            return len(self.frame_buffer) == self.buffer_size
            
        except Exception as e:
            logger.error(f"Error updating frame buffer: {str(e)}")
            return False
    
    def get_video_clip(self) -> Optional[torch.Tensor]:
        """
        Get preprocessed video clip from buffer
        
        Returns:
            Video clip tensor or None if buffer not ready
        """
        try:
            if len(self.frame_buffer) < self.buffer_size:
                return None
            
            return self.preprocess_video_clip(self.frame_buffer)
            
        except Exception as e:
            logger.error(f"Error getting video clip: {str(e)}")
            return None
    
    def extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frame arrays
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    def create_video_stream(self, source: Union[str, int]) -> cv2.VideoCapture:
        """
        Create video stream from source
        
        Args:
            source: Video file path or camera index
            
        Returns:
            VideoCapture object
        """
        try:
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video source: {source}")
            
            # Set properties for optimal performance
            if isinstance(source, int):  # Webcam
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
            return cap
            
        except Exception as e:
            logger.error(f"Error creating video stream: {str(e)}")
            raise
    
    def calculate_fps(self, frame_time: float) -> float:
        """
        Calculate current FPS
        
        Args:
            frame_time: Time taken to process current frame
            
        Returns:
            Current FPS
        """
        try:
            self.fps_tracker.append(frame_time)
            
            # Keep only recent measurements
            if len(self.fps_tracker) > 30:
                self.fps_tracker.pop(0)
            
            if len(self.fps_tracker) > 0:
                avg_time = sum(self.fps_tracker) / len(self.fps_tracker)
                return 1.0 / avg_time if avg_time > 0 else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating FPS: {str(e)}")
            return 0.0
    
    def resize_frame_for_display(self, frame: np.ndarray, max_width: int = 800) -> np.ndarray:
        """
        Resize frame for display while maintaining aspect ratio
        
        Args:
            frame: Input frame
            max_width: Maximum width for display
            
        Returns:
            Resized frame
        """
        try:
            height, width = frame.shape[:2]
            
            if width <= max_width:
                return frame
            
            # Calculate new dimensions
            aspect_ratio = height / width
            new_width = max_width
            new_height = int(new_width * aspect_ratio)
            
            return cv2.resize(frame, (new_width, new_height))
            
        except Exception as e:
            logger.error(f"Error resizing frame: {str(e)}")
            return frame
    
    def add_timestamp(self, frame: np.ndarray) -> np.ndarray:
        """
        Add timestamp overlay to frame
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with timestamp
        """
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Add background rectangle
            text_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (10, 10), (20 + text_size[0], 30 + text_size[1]), (0, 0, 0), -1)
            
            # Add timestamp text
            cv2.putText(frame, timestamp, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error adding timestamp: {str(e)}")
            return frame
    
    def save_frame(self, frame: np.ndarray, output_path: str) -> bool:
        """
        Save frame to file
        
        Args:
            frame: Frame to save
            output_path: Output file path
            
        Returns:
            True if successful
        """
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            success = cv2.imwrite(output_path, frame)
            
            if success:
                logger.info(f"Frame saved to {output_path}")
            else:
                logger.error(f"Failed to save frame to {output_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving frame: {str(e)}")
            return False
    
    def create_video_writer(self, output_path: str, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
        """
        Create video writer for saving processed video
        
        Args:
            output_path: Output video path
            fps: Output video FPS
            frame_size: Frame dimensions (width, height)
            
        Returns:
            VideoWriter object
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Define codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Create video writer
            writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            
            if not writer.isOpened():
                raise ValueError(f"Could not create video writer for {output_path}")
            
            logger.info(f"Video writer created for {output_path}")
            return writer
            
        except Exception as e:
            logger.error(f"Error creating video writer: {str(e)}")
            raise
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get video file information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"error": f"Could not open video: {video_path}"}
            
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
            }
            
            cap.release()
            
            logger.info(f"Video info for {video_path}: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return {"error": str(e)}
    
    def reset_buffer(self):
        """Reset the frame buffer"""
        self.frame_buffer.clear()
        logger.info("Frame buffer reset")
    
    def get_buffer_status(self) -> Dict:
        """
        Get current buffer status
        
        Returns:
            Buffer status information
        """
        return {
            "buffer_size": len(self.frame_buffer),
            "max_buffer_size": self.buffer_size,
            "is_full": len(self.frame_buffer) == self.buffer_size,
            "fill_percentage": len(self.frame_buffer) / self.buffer_size * 100
        } 