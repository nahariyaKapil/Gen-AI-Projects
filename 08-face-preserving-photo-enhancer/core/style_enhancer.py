import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import DDIMScheduler, LMSDiscreteScheduler
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class StyleEnhancer:
    """
    Style enhancement using Stable Diffusion and traditional image processing
    """
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Initialize the style enhancer
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion
        """
        self.model_id = model_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipe = None
        self.img2img_pipe = None
        self._loaded = False
        
        # Available enhancement styles
        self.styles = {
            "professional": {
                "prompt": "professional portrait, clean, polished, business attire, good lighting",
                "negative_prompt": "blurry, low quality, distorted, oversaturated",
                "strength": 0.3,
                "guidance_scale": 7.5
            },
            "glamorous": {
                "prompt": "glamorous portrait, enhanced features, subtle makeup, dramatic lighting, elegant",
                "negative_prompt": "blurry, low quality, distorted, over-processed",
                "strength": 0.4,
                "guidance_scale": 8.0
            },
            "casual": {
                "prompt": "natural portrait, casual, good lighting, warm tones, realistic",
                "negative_prompt": "blurry, low quality, artificial, over-processed",
                "strength": 0.25,
                "guidance_scale": 7.0
            },
            "artistic": {
                "prompt": "artistic portrait, creative, stylized, unique aesthetic, high quality",
                "negative_prompt": "blurry, low quality, generic, boring",
                "strength": 0.5,
                "guidance_scale": 9.0
            },
            "vintage": {
                "prompt": "vintage portrait, retro aesthetic, film grain, warm tones, nostalgic",
                "negative_prompt": "blurry, low quality, modern, digital artifacts",
                "strength": 0.4,
                "guidance_scale": 8.0
            },
            "modern": {
                "prompt": "modern portrait, contemporary, sharp details, vibrant colors, high quality",
                "negative_prompt": "blurry, low quality, dated, dull",
                "strength": 0.35,
                "guidance_scale": 7.5
            }
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Stable Diffusion models"""
        try:
            logger.info(f"Initializing Stable Diffusion models on {self.device}")
            
            # Load img2img pipeline
            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Move to device
            self.img2img_pipe = self.img2img_pipe.to(self.device)
            
            # Enable memory efficient attention
            if hasattr(self.img2img_pipe, 'enable_attention_slicing'):
                self.img2img_pipe.enable_attention_slicing()
            
            # Enable VAE slicing for lower memory usage
            if hasattr(self.img2img_pipe, 'enable_vae_slicing'):
                self.img2img_pipe.enable_vae_slicing()
            
            self._loaded = True
            logger.info("Style enhancer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing style enhancer: {str(e)}")
            self._loaded = False
    
    def is_loaded(self) -> bool:
        """Check if the style enhancer is loaded"""
        return self._loaded
    
    def get_available_styles(self) -> List[str]:
        """Get list of available enhancement styles"""
        return list(self.styles.keys())
    
    def enhance_image(self, image_path: str, style: str = "professional", 
                     enhancement_level: float = 0.8, resolution: str = "1024x1024",
                     additional_prompts: List[str] = None) -> Dict:
        """
        Enhance image using specified style
        
        Args:
            image_path: Path to input image
            style: Enhancement style
            enhancement_level: Enhancement strength (0-1)
            resolution: Output resolution
            additional_prompts: Additional prompts to include
            
        Returns:
            Dictionary with enhancement results
        """
        try:
            if not self._loaded:
                raise RuntimeError("Style enhancer not loaded")
            
            if style not in self.styles:
                raise ValueError(f"Unknown style: {style}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Resize image
            target_size = self._parse_resolution(resolution)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Get style configuration
            style_config = self.styles[style].copy()
            
            # Adjust strength based on enhancement level
            style_config["strength"] = style_config["strength"] * enhancement_level
            
            # Build prompt
            prompt = style_config["prompt"]
            if additional_prompts:
                prompt += ", " + ", ".join(additional_prompts)
            
            # Generate enhanced image
            with torch.no_grad():
                result = self.img2img_pipe(
                    prompt=prompt,
                    image=image,
                    strength=style_config["strength"],
                    guidance_scale=style_config["guidance_scale"],
                    negative_prompt=style_config["negative_prompt"],
                    num_inference_steps=20,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
            
            enhanced_image = result.images[0]
            
            # Apply post-processing
            enhanced_image = self._apply_post_processing(enhanced_image, style)
            
            return {
                "enhanced_image": enhanced_image,
                "style": style,
                "enhancement_level": enhancement_level,
                "resolution": resolution,
                "prompt_used": prompt,
                "negative_prompt": style_config["negative_prompt"],
                "strength": style_config["strength"],
                "guidance_scale": style_config["guidance_scale"]
            }
            
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            raise
    
    def _parse_resolution(self, resolution: str) -> Tuple[int, int]:
        """Parse resolution string to tuple"""
        try:
            if 'x' in resolution:
                w, h = map(int, resolution.split('x'))
                return (w, h)
            else:
                size = int(resolution)
                return (size, size)
        except:
            return (1024, 1024)  # Default
    
    def _apply_post_processing(self, image: Image.Image, style: str) -> Image.Image:
        """
        Apply post-processing based on style
        
        Args:
            image: Enhanced image
            style: Style used for enhancement
            
        Returns:
            Post-processed image
        """
        try:
            # Style-specific post-processing
            if style == "professional":
                # Slight sharpening and contrast enhancement
                image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)
                
            elif style == "glamorous":
                # Soft glow effect
                image = self._apply_soft_glow(image)
                
            elif style == "casual":
                # Warm tone adjustment
                image = self._adjust_color_temperature(image, warmth=0.1)
                
            elif style == "artistic":
                # Slight saturation boost
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.2)
                
            elif style == "vintage":
                # Vintage film effect
                image = self._apply_vintage_effect(image)
                
            elif style == "modern":
                # Enhanced sharpening and vibrance
                image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.15)
            
            return image
            
        except Exception as e:
            logger.error(f"Error applying post-processing: {str(e)}")
            return image
    
    def _apply_soft_glow(self, image: Image.Image) -> Image.Image:
        """Apply soft glow effect"""
        try:
            # Create a blurred version
            blurred = image.filter(ImageFilter.GaussianBlur(radius=3))
            
            # Blend with original
            blended = Image.blend(image, blurred, 0.2)
            
            return blended
        except:
            return image
    
    def _adjust_color_temperature(self, image: Image.Image, warmth: float) -> Image.Image:
        """Adjust color temperature"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Apply warm tint
            if warmth > 0:
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + warmth), 0, 255)  # Red
                img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - warmth * 0.3), 0, 255)  # Blue
            
            return Image.fromarray(img_array.astype(np.uint8))
        except:
            return image
    
    def _apply_vintage_effect(self, image: Image.Image) -> Image.Image:
        """Apply vintage film effect"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Add slight sepia tone
            sepia_filter = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            
            # Apply sepia (subtle)
            sepia_img = img_array @ sepia_filter.T
            sepia_img = np.clip(sepia_img, 0, 255)
            
            # Blend with original
            blended = img_array * 0.7 + sepia_img * 0.3
            
            # Add slight vignette
            blended = self._add_vignette(blended)
            
            return Image.fromarray(blended.astype(np.uint8))
        except:
            return image
    
    def _add_vignette(self, img_array: np.ndarray) -> np.ndarray:
        """Add subtle vignette effect"""
        try:
            h, w = img_array.shape[:2]
            
            # Create vignette mask
            x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
            radius = np.sqrt(x**2 + y**2)
            vignette = 1 - np.clip(radius - 0.5, 0, 1) * 0.3
            
            # Apply vignette
            for i in range(img_array.shape[2]):
                img_array[:, :, i] = img_array[:, :, i] * vignette
            
            return img_array
        except:
            return img_array
    
    def batch_enhance(self, image_paths: List[str], style: str = "professional",
                     enhancement_level: float = 0.8, resolution: str = "1024x1024") -> List[Dict]:
        """
        Enhance multiple images in batch
        
        Args:
            image_paths: List of image paths
            style: Enhancement style
            enhancement_level: Enhancement strength
            resolution: Output resolution
            
        Returns:
            List of enhancement results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.enhance_image(
                    image_path=image_path,
                    style=style,
                    enhancement_level=enhancement_level,
                    resolution=resolution
                )
                results.append({
                    "image_path": image_path,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def create_style_comparison(self, image_path: str, styles: List[str] = None,
                              enhancement_level: float = 0.8) -> Dict:
        """
        Create comparison of different styles
        
        Args:
            image_path: Path to input image
            styles: List of styles to compare (default: all styles)
            enhancement_level: Enhancement strength
            
        Returns:
            Dictionary with style comparisons
        """
        if styles is None:
            styles = list(self.styles.keys())
        
        comparisons = {}
        
        for style in styles:
            try:
                result = self.enhance_image(
                    image_path=image_path,
                    style=style,
                    enhancement_level=enhancement_level
                )
                comparisons[style] = {
                    "success": True,
                    "enhanced_image": result["enhanced_image"],
                    "config": result
                }
            except Exception as e:
                comparisons[style] = {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "original_image": image_path,
            "enhancement_level": enhancement_level,
            "styles_compared": styles,
            "comparisons": comparisons
        }
    
    def get_style_info(self, style: str) -> Dict:
        """
        Get detailed information about a style
        
        Args:
            style: Style name
            
        Returns:
            Style information dictionary
        """
        if style not in self.styles:
            return {"error": f"Unknown style: {style}"}
        
        style_config = self.styles[style]
        
        return {
            "name": style,
            "prompt": style_config["prompt"],
            "negative_prompt": style_config["negative_prompt"],
            "default_strength": style_config["strength"],
            "guidance_scale": style_config["guidance_scale"],
            "description": self._get_style_description(style)
        }
    
    def _get_style_description(self, style: str) -> str:
        """Get description for a style"""
        descriptions = {
            "professional": "Clean, polished look suitable for business and professional contexts",
            "glamorous": "Enhanced features with sophisticated styling and dramatic lighting",
            "casual": "Natural, warm enhancement perfect for everyday photos",
            "artistic": "Creative and unique aesthetic with stylized elements",
            "vintage": "Nostalgic retro look with film-inspired tones and textures",
            "modern": "Contemporary style with sharp details and vibrant colors"
        }
        return descriptions.get(style, "Custom style enhancement")
    
    def add_custom_style(self, name: str, prompt: str, negative_prompt: str = "",
                        strength: float = 0.4, guidance_scale: float = 7.5):
        """
        Add custom enhancement style
        
        Args:
            name: Style name
            prompt: Enhancement prompt
            negative_prompt: Negative prompt
            strength: Enhancement strength
            guidance_scale: Guidance scale
        """
        self.styles[name] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "guidance_scale": guidance_scale
        }
        
        logger.info(f"Added custom style: {name}")
    
    def remove_custom_style(self, name: str):
        """Remove custom style"""
        if name in self.styles:
            del self.styles[name]
            logger.info(f"Removed custom style: {name}")
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        try:
            if self.img2img_pipe is not None:
                del self.img2img_pipe
                self.img2img_pipe = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("GPU memory cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up memory: {str(e)}") 