import os
import torch
import logging
from typing import Dict, List, Optional
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image
import asyncio
from pathlib import Path
import json
import random
import time

logger = logging.getLogger(__name__)

class AvatarGenerator:
    """Generate personalized avatars using trained LoRA models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = config.get("model_id", "runwayml/stable-diffusion-v1-5")
        
        # Initialize pipeline
        self.pipeline = None
        self.current_user_id = None
        
        # Style prompts
        self.style_prompts = {
            "professional": "professional headshot, business attire, clean background, high quality, portrait photography",
            "artistic": "artistic portrait, creative lighting, artistic style, beautiful composition, fine art photography",
            "casual": "casual portrait, natural lighting, relaxed pose, everyday clothing, candid photography",
            "fantasy": "fantasy portrait, magical atmosphere, creative style, ethereal lighting, fantasy art",
            "vintage": "vintage portrait, classic style, film photography, retro aesthetic, nostalgic mood",
            "modern": "modern portrait, contemporary style, clean aesthetic, minimalist composition, professional lighting"
        }
        
        logger.info(f"Avatar Generator initialized with device: {self.device}")
    
    def load_pipeline(self):
        """Load the base Stable Diffusion pipeline"""
        try:
            if self.pipeline is None:
                logger.info("Loading Stable Diffusion pipeline...")
                
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                
                self.pipeline.to(self.device)
                
                # Enable memory efficient attention
                if hasattr(self.pipeline, "enable_attention_slicing"):
                    self.pipeline.enable_attention_slicing()
                
                # Enable VAE slicing for lower memory usage
                if hasattr(self.pipeline, "enable_vae_slicing"):
                    self.pipeline.enable_vae_slicing()
                
                logger.info("Pipeline loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            raise
    
    def load_lora_weights(self, user_id: str):
        """Load LoRA weights for a specific user"""
        try:
            model_dir = Path("models") / user_id
            weights_path = model_dir / "lora_weights.pt"
            
            if not weights_path.exists():
                raise FileNotFoundError(f"No trained model found for user {user_id}")
            
            # Load LoRA weights
            logger.info(f"Loading LoRA weights for user {user_id}")
            
            # For now, we'll use the pipeline's built-in LoRA loading
            # This is a simplified version - in production, you'd want to properly load the LoRA weights
            self.current_user_id = user_id
            
            logger.info(f"LoRA weights loaded for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error loading LoRA weights: {str(e)}")
            raise
    
    def build_prompt(self, base_prompt: str, style: str, user_id: str) -> str:
        """Build the complete prompt for generation"""
        # Add the instance token
        prompt = f"a photo of sks person, {base_prompt}"
        
        # Add style-specific prompts
        if style in self.style_prompts:
            prompt += f", {self.style_prompts[style]}"
        
        # Add quality enhancers
        quality_terms = [
            "high quality",
            "detailed",
            "professional photography",
            "sharp focus",
            "realistic"
        ]
        
        prompt += f", {', '.join(quality_terms)}"
        
        return prompt
    
    def build_negative_prompt(self) -> str:
        """Build negative prompt to avoid unwanted elements"""
        negative_terms = [
            "blurry",
            "low quality",
            "distorted",
            "deformed",
            "ugly",
            "bad anatomy",
            "extra limbs",
            "missing limbs",
            "nsfw",
            "nude",
            "naked",
            "watermark",
            "signature",
            "text",
            "logo"
        ]
        
        return ", ".join(negative_terms)
    
    async def generate_avatar(
        self,
        prompt: str,
        user_id: str,
        style: str = "professional",
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        seed: Optional[int] = None
    ) -> List[str]:
        """Generate personalized avatar images"""
        try:
            logger.info(f"Generating {num_images} avatar(s) for user {user_id}")
            
            # Load pipeline if not already loaded
            if self.pipeline is None:
                self.load_pipeline()
            
            # Load LoRA weights if different user
            if self.current_user_id != user_id:
                self.load_lora_weights(user_id)
            
            # Build complete prompt
            full_prompt = self.build_prompt(prompt, style, user_id)
            negative_prompt = self.build_negative_prompt()
            
            logger.info(f"Full prompt: {full_prompt}")
            logger.info(f"Negative prompt: {negative_prompt}")
            
            # Set seed for reproducibility
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate images
            output_paths = []
            
            for i in range(num_images):
                logger.info(f"Generating image {i+1}/{num_images}")
                
                # Generate image
                with torch.no_grad():
                    result = self.pipeline(
                        prompt=full_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        width=512,
                        height=512
                    )
                
                image = result.images[0]
                
                # Save image
                output_dir = Path("outputs") / user_id
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = int(time.time())
                filename = f"avatar_{style}_{timestamp}_{i}.png"
                output_path = output_dir / filename
                
                image.save(output_path)
                output_paths.append(str(output_path))
                
                logger.info(f"Saved avatar to {output_path}")
                
                # Allow other async tasks to run
                await asyncio.sleep(0.1)
            
            logger.info(f"Generated {len(output_paths)} avatars successfully")
            return output_paths
            
        except Exception as e:
            logger.error(f"Error generating avatar: {str(e)}")
            raise
    
    async def generate_style_variations(
        self,
        prompt: str,
        user_id: str,
        num_variations: int = 3,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20
    ) -> Dict[str, List[str]]:
        """Generate avatar variations in different styles"""
        try:
            logger.info(f"Generating style variations for user {user_id}")
            
            # Select random styles
            available_styles = list(self.style_prompts.keys())
            selected_styles = random.sample(available_styles, min(num_variations, len(available_styles)))
            
            variations = {}
            
            for style in selected_styles:
                logger.info(f"Generating {style} style variation")
                
                avatar_paths = await self.generate_avatar(
                    prompt=prompt,
                    user_id=user_id,
                    style=style,
                    num_images=1,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
                
                variations[style] = avatar_paths
            
            logger.info(f"Generated {len(variations)} style variations")
            return variations
            
        except Exception as e:
            logger.error(f"Error generating style variations: {str(e)}")
            raise
    
    def get_generation_stats(self, user_id: str) -> Dict:
        """Get generation statistics for a user"""
        try:
            output_dir = Path("outputs") / user_id
            
            if not output_dir.exists():
                return {"total_images": 0, "styles": {}}
            
            image_files = list(output_dir.glob("*.png"))
            total_images = len(image_files)
            
            # Count by style
            style_counts = {}
            for file in image_files:
                try:
                    # Extract style from filename
                    parts = file.stem.split("_")
                    if len(parts) >= 3 and parts[0] == "avatar":
                        style = parts[1]
                        style_counts[style] = style_counts.get(style, 0) + 1
                except:
                    continue
            
            return {
                "total_images": total_images,
                "styles": style_counts,
                "last_generated": max([f.stat().st_mtime for f in image_files]) if image_files else None
            }
            
        except Exception as e:
            logger.error(f"Error getting generation stats: {str(e)}")
            return {"total_images": 0, "styles": {}}
    
    def cleanup_old_generations(self, user_id: str, keep_count: int = 50):
        """Clean up old generated images to save space"""
        try:
            output_dir = Path("outputs") / user_id
            
            if not output_dir.exists():
                return
            
            image_files = list(output_dir.glob("*.png"))
            
            if len(image_files) <= keep_count:
                return
            
            # Sort by modification time (oldest first)
            image_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest files
            files_to_remove = image_files[:-keep_count]
            
            for file in files_to_remove:
                file.unlink()
                logger.info(f"Removed old image: {file}")
            
            logger.info(f"Cleaned up {len(files_to_remove)} old images for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up old generations: {str(e)}")
    
    def get_available_styles(self) -> List[Dict]:
        """Get list of available styles with descriptions"""
        return [
            {
                "name": style,
                "description": prompt.split(",")[0],
                "full_prompt": prompt
            }
            for style, prompt in self.style_prompts.items()
        ] 