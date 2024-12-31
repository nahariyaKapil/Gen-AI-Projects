import torch
import logging
from typing import Dict, List, Optional, Any
from PIL import Image
import time
import asyncio
from pathlib import Path
import random
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    DiffusionPipeline,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler
)
import numpy as np

logger = logging.getLogger(__name__)

class ImageEditor:
    """Image generation and editing using Stable Diffusion"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = config.get("model_id", "runwayml/stable-diffusion-v1-5")
        
        # Pipelines
        self.text_to_image_pipeline = None
        self.image_to_image_pipeline = None
        self.inpaint_pipeline = None
        self.instruct_pix2pix_pipeline = None
        
        # Style prompts
        self.style_prompts = {
            "photorealistic": "photorealistic, high quality, detailed, professional photography",
            "artistic": "artistic, painting style, creative, beautiful artwork",
            "cartoon": "cartoon style, animated, colorful, stylized",
            "sketch": "pencil sketch, black and white, artistic drawing",
            "watercolor": "watercolor painting, soft colors, artistic",
            "oil_painting": "oil painting, traditional art, textured brushstrokes",
            "digital_art": "digital art, modern, clean, high resolution",
            "vintage": "vintage style, retro, aged, nostalgic"
        }
        
        logger.info(f"Image Editor initialized with device: {self.device}")
    
    async def initialize(self):
        """Initialize default pipelines"""
        try:
            # Load text-to-image pipeline
            await self.load_pipeline("text_to_image")
            logger.info("Default image generation pipeline loaded")
            
        except Exception as e:
            logger.error(f"Error initializing image editor: {str(e)}")
            raise
    
    async def load_pipeline(self, pipeline_type: str):
        """Load a specific pipeline"""
        try:
            logger.info(f"Loading {pipeline_type} pipeline...")
            
            if pipeline_type == "text_to_image":
                if self.text_to_image_pipeline is None:
                    self.text_to_image_pipeline = StableDiffusionPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                    self.text_to_image_pipeline.to(self.device)
                    
                    # Enable memory efficient attention
                    if hasattr(self.text_to_image_pipeline, "enable_attention_slicing"):
                        self.text_to_image_pipeline.enable_attention_slicing()
                    
                    if hasattr(self.text_to_image_pipeline, "enable_vae_slicing"):
                        self.text_to_image_pipeline.enable_vae_slicing()
                
                return self.text_to_image_pipeline
            
            elif pipeline_type == "image_to_image":
                if self.image_to_image_pipeline is None:
                    self.image_to_image_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                    self.image_to_image_pipeline.to(self.device)
                    
                    if hasattr(self.image_to_image_pipeline, "enable_attention_slicing"):
                        self.image_to_image_pipeline.enable_attention_slicing()
                    
                    if hasattr(self.image_to_image_pipeline, "enable_vae_slicing"):
                        self.image_to_image_pipeline.enable_vae_slicing()
                
                return self.image_to_image_pipeline
            
            elif pipeline_type == "inpaint":
                if self.inpaint_pipeline is None:
                    self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                        "runwayml/stable-diffusion-inpainting",
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                    self.inpaint_pipeline.to(self.device)
                    
                    if hasattr(self.inpaint_pipeline, "enable_attention_slicing"):
                        self.inpaint_pipeline.enable_attention_slicing()
                    
                    if hasattr(self.inpaint_pipeline, "enable_vae_slicing"):
                        self.inpaint_pipeline.enable_vae_slicing()
                
                return self.inpaint_pipeline
            
            elif pipeline_type == "instruct_pix2pix":
                if self.instruct_pix2pix_pipeline is None:
                    self.instruct_pix2pix_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        "timbrooks/instruct-pix2pix",
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                    self.instruct_pix2pix_pipeline.to(self.device)
                    
                    if hasattr(self.instruct_pix2pix_pipeline, "enable_attention_slicing"):
                        self.instruct_pix2pix_pipeline.enable_attention_slicing()
                    
                    if hasattr(self.instruct_pix2pix_pipeline, "enable_vae_slicing"):
                        self.instruct_pix2pix_pipeline.enable_vae_slicing()
                
                return self.instruct_pix2pix_pipeline
            
            else:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")
                
        except Exception as e:
            logger.error(f"Error loading {pipeline_type} pipeline: {str(e)}")
            raise
    
    async def generate_image(
        self,
        prompt: str,
        style: str = "photorealistic",
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate image from text prompt"""
        try:
            start_time = time.time()
            
            # Load pipeline if not loaded
            if self.text_to_image_pipeline is None:
                await self.load_pipeline("text_to_image")
            
            # Build complete prompt
            full_prompt = self.build_prompt(prompt, style)
            negative_prompt = self.build_negative_prompt()
            
            # Set seed for reproducibility
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate images
            generated_images = []
            
            for i in range(num_images):
                logger.info(f"Generating image {i+1}/{num_images}")
                
                with torch.no_grad():
                    result = self.text_to_image_pipeline(
                        prompt=full_prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    )
                
                image = result.images[0]
                
                # Save image
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)
                
                timestamp = int(time.time())
                filename = f"generated_{style}_{timestamp}_{i}.png"
                output_path = output_dir / filename
                
                image.save(output_path)
                generated_images.append(str(output_path))
                
                logger.info(f"Generated image saved: {output_path}")
                
                # Allow other async tasks to run
                await asyncio.sleep(0.1)
            
            processing_time = time.time() - start_time
            
            result = {
                "generated_images": generated_images,
                "prompt": prompt,
                "full_prompt": full_prompt,
                "style": style,
                "dimensions": f"{width}x{height}",
                "num_images": len(generated_images),
                "processing_time": processing_time,
                "seed": seed
            }
            
            logger.info(f"Generated {len(generated_images)} images successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise
    
    async def edit_image(
        self,
        image_path: str,
        instruction: str,
        model_type: str = "instruct_pix2pix",
        strength: float = 0.7,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20
    ) -> Dict[str, Any]:
        """Edit image based on text instruction"""
        try:
            start_time = time.time()
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            if model_type == "instruct_pix2pix":
                edited_image = await self._edit_with_instruct_pix2pix(
                    image, instruction, strength, guidance_scale, num_inference_steps
                )
            elif model_type == "img2img":
                edited_image = await self._edit_with_img2img(
                    image, instruction, strength, guidance_scale, num_inference_steps
                )
            else:
                raise ValueError(f"Unknown editing model type: {model_type}")
            
            # Save edited image
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"edited_{timestamp}.png"
            output_path = output_dir / filename
            
            edited_image.save(output_path)
            
            processing_time = time.time() - start_time
            
            result = {
                "edited_image_path": str(output_path),
                "original_image": image_path,
                "instruction": instruction,
                "model_type": model_type,
                "processing_time": processing_time
            }
            
            logger.info(f"Edited image saved: {output_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error editing image: {str(e)}")
            raise
    
    async def _edit_with_instruct_pix2pix(
        self,
        image: Image.Image,
        instruction: str,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int
    ) -> Image.Image:
        """Edit image using InstructPix2Pix"""
        try:
            # Load pipeline if not loaded
            if self.instruct_pix2pix_pipeline is None:
                await self.load_pipeline("instruct_pix2pix")
            
            # Edit image
            with torch.no_grad():
                result = self.instruct_pix2pix_pipeline(
                    instruction,
                    image=image,
                    num_inference_steps=num_inference_steps,
                    image_guidance_scale=guidance_scale,
                    guidance_scale=guidance_scale
                )
            
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Error with InstructPix2Pix: {str(e)}")
            raise
    
    async def _edit_with_img2img(
        self,
        image: Image.Image,
        instruction: str,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int
    ) -> Image.Image:
        """Edit image using img2img pipeline"""
        try:
            # Load pipeline if not loaded
            if self.image_to_image_pipeline is None:
                await self.load_pipeline("image_to_image")
            
            # Use instruction as prompt
            negative_prompt = self.build_negative_prompt()
            
            # Edit image
            with torch.no_grad():
                result = self.image_to_image_pipeline(
                    prompt=instruction,
                    image=image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    negative_prompt=negative_prompt
                )
            
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Error with img2img: {str(e)}")
            raise
    
    def build_prompt(self, base_prompt: str, style: str) -> str:
        """Build complete prompt with style"""
        # Add style-specific prompts
        if style in self.style_prompts:
            prompt = f"{base_prompt}, {self.style_prompts[style]}"
        else:
            prompt = base_prompt
        
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
            "disfigured",
            "poorly drawn",
            "bad proportions",
            "gross proportions",
            "malformed",
            "mutated",
            "out of frame",
            "extra arms",
            "extra legs",
            "fused fingers",
            "too many fingers",
            "long neck",
            "duplicate",
            "morbid",
            "mutilated",
            "poorly drawn face",
            "bad face",
            "bad eyes",
            "bad hands",
            "text",
            "watermark",
            "logo",
            "signature"
        ]
        
        return ", ".join(negative_terms)
    
    async def apply_style_transfer(
        self,
        image_path: str,
        target_style: str,
        strength: float = 0.7,
        guidance_scale: float = 7.5
    ) -> Dict[str, Any]:
        """Apply style transfer to an image"""
        try:
            start_time = time.time()
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Load img2img pipeline
            if self.image_to_image_pipeline is None:
                await self.load_pipeline("image_to_image")
            
            # Create style prompt
            style_prompt = f"in the style of {target_style}, {self.style_prompts.get(target_style, 'artistic style')}"
            negative_prompt = self.build_negative_prompt()
            
            # Apply style transfer
            with torch.no_grad():
                result = self.image_to_image_pipeline(
                    prompt=style_prompt,
                    image=image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=20,
                    negative_prompt=negative_prompt
                )
            
            styled_image = result.images[0]
            
            # Save styled image
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"styled_{target_style}_{timestamp}.png"
            output_path = output_dir / filename
            
            styled_image.save(output_path)
            
            processing_time = time.time() - start_time
            
            result = {
                "styled_image_path": str(output_path),
                "original_image": image_path,
                "target_style": target_style,
                "style_prompt": style_prompt,
                "processing_time": processing_time
            }
            
            logger.info(f"Style transfer completed: {output_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error applying style transfer: {str(e)}")
            raise
    
    async def create_variations(
        self,
        image_path: str,
        num_variations: int = 3,
        variation_strength: float = 0.5,
        guidance_scale: float = 7.5
    ) -> Dict[str, Any]:
        """Create variations of an image"""
        try:
            start_time = time.time()
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Load img2img pipeline
            if self.image_to_image_pipeline is None:
                await self.load_pipeline("image_to_image")
            
            # Generate variations
            variations = []
            
            for i in range(num_variations):
                # Use generic prompt for variations
                prompt = "high quality, detailed, beautiful"
                negative_prompt = self.build_negative_prompt()
                
                # Vary the seed for each variation
                seed = random.randint(0, 2**32 - 1)
                generator = torch.Generator(device=self.device).manual_seed(seed)
                
                with torch.no_grad():
                    result = self.image_to_image_pipeline(
                        prompt=prompt,
                        image=image,
                        strength=variation_strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=20,
                        negative_prompt=negative_prompt,
                        generator=generator
                    )
                
                variation_image = result.images[0]
                
                # Save variation
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)
                
                timestamp = int(time.time())
                filename = f"variation_{i}_{timestamp}.png"
                output_path = output_dir / filename
                
                variation_image.save(output_path)
                variations.append(str(output_path))
                
                logger.info(f"Created variation {i+1}: {output_path}")
                
                # Allow other async tasks to run
                await asyncio.sleep(0.1)
            
            processing_time = time.time() - start_time
            
            result = {
                "variations": variations,
                "original_image": image_path,
                "num_variations": len(variations),
                "variation_strength": variation_strength,
                "processing_time": processing_time
            }
            
            logger.info(f"Created {len(variations)} variations")
            return result
            
        except Exception as e:
            logger.error(f"Error creating variations: {str(e)}")
            raise
    
    def get_available_styles(self) -> List[Dict[str, str]]:
        """Get available image styles"""
        return [
            {"name": style, "description": prompt}
            for style, prompt in self.style_prompts.items()
        ]
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about loaded pipelines"""
        return {
            "text_to_image_loaded": self.text_to_image_pipeline is not None,
            "image_to_image_loaded": self.image_to_image_pipeline is not None,
            "inpaint_loaded": self.inpaint_pipeline is not None,
            "instruct_pix2pix_loaded": self.instruct_pix2pix_pipeline is not None,
            "model_id": self.model_id,
            "device": str(self.device)
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        memory_info = {
            "device": str(self.device),
            "loaded_pipelines": []
        }
        
        if self.text_to_image_pipeline is not None:
            memory_info["loaded_pipelines"].append("text_to_image")
        if self.image_to_image_pipeline is not None:
            memory_info["loaded_pipelines"].append("image_to_image")
        if self.inpaint_pipeline is not None:
            memory_info["loaded_pipelines"].append("inpaint")
        if self.instruct_pix2pix_pipeline is not None:
            memory_info["loaded_pipelines"].append("instruct_pix2pix")
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_reserved": torch.cuda.memory_reserved()
            })
        
        return memory_info 