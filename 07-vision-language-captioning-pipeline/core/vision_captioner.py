import torch
import logging
from typing import Dict, List, Optional, Any
from PIL import Image
import time
import asyncio
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    GitProcessor, GitForCausalLM
)
import numpy as np

logger = logging.getLogger(__name__)

class VisionCaptioner:
    """Vision-based image captioning using multiple models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configurations
        self.model_configs = {
            "blip": {
                "model_name": "Salesforce/blip-image-captioning-base",
                "processor": None,
                "model": None
            },
            "blip2": {
                "model_name": "Salesforce/blip2-opt-2.7b",
                "processor": None,
                "model": None
            },
            "clip": {
                "model_name": "openai/clip-vit-base-patch32",
                "processor": None,
                "model": None
            },
            "git": {
                "model_name": "microsoft/git-base",
                "processor": None,
                "model": None
            }
        }
        
        # Loaded models
        self.loaded_models = {}
        
        logger.info(f"Vision Captioner initialized with device: {self.device}")
    
    async def initialize(self):
        """Initialize default models"""
        try:
            # Load default BLIP model
            await self.load_model("blip")
            logger.info("Default models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    async def load_model(self, model_type: str):
        """Load a specific model"""
        try:
            if model_type in self.loaded_models:
                return self.loaded_models[model_type]
            
            if model_type not in self.model_configs:
                raise ValueError(f"Unknown model type: {model_type}")
            
            logger.info(f"Loading {model_type} model...")
            
            if model_type == "blip":
                processor = BlipProcessor.from_pretrained(self.model_configs[model_type]["model_name"])
                model = BlipForConditionalGeneration.from_pretrained(self.model_configs[model_type]["model_name"])
                
            elif model_type == "blip2":
                processor = Blip2Processor.from_pretrained(self.model_configs[model_type]["model_name"])
                model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_configs[model_type]["model_name"],
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
                
            elif model_type == "clip":
                processor = CLIPProcessor.from_pretrained(self.model_configs[model_type]["model_name"])
                model = CLIPModel.from_pretrained(self.model_configs[model_type]["model_name"])
                
            elif model_type == "git":
                processor = GitProcessor.from_pretrained(self.model_configs[model_type]["model_name"])
                model = GitForCausalLM.from_pretrained(self.model_configs[model_type]["model_name"])
            
            # Move model to device
            model.to(self.device)
            
            self.loaded_models[model_type] = {
                "processor": processor,
                "model": model
            }
            
            logger.info(f"{model_type} model loaded successfully")
            return self.loaded_models[model_type]
            
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")
            raise
    
    async def caption_image(
        self,
        image_path: str,
        model_type: str = "blip",
        max_length: int = 50,
        temperature: float = 0.7,
        num_beams: int = 5
    ) -> Dict[str, Any]:
        """Generate caption for an image"""
        try:
            start_time = time.time()
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Load model if not already loaded
            if model_type not in self.loaded_models:
                await self.load_model(model_type)
            
            model_info = self.loaded_models[model_type]
            processor = model_info["processor"]
            model = model_info["model"]
            
            # Generate caption based on model type
            if model_type == "blip":
                caption = await self._caption_with_blip(image, processor, model, max_length, num_beams)
                
            elif model_type == "blip2":
                caption = await self._caption_with_blip2(image, processor, model, max_length, temperature)
                
            elif model_type == "clip":
                caption = await self._caption_with_clip(image, processor, model)
                
            elif model_type == "git":
                caption = await self._caption_with_git(image, processor, model, max_length, num_beams)
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            processing_time = time.time() - start_time
            
            # Calculate confidence (simplified)
            confidence = min(0.9, max(0.1, 1.0 - (processing_time / 10.0)))
            
            result = {
                "caption": caption,
                "confidence": confidence,
                "processing_time": processing_time,
                "model_type": model_type
            }
            
            logger.info(f"Generated caption: {caption} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error captioning image: {str(e)}")
            raise
    
    async def _caption_with_blip(
        self,
        image: Image.Image,
        processor,
        model,
        max_length: int,
        num_beams: int
    ) -> str:
        """Generate caption using BLIP model"""
        try:
            # Process image
            inputs = processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logger.error(f"Error with BLIP captioning: {str(e)}")
            raise
    
    async def _caption_with_blip2(
        self,
        image: Image.Image,
        processor,
        model,
        max_length: int,
        temperature: float
    ) -> str:
        """Generate caption using BLIP-2 model"""
        try:
            # Process image
            inputs = processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9
                )
            
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logger.error(f"Error with BLIP-2 captioning: {str(e)}")
            raise
    
    async def _caption_with_clip(
        self,
        image: Image.Image,
        processor,
        model
    ) -> str:
        """Generate caption using CLIP model with predefined text templates"""
        try:
            # Predefined text templates
            text_templates = [
                "a photo of a person",
                "a photo of an animal",
                "a photo of a car",
                "a photo of a building",
                "a photo of nature",
                "a photo of food",
                "a photo of a object",
                "a beautiful landscape",
                "a city view",
                "an indoor scene",
                "an outdoor scene",
                "a close-up shot",
                "a wide-angle view",
                "a professional photograph",
                "a casual photograph"
            ]
            
            # Process inputs
            inputs = processor(
                text=text_templates,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Calculate similarities
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get best matching template
            best_idx = probs.argmax().item()
            best_template = text_templates[best_idx]
            
            # Enhance the template with more specific description
            enhanced_caption = self._enhance_clip_caption(best_template, probs[0][best_idx].item())
            
            return enhanced_caption
            
        except Exception as e:
            logger.error(f"Error with CLIP captioning: {str(e)}")
            raise
    
    def _enhance_clip_caption(self, template: str, confidence: float) -> str:
        """Enhance CLIP caption with additional descriptive words"""
        enhancements = {
            "person": ["smiling", "standing", "walking", "sitting"],
            "animal": ["cute", "wild", "domestic", "playing"],
            "car": ["red", "blue", "modern", "vintage"],
            "building": ["tall", "historic", "modern", "beautiful"],
            "nature": ["green", "peaceful", "scenic", "wild"],
            "food": ["delicious", "fresh", "colorful", "appetizing"],
            "object": ["interesting", "unique", "colorful", "detailed"]
        }
        
        # Find relevant enhancements
        for key, adjectives in enhancements.items():
            if key in template:
                import random
                adjective = random.choice(adjectives)
                template = template.replace(f"a {key}", f"a {adjective} {key}")
                break
        
        # Add confidence-based qualifiers
        if confidence > 0.8:
            template = "A clear " + template
        elif confidence > 0.6:
            template = "A " + template
        else:
            template = "Possibly " + template
        
        return template
    
    async def _caption_with_git(
        self,
        image: Image.Image,
        processor,
        model,
        max_length: int,
        num_beams: int
    ) -> str:
        """Generate caption using GIT model"""
        try:
            # Process image
            inputs = processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values=inputs.pixel_values,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logger.error(f"Error with GIT captioning: {str(e)}")
            raise
    
    async def batch_caption_images(
        self,
        image_paths: List[str],
        model_type: str = "blip",
        max_length: int = 50,
        batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """Caption multiple images in batches"""
        try:
            results = []
            
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_results = []
                
                for path in batch_paths:
                    try:
                        result = await self.caption_image(
                            image_path=path,
                            model_type=model_type,
                            max_length=max_length
                        )
                        result["image_path"] = path
                        batch_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error captioning {path}: {str(e)}")
                        batch_results.append({
                            "image_path": path,
                            "caption": "Error generating caption",
                            "confidence": 0.0,
                            "error": str(e)
                        })
                
                results.extend(batch_results)
                
                # Allow other tasks to run
                await asyncio.sleep(0.1)
            
            logger.info(f"Batch captioned {len(results)} images")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch captioning: {str(e)}")
            raise
    
    async def compare_captions(
        self,
        image_path: str,
        model_types: List[str] = ["blip", "blip2", "clip"]
    ) -> Dict[str, Any]:
        """Compare captions from different models"""
        try:
            results = {}
            
            for model_type in model_types:
                try:
                    result = await self.caption_image(
                        image_path=image_path,
                        model_type=model_type
                    )
                    results[model_type] = result
                    
                except Exception as e:
                    logger.error(f"Error with {model_type}: {str(e)}")
                    results[model_type] = {
                        "caption": f"Error with {model_type}",
                        "confidence": 0.0,
                        "error": str(e)
                    }
            
            # Find best caption based on confidence
            best_model = max(results.keys(), key=lambda k: results[k].get("confidence", 0))
            
            comparison_result = {
                "image_path": image_path,
                "captions": results,
                "best_model": best_model,
                "best_caption": results[best_model]["caption"]
            }
            
            logger.info(f"Compared captions from {len(model_types)} models")
            return comparison_result
            
        except Exception as e:
            logger.error(f"Error comparing captions: {str(e)}")
            raise
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = self.model_configs[model_type]
        is_loaded = model_type in self.loaded_models
        
        return {
            "model_type": model_type,
            "model_name": config["model_name"],
            "is_loaded": is_loaded,
            "device": str(self.device),
            "capabilities": self._get_model_capabilities(model_type)
        }
    
    def _get_model_capabilities(self, model_type: str) -> Dict[str, bool]:
        """Get capabilities of a specific model"""
        capabilities = {
            "blip": {
                "conditional_generation": True,
                "beam_search": True,
                "temperature_sampling": False,
                "batch_processing": True
            },
            "blip2": {
                "conditional_generation": True,
                "beam_search": True,
                "temperature_sampling": True,
                "batch_processing": True
            },
            "clip": {
                "conditional_generation": False,
                "beam_search": False,
                "temperature_sampling": False,
                "batch_processing": True,
                "text_matching": True
            },
            "git": {
                "conditional_generation": True,
                "beam_search": True,
                "temperature_sampling": False,
                "batch_processing": True
            }
        }
        
        return capabilities.get(model_type, {})
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types"""
        return list(self.model_configs.keys())
    
    def unload_model(self, model_type: str):
        """Unload a specific model to free memory"""
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded {model_type} model")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        memory_info = {
            "loaded_models": list(self.loaded_models.keys()),
            "device": str(self.device)
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_reserved": torch.cuda.memory_reserved(),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory
            })
        
        return memory_info 