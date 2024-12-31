import torch
import logging
from typing import Dict, List, Optional, Any
import time
import asyncio
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    GPT2LMHeadModel, GPT2Tokenizer
)
import openai
import os

logger = logging.getLogger(__name__)

class LanguageProcessor:
    """Language processing for caption enhancement and variation generation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configurations
        self.model_configs = {
            "t5": {
                "model_name": "t5-base",
                "tokenizer": None,
                "model": None
            },
            "bart": {
                "model_name": "facebook/bart-base",
                "tokenizer": None,
                "model": None
            },
            "gpt2": {
                "model_name": "gpt2-medium",
                "tokenizer": None,
                "model": None
            }
        }
        
        # Loaded models
        self.loaded_models = {}
        
        # OpenAI API key
        self.openai_api_key = config.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        logger.info(f"Language Processor initialized with device: {self.device}")
    
    async def initialize(self):
        """Initialize default models"""
        try:
            # Load default T5 model
            await self.load_model("t5")
            logger.info("Default language models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing language models: {str(e)}")
            raise
    
    async def load_model(self, model_type: str):
        """Load a specific language model"""
        try:
            if model_type in self.loaded_models:
                return self.loaded_models[model_type]
            
            if model_type not in self.model_configs:
                raise ValueError(f"Unknown model type: {model_type}")
            
            logger.info(f"Loading {model_type} model...")
            
            if model_type == "t5":
                tokenizer = T5Tokenizer.from_pretrained(self.model_configs[model_type]["model_name"])
                model = T5ForConditionalGeneration.from_pretrained(self.model_configs[model_type]["model_name"])
                
            elif model_type == "bart":
                tokenizer = BartTokenizer.from_pretrained(self.model_configs[model_type]["model_name"])
                model = BartForConditionalGeneration.from_pretrained(self.model_configs[model_type]["model_name"])
                
            elif model_type == "gpt2":
                tokenizer = GPT2Tokenizer.from_pretrained(self.model_configs[model_type]["model_name"])
                model = GPT2LMHeadModel.from_pretrained(self.model_configs[model_type]["model_name"])
                
                # Add pad token for GPT2
                tokenizer.pad_token = tokenizer.eos_token
            
            # Move model to device
            model.to(self.device)
            
            self.loaded_models[model_type] = {
                "tokenizer": tokenizer,
                "model": model
            }
            
            logger.info(f"{model_type} model loaded successfully")
            return self.loaded_models[model_type]
            
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")
            raise
    
    async def enhance_caption(
        self,
        caption: str,
        enhancement_type: str = "detailed",
        style: str = "descriptive",
        max_length: int = 100,
        model_type: str = "t5"
    ) -> Dict[str, Any]:
        """Enhance a caption with more detail or different style"""
        try:
            start_time = time.time()
            
            # Use OpenAI API if available for better results
            if self.openai_api_key and enhancement_type in ["detailed", "creative", "professional"]:
                enhanced_caption = await self._enhance_with_openai(caption, enhancement_type, style, max_length)
            else:
                enhanced_caption = await self._enhance_with_local_model(caption, enhancement_type, style, max_length, model_type)
            
            processing_time = time.time() - start_time
            
            result = {
                "enhanced_caption": enhanced_caption,
                "original_caption": caption,
                "enhancement_type": enhancement_type,
                "style": style,
                "processing_time": processing_time,
                "model_type": model_type
            }
            
            logger.info(f"Enhanced caption: {enhanced_caption}")
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing caption: {str(e)}")
            raise
    
    async def _enhance_with_openai(
        self,
        caption: str,
        enhancement_type: str,
        style: str,
        max_length: int
    ) -> str:
        """Enhance caption using OpenAI API"""
        try:
            # Create prompt based on enhancement type
            if enhancement_type == "detailed":
                prompt = f"Expand this image caption with more vivid details while keeping it accurate: '{caption}'"
            elif enhancement_type == "creative":
                prompt = f"Rewrite this image caption in a more creative and engaging way: '{caption}'"
            elif enhancement_type == "professional":
                prompt = f"Rewrite this image caption in a professional, formal tone: '{caption}'"
            else:
                prompt = f"Improve this image caption: '{caption}'"
            
            # Add style instruction
            if style == "poetic":
                prompt += " Use poetic language."
            elif style == "technical":
                prompt += " Use technical and precise language."
            elif style == "casual":
                prompt += " Use casual, conversational language."
            
            # Call OpenAI API
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    frequency_penalty=0.2,
                    presence_penalty=0.1
                )
            )
            
            enhanced_caption = response.choices[0].text.strip()
            return enhanced_caption
            
        except Exception as e:
            logger.error(f"Error with OpenAI enhancement: {str(e)}")
            # Fallback to local model
            return await self._enhance_with_local_model(caption, enhancement_type, style, max_length, "t5")
    
    async def _enhance_with_local_model(
        self,
        caption: str,
        enhancement_type: str,
        style: str,
        max_length: int,
        model_type: str
    ) -> str:
        """Enhance caption using local language model"""
        try:
            # Load model if not already loaded
            if model_type not in self.loaded_models:
                await self.load_model(model_type)
            
            model_info = self.loaded_models[model_type]
            tokenizer = model_info["tokenizer"]
            model = model_info["model"]
            
            # Create enhancement prompt
            prompt = self._create_enhancement_prompt(caption, enhancement_type, style)
            
            # Generate enhancement based on model type
            if model_type == "t5":
                enhanced_caption = await self._enhance_with_t5(prompt, tokenizer, model, max_length)
            elif model_type == "bart":
                enhanced_caption = await self._enhance_with_bart(prompt, tokenizer, model, max_length)
            elif model_type == "gpt2":
                enhanced_caption = await self._enhance_with_gpt2(prompt, tokenizer, model, max_length)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return enhanced_caption
            
        except Exception as e:
            logger.error(f"Error with local model enhancement: {str(e)}")
            raise
    
    def _create_enhancement_prompt(self, caption: str, enhancement_type: str, style: str) -> str:
        """Create enhancement prompt based on type and style"""
        if enhancement_type == "detailed":
            prompt = f"expand with details: {caption}"
        elif enhancement_type == "creative":
            prompt = f"rewrite creatively: {caption}"
        elif enhancement_type == "professional":
            prompt = f"rewrite professionally: {caption}"
        elif enhancement_type == "summary":
            prompt = f"summarize: {caption}"
        else:
            prompt = f"improve: {caption}"
        
        return prompt
    
    async def _enhance_with_t5(self, prompt: str, tokenizer, model, max_length: int) -> str:
        """Enhance using T5 model"""
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Decode output
            enhanced_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return enhanced_caption
            
        except Exception as e:
            logger.error(f"Error with T5 enhancement: {str(e)}")
            raise
    
    async def _enhance_with_bart(self, prompt: str, tokenizer, model, max_length: int) -> str:
        """Enhance using BART model"""
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Decode output
            enhanced_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return enhanced_caption
            
        except Exception as e:
            logger.error(f"Error with BART enhancement: {str(e)}")
            raise
    
    async def _enhance_with_gpt2(self, prompt: str, tokenizer, model, max_length: int) -> str:
        """Enhance using GPT-2 model"""
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs.input_ids.shape[1] + max_length,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode output and extract generated part
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            enhanced_caption = full_text[len(prompt):].strip()
            
            return enhanced_caption
            
        except Exception as e:
            logger.error(f"Error with GPT-2 enhancement: {str(e)}")
            raise
    
    async def generate_variations(
        self,
        caption: str,
        num_variations: int = 3,
        variation_type: str = "creative",
        model_type: str = "t5"
    ) -> Dict[str, Any]:
        """Generate multiple variations of a caption"""
        try:
            start_time = time.time()
            variations = []
            
            # Use OpenAI API if available
            if self.openai_api_key and variation_type in ["creative", "style", "tone"]:
                variations = await self._generate_variations_with_openai(caption, num_variations, variation_type)
            else:
                variations = await self._generate_variations_with_local_model(caption, num_variations, variation_type, model_type)
            
            processing_time = time.time() - start_time
            
            result = {
                "variations": variations,
                "original_caption": caption,
                "variation_type": variation_type,
                "num_variations": len(variations),
                "processing_time": processing_time,
                "model_type": model_type
            }
            
            logger.info(f"Generated {len(variations)} caption variations")
            return result
            
        except Exception as e:
            logger.error(f"Error generating variations: {str(e)}")
            raise
    
    async def _generate_variations_with_openai(
        self,
        caption: str,
        num_variations: int,
        variation_type: str
    ) -> List[str]:
        """Generate variations using OpenAI API"""
        try:
            variations = []
            
            for i in range(num_variations):
                if variation_type == "creative":
                    prompt = f"Rewrite this image caption in a different creative way (variation {i+1}): '{caption}'"
                elif variation_type == "style":
                    styles = ["poetic", "technical", "casual", "formal", "artistic"]
                    style = styles[i % len(styles)]
                    prompt = f"Rewrite this image caption in a {style} style: '{caption}'"
                elif variation_type == "tone":
                    tones = ["enthusiastic", "calm", "professional", "playful", "dramatic"]
                    tone = tones[i % len(tones)]
                    prompt = f"Rewrite this image caption with a {tone} tone: '{caption}'"
                else:
                    prompt = f"Provide an alternative version of this image caption: '{caption}'"
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        max_tokens=100,
                        temperature=0.8,
                        top_p=0.9,
                        frequency_penalty=0.3,
                        presence_penalty=0.2
                    )
                )
                
                variation = response.choices[0].text.strip()
                variations.append(variation)
            
            return variations
            
        except Exception as e:
            logger.error(f"Error with OpenAI variations: {str(e)}")
            return []
    
    async def _generate_variations_with_local_model(
        self,
        caption: str,
        num_variations: int,
        variation_type: str,
        model_type: str
    ) -> List[str]:
        """Generate variations using local language model"""
        try:
            variations = []
            
            # Load model if not already loaded
            if model_type not in self.loaded_models:
                await self.load_model(model_type)
            
            model_info = self.loaded_models[model_type]
            tokenizer = model_info["tokenizer"]
            model = model_info["model"]
            
            # Generate variations with different prompts/temperatures
            for i in range(num_variations):
                prompt = self._create_variation_prompt(caption, variation_type, i)
                temperature = 0.7 + (i * 0.1)  # Increase temperature for more variety
                
                if model_type == "t5":
                    variation = await self._generate_variation_with_t5(prompt, tokenizer, model, temperature)
                elif model_type == "bart":
                    variation = await self._generate_variation_with_bart(prompt, tokenizer, model, temperature)
                elif model_type == "gpt2":
                    variation = await self._generate_variation_with_gpt2(prompt, tokenizer, model, temperature)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                variations.append(variation)
            
            return variations
            
        except Exception as e:
            logger.error(f"Error with local model variations: {str(e)}")
            raise
    
    def _create_variation_prompt(self, caption: str, variation_type: str, index: int) -> str:
        """Create variation prompt based on type and index"""
        if variation_type == "creative":
            prompts = [
                f"rewrite creatively: {caption}",
                f"make more vivid: {caption}",
                f"add artistic flair: {caption}"
            ]
        elif variation_type == "style":
            prompts = [
                f"rewrite poetically: {caption}",
                f"rewrite technically: {caption}",
                f"rewrite casually: {caption}"
            ]
        elif variation_type == "tone":
            prompts = [
                f"rewrite enthusiastically: {caption}",
                f"rewrite calmly: {caption}",
                f"rewrite dramatically: {caption}"
            ]
        else:
            prompts = [
                f"paraphrase: {caption}",
                f"rephrase: {caption}",
                f"reword: {caption}"
            ]
        
        return prompts[index % len(prompts)]
    
    async def _generate_variation_with_t5(self, prompt: str, tokenizer, model, temperature: float) -> str:
        """Generate variation using T5 model"""
        try:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=4,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9
                )
            
            variation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return variation
            
        except Exception as e:
            logger.error(f"Error generating T5 variation: {str(e)}")
            raise
    
    async def _generate_variation_with_bart(self, prompt: str, tokenizer, model, temperature: float) -> str:
        """Generate variation using BART model"""
        try:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=4,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9
                )
            
            variation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return variation
            
        except Exception as e:
            logger.error(f"Error generating BART variation: {str(e)}")
            raise
    
    async def _generate_variation_with_gpt2(self, prompt: str, tokenizer, model, temperature: float) -> str:
        """Generate variation using GPT-2 model"""
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs.input_ids.shape[1] + 50,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            variation = full_text[len(prompt):].strip()
            
            return variation
            
        except Exception as e:
            logger.error(f"Error generating GPT-2 variation: {str(e)}")
            raise
    
    async def summarize_captions(
        self,
        captions: List[str],
        max_length: int = 50,
        model_type: str = "bart"
    ) -> Dict[str, Any]:
        """Summarize multiple captions into a single summary"""
        try:
            start_time = time.time()
            
            # Combine captions
            combined_text = " ".join(captions)
            
            # Use appropriate model for summarization
            if model_type not in self.loaded_models:
                await self.load_model(model_type)
            
            model_info = self.loaded_models[model_type]
            tokenizer = model_info["tokenizer"]
            model = model_info["model"]
            
            # Create summarization prompt
            prompt = f"summarize: {combined_text}"
            
            # Generate summary
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=True
                )
            
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            result = {
                "summary": summary,
                "original_captions": captions,
                "num_captions": len(captions),
                "processing_time": processing_time,
                "model_type": model_type
            }
            
            logger.info(f"Summarized {len(captions)} captions")
            return result
            
        except Exception as e:
            logger.error(f"Error summarizing captions: {str(e)}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available language models"""
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
            "device": str(self.device),
            "openai_available": bool(self.openai_api_key)
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_reserved": torch.cuda.memory_reserved()
            })
        
        return memory_info 