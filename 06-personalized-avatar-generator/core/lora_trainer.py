import os
import torch
import logging
from typing import Dict, List, Optional, Callable
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class PersonalizedDataset(Dataset):
    """Dataset for personalized avatar training"""
    
    def __init__(self, image_paths: List[str], tokenizer, size: int = 512):
        self.image_paths = image_paths
        self.tokenizer = tokenizer
        self.size = size
        self.instance_prompt = "a photo of sks person"
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Resize and center crop
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Tokenize prompt
        text_inputs = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids.flatten(),
            "attention_mask": text_inputs.attention_mask.flatten()
        }

class LoRATrainer:
    """LoRA fine-tuning trainer for Stable Diffusion"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = config.get("model_id", "runwayml/stable-diffusion-v1-5")
        
        # Initialize model components
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.noise_scheduler = None
        
        logger.info(f"LoRA Trainer initialized with device: {self.device}")
    
    def load_model(self):
        """Load Stable Diffusion model components"""
        try:
            logger.info("Loading Stable Diffusion model...")
            
            # Load tokenizer and text encoder
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.model_id, subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.model_id, subfolder="text_encoder"
            )
            
            # Load VAE and UNet
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            self.vae = pipeline.vae
            self.unet = pipeline.unet
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.model_id, subfolder="scheduler"
            )
            
            # Move to device
            self.text_encoder.to(self.device)
            self.vae.to(self.device)
            self.unet.to(self.device)
            
            # Enable gradient checkpointing
            self.unet.enable_gradient_checkpointing()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def setup_lora(self, rank: int = 32):
        """Setup LoRA layers for UNet"""
        try:
            logger.info(f"Setting up LoRA with rank {rank}...")
            
            # Add LoRA layers to UNet attention blocks
            lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]
                
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=rank
                )
            
            self.unet.set_attn_processor(lora_attn_procs)
            
            # Get trainable parameters
            lora_layers = AttnProcsLayers(self.unet.attn_processors)
            
            logger.info("LoRA layers setup completed")
            return lora_layers
            
        except Exception as e:
            logger.error(f"Error setting up LoRA: {str(e)}")
            raise
    
    async def train_lora(
        self,
        user_id: str,
        training_steps: int = 500,
        learning_rate: float = 1e-4,
        rank: int = 32,
        job_id: str = None,
        progress_callback: Optional[Callable] = None
    ):
        """Train LoRA model for personalized avatar generation"""
        try:
            logger.info(f"Starting LoRA training for user {user_id}")
            
            # Update progress
            if progress_callback:
                progress_callback({"status": "loading_model", "progress": 10})
            
            # Load model if not already loaded
            if self.unet is None:
                self.load_model()
            
            # Setup LoRA
            lora_layers = self.setup_lora(rank)
            lora_layers.to(self.device)
            
            # Get training images
            image_dir = Path("uploads") / user_id
            image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
            
            if len(image_paths) < 5:
                raise ValueError(f"Insufficient training images: {len(image_paths)}")
            
            # Create dataset and dataloader
            dataset = PersonalizedDataset(
                image_paths=[str(p) for p in image_paths],
                tokenizer=self.tokenizer
            )
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            # Setup optimizer and scheduler
            optimizer = torch.optim.AdamW(
                lora_layers.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-08
            )
            
            lr_scheduler = get_scheduler(
                "constant",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=training_steps
            )
            
            # Training loop
            logger.info(f"Starting training for {training_steps} steps...")
            
            if progress_callback:
                progress_callback({"status": "training", "progress": 20})
            
            self.unet.train()
            step = 0
            
            while step < training_steps:
                for batch in dataloader:
                    if step >= training_steps:
                        break
                    
                    # Move batch to device
                    pixel_values = batch["pixel_values"].to(self.device)
                    input_ids = batch["input_ids"].to(self.device)
                    
                    # Encode images
                    with torch.no_grad():
                        latents = self.vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor
                    
                    # Sample noise and timesteps
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (bsz,), device=latents.device
                    ).long()
                    
                    # Add noise to latents
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Encode prompts
                    with torch.no_grad():
                        encoder_hidden_states = self.text_encoder(input_ids)[0]
                    
                    # Predict noise
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    
                    # Calculate loss
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lora_layers.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    step += 1
                    
                    # Update progress
                    if step % 50 == 0:
                        progress_pct = 20 + (step / training_steps) * 70
                        if progress_callback:
                            progress_callback({
                                "status": "training",
                                "progress": int(progress_pct),
                                "step": step,
                                "total_steps": training_steps,
                                "loss": loss.item()
                            })
                        
                        logger.info(f"Step {step}/{training_steps}, Loss: {loss.item():.4f}")
                    
                    # Allow other async tasks to run
                    await asyncio.sleep(0)
            
            # Save LoRA weights
            model_dir = Path("models") / user_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            lora_state_dict = {
                k: v.clone().cpu() for k, v in lora_layers.state_dict().items()
            }
            
            torch.save(lora_state_dict, model_dir / "lora_weights.pt")
            
            # Save training config
            training_config = {
                "user_id": user_id,
                "training_steps": training_steps,
                "learning_rate": learning_rate,
                "rank": rank,
                "model_id": self.model_id,
                "num_training_images": len(image_paths)
            }
            
            with open(model_dir / "training_config.json", "w") as f:
                json.dump(training_config, f, indent=2)
            
            logger.info(f"LoRA training completed for user {user_id}")
            
            if progress_callback:
                progress_callback({
                    "status": "completed",
                    "progress": 100,
                    "model_path": str(model_dir),
                    "message": "Training completed successfully"
                })
            
        except Exception as e:
            logger.error(f"Error during LoRA training: {str(e)}")
            if progress_callback:
                progress_callback({
                    "status": "failed",
                    "progress": 0,
                    "error": str(e)
                })
            raise
    
    def load_lora_weights(self, user_id: str):
        """Load trained LoRA weights for a user"""
        try:
            model_dir = Path("models") / user_id
            weights_path = model_dir / "lora_weights.pt"
            
            if not weights_path.exists():
                raise FileNotFoundError(f"No trained model found for user {user_id}")
            
            # Load LoRA weights
            lora_state_dict = torch.load(weights_path, map_location=self.device)
            
            # Load training config
            config_path = model_dir / "training_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    training_config = json.load(f)
            else:
                training_config = {"rank": 32}  # Default rank
            
            # Setup LoRA layers
            lora_layers = self.setup_lora(training_config.get("rank", 32))
            lora_layers.load_state_dict(lora_state_dict)
            
            logger.info(f"LoRA weights loaded for user {user_id}")
            return lora_layers
            
        except Exception as e:
            logger.error(f"Error loading LoRA weights: {str(e)}")
            raise 