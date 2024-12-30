import asyncio
import numpy as np
import torch
import torchvision.transforms.functional as TF
from typing import Any, List
from prefect import task
from diffusers import AutoencoderTiny
from PIL import Image
import io

from .base import BaseModelModule


class VAEModule(BaseModelModule):
    @task
    def __init__(self, model_name: str = "madebyollin/taesd", device: str = "cuda", concurrent_per_model: int = 2, version: str = 'latest'):
        """
        Args:
            model_name (str): The name/path of the model on HuggingFace
            device (str): The device to load model on (e.g. 'cuda:0', 'cpu')
            concurrent_per_model (int): Maximum number of concurrent tasks per model
            version (str): Model version to use
        """
        super().__init__(model_name, device, concurrent_per_model, version)

    @task
    async def _load_model_on_device(self, device: str, local_model_checkpoint_path: str, **kwargs) -> Any:
        """Load AutoencoderTiny model on specified device"""
        if not local_model_checkpoint_path:
            raise ValueError("Model checkpoint must be initialized")

        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            None,
            self._load_autoencoder,
            local_model_checkpoint_path,
            device
        )
        return model

    def _load_autoencoder(self, checkpoint_path: str, device: str) -> Any:
        """Helper method to load AutoencoderTiny model"""
        model = AutoencoderTiny.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16
        ).to(device)
        model.eval()
        return model

    def _preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess image bytes to tensor
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Convert to tensor and normalize to [-1, 1]
        image = TF.to_tensor(image)  # Scales to [0, 1]
        image = image * 2.0 - 1.0    # Scale to [-1, 1]
        
        return image

    @task
    def forward(self, model: AutoencoderTiny, frames: List[bytes]) -> List[np.ndarray]:
        """Encode frames using the VAE encoder
        
        Args:
            model: The loaded AutoencoderTiny model
            frames: List of image bytes to encode
            
        Returns:
            List of encoded frame latents as numpy arrays
        """
        encoded_frames = []
        
        with torch.no_grad():
            for frame_bytes in frames:
                # Preprocess image bytes to tensor
                image = self._preprocess_image(frame_bytes)
                
                # Add batch dimension and move to device
                image = image.unsqueeze(0)
                image = image.to(model.device, dtype=model.dtype)
                
                # Encode
                latents = model.encoder(image)
                
                # Convert to numpy and append
                encoded_frames.append(latents[0].cpu().numpy())
        
        return encoded_frames
