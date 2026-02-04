"""FAL AI model wrappers for image editing."""

import os
import io
import base64
from PIL import Image
from typing import Optional
from .base import BaseEditModel


class BaseFalEditModel(BaseEditModel):
    """Base class for FAL AI editing models."""
    
    model_id: str = ""
    uses_image_urls_list: bool = False  # Some models use image_urls (list), others use image_url (single)
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FAL model.
        
        Args:
            api_key: FAL AI API key. If None, reads from FAL_KEY env var.
        """
        self.api_key = api_key or os.environ.get("FAL_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set FAL_KEY env var or pass api_key parameter."
            )
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of FAL client."""
        if self._client is None:
            try:
                import fal_client
                os.environ["FAL_KEY"] = self.api_key
                self._client = fal_client
            except ImportError:
                raise ImportError(
                    "fal-client is required. "
                    "Install with: pip install fal-client"
                )
        return self._client
    
    def _image_to_data_uri(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URI."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    
    def _build_arguments(self, image_url: str, prompt: str) -> dict:
        """Build API arguments. Override in subclasses for custom params."""
        if self.uses_image_urls_list:
            return {
                "prompt": prompt,
                "image_urls": [image_url],
                "num_images": 1,
                "enable_safety_checker": False,
            }
        else:
            return {
                "prompt": prompt,
                "image_url": image_url,
                "num_images": 1,
                "enable_safety_checker": False,
            }
    
    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        """
        Apply an edit to an image based on a text prompt.
        
        Args:
            image: Input PIL Image
            prompt: Text prompt describing the edit
            
        Returns:
            Edited PIL Image
        """
        client = self._get_client()
        
        # Convert image to data URI
        image_url = self._image_to_data_uri(image)
        
        # Build arguments
        arguments = self._build_arguments(image_url, prompt)
        
        # Call FAL API
        result = client.subscribe(
            self.model_id,
            arguments=arguments
        )
        
        # Get the result image
        images = result.get("images", [])
        if not images:
            raise RuntimeError(f"No image returned from {self.model_id}")
        
        image_info = images[0]
        image_url = image_info.get("url")
        
        if not image_url:
            raise RuntimeError("No image URL in response")
        
        # Download the image
        import urllib.request
        with urllib.request.urlopen(image_url) as response:
            image_data = response.read()
        
        return Image.open(io.BytesIO(image_data)).convert("RGB")


class SeedreamModel(BaseFalEditModel):
    """ByteDance Seedream v4.5 model for image editing."""
    
    name = "seedream-v4.5"
    model_id = "fal-ai/bytedance/seedream/v4.5/edit"
    uses_image_urls_list = True
    
    def _build_arguments(self, image_url: str, prompt: str) -> dict:
        return {
            "prompt": prompt,
            "image_urls": [image_url],
            "num_images": 1,
            "enable_safety_checker": False,
        }


class GrokImagineModel(BaseFalEditModel):
    """xAI Grok Imagine model for image editing."""
    
    name = "grok-imagine"
    model_id = "xai/grok-imagine-image/edit"
    uses_image_urls_list = False
    
    def _build_arguments(self, image_url: str, prompt: str) -> dict:
        return {
            "prompt": prompt,
            "image_url": image_url,
            "num_images": 1,
            "output_format": "png",
        }


class QwenEditModel(BaseFalEditModel):
    """Qwen Image Edit model for image editing."""
    
    name = "qwen-edit"
    model_id = "fal-ai/qwen-image-edit"
    uses_image_urls_list = False
    
    def _build_arguments(self, image_url: str, prompt: str) -> dict:
        return {
            "prompt": prompt,
            "image_url": image_url,
            "num_images": 1,
            "num_inference_steps": 30,
            "guidance_scale": 4,
            "enable_safety_checker": False,
            "output_format": "png",
        }


class NanoBananaEditModel(BaseFalEditModel):
    """
    Nano Banana Edit model (Gemini 2.5 Flash via FAL).
    
    This is the FAL-hosted version of Google's Gemini image editing.
    Different from the direct Google API version in gemini.py.
    """
    
    name = "nano-banana-edit"
    model_id = "fal-ai/nano-banana/edit"
    uses_image_urls_list = True
    
    def _build_arguments(self, image_url: str, prompt: str) -> dict:
        return {
            "prompt": prompt,
            "image_urls": [image_url],
            "num_images": 1,
            "output_format": "png",
            "aspect_ratio": "auto",
            "limit_generations": True,
        }


# Convenience functions
def create_seedream(api_key: Optional[str] = None) -> SeedreamModel:
    """Create Seedream v4.5 model."""
    return SeedreamModel(api_key=api_key)

def create_grok(api_key: Optional[str] = None) -> GrokImagineModel:
    """Create Grok Imagine model."""
    return GrokImagineModel(api_key=api_key)

def create_qwen(api_key: Optional[str] = None) -> QwenEditModel:
    """Create Qwen Edit model."""
    return QwenEditModel(api_key=api_key)

def create_nano_banana_edit(api_key: Optional[str] = None) -> NanoBananaEditModel:
    """Create Nano Banana Edit model (Gemini via FAL)."""
    return NanoBananaEditModel(api_key=api_key)
