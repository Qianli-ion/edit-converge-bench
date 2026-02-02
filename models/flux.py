"""FLUX Kontext model wrapper via FAL AI."""

import os
import io
import base64
import tempfile
from PIL import Image
from typing import Optional
from .base import BaseEditModel


class FluxKontextModel(BaseEditModel):
    """
    Wrapper for FLUX Kontext models via FAL AI.
    
    Supports:
    - fal-ai/flux-kontext/dev (dev version, cheaper)
    - fal-ai/flux-pro/kontext (pro version)
    - fal-ai/flux-pro/kontext/max (max version, most capable)
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_id: str = "fal-ai/flux-kontext/dev",
        num_inference_steps: int = 28,
        guidance_scale: float = 2.5
    ):
        """
        Initialize the FLUX Kontext model.
        
        Args:
            api_key: FAL AI API key. If None, reads from FAL_KEY env var.
            model_id: FAL model ID (dev, pro, or max variant).
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale.
        """
        self.api_key = api_key or os.environ.get("FAL_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set FAL_KEY env var or pass api_key parameter."
            )
        
        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self._client = None
    
    @property
    def name(self) -> str:
        """Return model name based on variant."""
        if "max" in self.model_id:
            return "flux-kontext-max"
        elif "pro" in self.model_id and "kontext" in self.model_id:
            return "flux-kontext-pro"
        else:
            return "flux-kontext-dev"
    
    def _get_client(self):
        """Lazy initialization of FAL client."""
        if self._client is None:
            try:
                import fal_client
                # Set credentials
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
        
        # Call FAL API
        result = client.subscribe(
            self.model_id,
            arguments={
                "prompt": prompt,
                "image_url": image_url,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "output_format": "png",
                "num_images": 1,
                "enable_safety_checker": False,
            }
        )
        
        # Get the result image
        images = result.get("images", [])
        if not images:
            raise RuntimeError("No image returned from FLUX API")
        
        image_info = images[0]
        image_url = image_info.get("url")
        
        if not image_url:
            raise RuntimeError("No image URL in response")
        
        # Download the image
        import urllib.request
        with urllib.request.urlopen(image_url) as response:
            image_data = response.read()
        
        return Image.open(io.BytesIO(image_data)).convert("RGB")


class FluxKontextDev(FluxKontextModel):
    """FLUX Kontext Dev variant (faster, cheaper)."""
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, model_id="fal-ai/flux-kontext/dev")


class FluxKontextPro(FluxKontextModel):
    """FLUX Kontext Pro variant (better quality)."""
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, model_id="fal-ai/flux-pro/kontext")


class FluxKontextMax(FluxKontextModel):
    """FLUX Kontext Max variant (most capable)."""
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, model_id="fal-ai/flux-pro/kontext/max")


# Convenience functions
def create_flux_dev(api_key: Optional[str] = None) -> FluxKontextDev:
    """Create FLUX Kontext Dev model."""
    return FluxKontextDev(api_key=api_key)

def create_flux_pro(api_key: Optional[str] = None) -> FluxKontextPro:
    """Create FLUX Kontext Pro model."""
    return FluxKontextPro(api_key=api_key)

def create_flux_max(api_key: Optional[str] = None) -> FluxKontextMax:
    """Create FLUX Kontext Max model."""
    return FluxKontextMax(api_key=api_key)
