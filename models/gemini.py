"""Gemini 2.5 Flash Image (Nano Banana) model wrapper."""

import os
import io
import base64
from PIL import Image
from typing import Optional
from .base import BaseEditModel


class GeminiEditModel(BaseEditModel):
    """
    Wrapper for Gemini 2.5 Flash Image (Nano Banana) model.
    
    Uses the Google Generative AI API for image editing.
    """
    
    name = "gemini-flash-image"
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp-image-generation"
    ):
        """
        Initialize the Gemini model.
        
        Args:
            api_key: Google AI API key. If None, reads from GEMINI_API_KEY env var.
            model_name: Model variant to use.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set GEMINI_API_KEY env var or pass api_key parameter."
            )
        
        self.model_name = model_name
        self._client = None
        self._model = None
    
    def _get_client(self):
        """Lazy initialization of the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
                self._model = genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError(
                    "google-generativeai is required. "
                    "Install with: pip install google-generativeai"
                )
        return self._client, self._model
    
    def _image_to_part(self, image: Image.Image) -> dict:
        """Convert PIL Image to Gemini API format."""
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        return {
            "mime_type": "image/png",
            "data": base64.b64encode(image_bytes).decode("utf-8")
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
        genai, model = self._get_client()
        
        # Prepare the image part
        image_part = self._image_to_part(image)
        
        # Create the request with image and text
        response = model.generate_content(
            [
                {"inline_data": image_part},
                prompt
            ],
            generation_config=genai.GenerationConfig(
                response_modalities=["image", "text"]
            )
        )
        
        # Extract the generated image
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data is not None:
                image_data = part.inline_data.data
                image_bytes = base64.b64decode(image_data)
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        raise RuntimeError("No image returned from Gemini API")


class NanoBananaModel(GeminiEditModel):
    """Alias for Gemini Flash Image model (Nano Banana)."""
    name = "nano-banana"


# Convenience function
def create_gemini_model(api_key: Optional[str] = None) -> GeminiEditModel:
    """Create a Gemini editing model instance."""
    return GeminiEditModel(api_key=api_key)
