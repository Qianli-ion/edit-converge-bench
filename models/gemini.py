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
    
    Uses the Google GenAI API for image editing.
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
    
    def _get_client(self):
        """Lazy initialization of the GenAI client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai is required. "
                    "Install with: pip install google-genai"
                )
        return self._client
    
    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        """
        Apply an edit to an image based on a text prompt.
        
        Args:
            image: Input PIL Image
            prompt: Text prompt describing the edit
            
        Returns:
            Edited PIL Image
        """
        from google.genai import types
        
        client = self._get_client()
        
        # Convert PIL Image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        # Create the request with image and text
        response = client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )
        
        # Extract the generated image
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_bytes = part.inline_data.data
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        raise RuntimeError("No image returned from Gemini API")


class NanoBananaModel(GeminiEditModel):
    """Alias for Gemini Flash Image model (Nano Banana)."""
    name = "nano-banana"


# Convenience function
def create_gemini_model(api_key: Optional[str] = None) -> GeminiEditModel:
    """Create a Gemini editing model instance."""
    return GeminiEditModel(api_key=api_key)
