"""Base class for image editing models."""

from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
from typing import Union
import io


class BaseEditModel(ABC):
    """Abstract base class for image editing models."""
    
    name: str = "base"
    
    @abstractmethod
    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        """
        Apply an edit to an image based on a text prompt.
        
        Args:
            image: Input PIL Image
            prompt: Text prompt describing the edit
            
        Returns:
            Edited PIL Image
        """
        pass
    
    def edit_from_path(self, image_path: Union[str, Path], prompt: str) -> Image.Image:
        """
        Convenience method to edit an image from a file path.
        
        Args:
            image_path: Path to input image
            prompt: Text prompt describing the edit
            
        Returns:
            Edited PIL Image
        """
        image = Image.open(image_path).convert("RGB")
        return self.edit(image, prompt)
    
    def round_trip(
        self, 
        image: Image.Image, 
        forward_prompt: str, 
        backward_prompt: str,
        n_rounds: int = 1
    ) -> list[Image.Image]:
        """
        Perform n round-trip edits and return all intermediate images.
        
        Args:
            image: Input PIL Image (I₀)
            forward_prompt: Prompt for forward edit
            backward_prompt: Prompt for backward/inverse edit
            n_rounds: Number of round-trips to perform
            
        Returns:
            List of images: [I₁, I₁', I₂, I₂', ..., Iₙ, Iₙ']
        """
        results = []
        current = image
        
        for _ in range(n_rounds):
            # Forward edit
            forward_result = self.edit(current, forward_prompt)
            results.append(forward_result)
            
            # Backward edit
            backward_result = self.edit(forward_result, backward_prompt)
            results.append(backward_result)
            
            # Use backward result as input for next round
            current = backward_result
            
        return results
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
