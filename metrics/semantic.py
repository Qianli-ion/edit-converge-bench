"""Semantic consistency metrics: CLIP similarity."""

import numpy as np
from PIL import Image
from typing import Optional
from dataclasses import dataclass

# Lazy imports
_clip_model = None
_clip_preprocess = None
_torch = None


@dataclass
class SemanticMetrics:
    """Container for semantic metrics."""
    clip_similarity: float
    
    def to_dict(self) -> dict:
        return {
            "clip_similarity": self.clip_similarity
        }


def _load_clip(model_name: str = "ViT-L/14"):
    """Load CLIP model lazily."""
    global _clip_model, _clip_preprocess, _torch
    
    if _clip_model is None:
        try:
            import clip
            import torch
            _torch = torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _clip_model, _clip_preprocess = clip.load(model_name, device=device)
            _clip_model.eval()
        except ImportError:
            raise ImportError(
                "CLIP is required. Install with: pip install git+https://github.com/openai/CLIP.git"
            )
    
    return _clip_model, _clip_preprocess


def clip_similarity(img1: Image.Image, img2: Image.Image, model_name: str = "ViT-L/14") -> float:
    """
    Compute CLIP-based visual similarity between two images.
    
    Higher is better (range: -1 to 1, typically 0.5 to 1 for similar images).
    
    Args:
        img1: First image (reference)
        img2: Second image (test)
        model_name: CLIP model variant
        
    Returns:
        Cosine similarity of CLIP embeddings
    """
    model, preprocess = _load_clip(model_name)
    
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    
    # Preprocess images
    t1 = preprocess(img1.convert("RGB")).unsqueeze(0).to(device)
    t2 = preprocess(img2.convert("RGB")).unsqueeze(0).to(device)
    
    with _torch.no_grad():
        # Get image embeddings
        emb1 = model.encode_image(t1)
        emb2 = model.encode_image(t2)
        
        # Normalize
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        similarity = (emb1 @ emb2.T).item()
    
    return similarity


def clip_text_image_similarity(
    image: Image.Image, 
    text: str, 
    model_name: str = "ViT-L/14"
) -> float:
    """
    Compute CLIP similarity between an image and text description.
    
    Useful for verifying if an edit was applied correctly.
    
    Args:
        image: Image to evaluate
        text: Text description
        model_name: CLIP model variant
        
    Returns:
        Cosine similarity between image and text embeddings
    """
    model, preprocess = _load_clip(model_name)
    
    try:
        import clip
    except ImportError:
        raise ImportError("CLIP is required.")
    
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    
    # Preprocess
    image_input = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    text_input = clip.tokenize([text]).to(device)
    
    with _torch.no_grad():
        image_emb = model.encode_image(image_input)
        text_emb = model.encode_text(text_input)
        
        # Normalize
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        
        similarity = (image_emb @ text_emb.T).item()
    
    return similarity


def compute_semantic_metrics(
    reference: Image.Image,
    test: Image.Image,
    model_name: str = "ViT-L/14"
) -> SemanticMetrics:
    """
    Compute all semantic metrics between reference and test images.
    
    Args:
        reference: Reference image (original I₀)
        test: Test image (after round-trip Iₙ')
        model_name: CLIP model variant
        
    Returns:
        SemanticMetrics dataclass with clip_similarity
    """
    sim = clip_similarity(reference, test, model_name)
    
    return SemanticMetrics(clip_similarity=sim)
