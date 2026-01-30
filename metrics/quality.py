"""Quality preservation metrics: PSNR, SSIM, LPIPS."""

import numpy as np
from PIL import Image
from typing import Union
from dataclasses import dataclass

# Lazy imports for optional dependencies
_ssim_fn = None
_lpips_model = None


@dataclass
class QualityMetrics:
    """Container for quality metrics."""
    psnr: float
    ssim: float
    lpips: float
    
    def to_dict(self) -> dict:
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips": self.lpips
        }


def psnr(img1: Image.Image, img2: Image.Image, max_val: float = 255.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Higher is better (more similar).
    
    Args:
        img1: First image (reference)
        img2: Second image (test)
        max_val: Maximum pixel value
        
    Returns:
        PSNR in dB
    """
    arr1 = np.array(img1.convert("RGB")).astype(np.float64)
    arr2 = np.array(img2.convert("RGB")).astype(np.float64)
    
    # Resize if dimensions don't match
    if arr1.shape != arr2.shape:
        img2_resized = img2.resize(img1.size, Image.Resampling.LANCZOS)
        arr2 = np.array(img2_resized.convert("RGB")).astype(np.float64)
    
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(max_val / np.sqrt(mse))


def ssim(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute Structural Similarity Index between two images.
    
    Higher is better (range: -1 to 1, typically 0 to 1).
    
    Args:
        img1: First image (reference)
        img2: Second image (test)
        
    Returns:
        SSIM score
    """
    global _ssim_fn
    
    if _ssim_fn is None:
        try:
            from skimage.metrics import structural_similarity
            _ssim_fn = structural_similarity
        except ImportError:
            raise ImportError(
                "scikit-image is required for SSIM. "
                "Install with: pip install scikit-image"
            )
    
    arr1 = np.array(img1.convert("RGB"))
    arr2 = np.array(img2.convert("RGB"))
    
    # Resize if dimensions don't match
    if arr1.shape != arr2.shape:
        img2_resized = img2.resize(img1.size, Image.Resampling.LANCZOS)
        arr2 = np.array(img2_resized.convert("RGB"))
    
    return _ssim_fn(arr1, arr2, channel_axis=2, data_range=255)


def lpips(img1: Image.Image, img2: Image.Image, net: str = "alex") -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) between two images.
    
    Lower is better (more similar). Range: 0 to ~1.
    
    Args:
        img1: First image (reference)
        img2: Second image (test)
        net: Backbone network ("alex", "vgg", "squeeze")
        
    Returns:
        LPIPS distance
    """
    global _lpips_model
    
    if _lpips_model is None:
        try:
            import lpips as lpips_lib
            import torch
            _lpips_model = lpips_lib.LPIPS(net=net)
            if torch.cuda.is_available():
                _lpips_model = _lpips_model.cuda()
        except ImportError:
            raise ImportError(
                "lpips is required. Install with: pip install lpips"
            )
    
    import torch
    from torchvision import transforms
    
    # Preprocess images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    t1 = transform(img1.convert("RGB")).unsqueeze(0)
    t2 = transform(img2.convert("RGB")).unsqueeze(0)
    
    if torch.cuda.is_available():
        t1 = t1.cuda()
        t2 = t2.cuda()
    
    with torch.no_grad():
        distance = _lpips_model(t1, t2)
    
    return distance.item()


def compute_quality_metrics(
    reference: Image.Image, 
    test: Image.Image,
    include_lpips: bool = True
) -> QualityMetrics:
    """
    Compute all quality metrics between reference and test images.
    
    Args:
        reference: Reference image (original I₀)
        test: Test image (after round-trip Iₙ')
        include_lpips: Whether to compute LPIPS (slower, requires GPU ideally)
        
    Returns:
        QualityMetrics dataclass with psnr, ssim, lpips
    """
    psnr_val = psnr(reference, test)
    ssim_val = ssim(reference, test)
    lpips_val = lpips(reference, test) if include_lpips else 0.0
    
    return QualityMetrics(
        psnr=psnr_val,
        ssim=ssim_val,
        lpips=lpips_val
    )
