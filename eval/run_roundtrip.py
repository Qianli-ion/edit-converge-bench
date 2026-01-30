#!/usr/bin/env python3
"""
Core evaluation loop for ConvergeBench.

Runs round-trip edits and measures quality/semantic degradation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base import BaseEditModel
from metrics.quality import compute_quality_metrics, QualityMetrics
from metrics.semantic import compute_semantic_metrics, SemanticMetrics


def load_model(model_name: str, api_key: Optional[str] = None) -> BaseEditModel:
    """Load an editing model by name."""
    if model_name in ["gemini", "nano-banana", "gemini-flash"]:
        from models.gemini import GeminiEditModel
        return GeminiEditModel(api_key=api_key)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_edit_pairs(path: Path) -> list[dict]:
    """Load edit pairs from JSON file."""
    with open(path) as f:
        return json.load(f)


def run_single_evaluation(
    model: BaseEditModel,
    image_path: Path,
    forward_prompt: str,
    backward_prompt: str,
    max_rounds: int = 10,
    output_dir: Optional[Path] = None,
    save_intermediates: bool = True,
    include_lpips: bool = True
) -> dict:
    """
    Run round-trip evaluation on a single image/edit pair.
    
    Args:
        model: The editing model to evaluate
        image_path: Path to source image
        forward_prompt: Forward edit prompt
        backward_prompt: Backward/inverse edit prompt
        max_rounds: Maximum number of round-trips
        output_dir: Directory to save intermediate images
        save_intermediates: Whether to save intermediate images
        include_lpips: Whether to compute LPIPS (slower)
        
    Returns:
        Dictionary with results for each round
    """
    # Load source image
    source_image = Image.open(image_path).convert("RGB")
    
    # Prepare output directory
    if output_dir and save_intermediates:
        output_dir.mkdir(parents=True, exist_ok=True)
        source_image.save(output_dir / "I0_source.png")
    
    results = {
        "source_image": str(image_path),
        "forward_prompt": forward_prompt,
        "backward_prompt": backward_prompt,
        "model": model.name,
        "rounds": []
    }
    
    current_image = source_image
    
    for n in range(1, max_rounds + 1):
        round_result = {"round": n}
        
        try:
            # Forward edit
            print(f"  Round {n}: Forward edit...")
            forward_image = model.edit(current_image, forward_prompt)
            
            if output_dir and save_intermediates:
                forward_image.save(output_dir / f"I{n}_forward.png")
            
            # Backward edit
            print(f"  Round {n}: Backward edit...")
            backward_image = model.edit(forward_image, backward_prompt)
            
            if output_dir and save_intermediates:
                backward_image.save(output_dir / f"I{n}_backward.png")
            
            # Compute metrics (compare backward result to original source)
            print(f"  Round {n}: Computing metrics...")
            quality = compute_quality_metrics(
                source_image, backward_image, include_lpips=include_lpips
            )
            semantic = compute_semantic_metrics(source_image, backward_image)
            
            round_result["quality"] = quality.to_dict()
            round_result["semantic"] = semantic.to_dict()
            round_result["success"] = True
            
            # Update current image for next round
            current_image = backward_image
            
        except Exception as e:
            print(f"  Round {n}: Error - {e}")
            round_result["success"] = False
            round_result["error"] = str(e)
            break  # Stop on first error
        
        results["rounds"].append(round_result)
    
    return results


def run_benchmark(
    model_name: str,
    images_dir: Path,
    edit_pairs_path: Path,
    output_dir: Path,
    max_rounds: int = 10,
    api_key: Optional[str] = None,
    include_lpips: bool = True,
    limit: Optional[int] = None
) -> dict:
    """
    Run the full benchmark evaluation.
    
    Args:
        model_name: Name of model to evaluate
        images_dir: Directory containing source images
        edit_pairs_path: Path to edit_pairs.json
        output_dir: Directory for results
        max_rounds: Maximum round-trips per test case
        api_key: API key for model
        include_lpips: Whether to compute LPIPS
        limit: Limit number of test cases (for debugging)
        
    Returns:
        Full benchmark results dictionary
    """
    print(f"Loading model: {model_name}")
    model = load_model(model_name, api_key)
    
    print(f"Loading edit pairs from: {edit_pairs_path}")
    edit_pairs = load_edit_pairs(edit_pairs_path)
    
    # Get all images
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    if limit:
        image_files = image_files[:limit]
    
    print(f"Found {len(image_files)} images, {len(edit_pairs)} edit pairs")
    
    # Prepare results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "metadata": {
            "model": model_name,
            "timestamp": timestamp,
            "max_rounds": max_rounds,
            "num_images": len(image_files),
            "num_edit_pairs": len(edit_pairs)
        },
        "evaluations": []
    }
    
    # Run evaluations
    total = len(image_files) * len(edit_pairs)
    count = 0
    
    for image_path in image_files:
        for edit_pair in edit_pairs:
            count += 1
            print(f"\n[{count}/{total}] {image_path.name} - {edit_pair['name']}")
            
            # Create output subdirectory
            eval_output_dir = output_dir / f"{image_path.stem}_{edit_pair['name']}"
            
            eval_result = run_single_evaluation(
                model=model,
                image_path=image_path,
                forward_prompt=edit_pair["forward"],
                backward_prompt=edit_pair["backward"],
                max_rounds=max_rounds,
                output_dir=eval_output_dir,
                include_lpips=include_lpips
            )
            eval_result["edit_pair_name"] = edit_pair["name"]
            eval_result["edit_category"] = edit_pair.get("category", "unknown")
            
            results["evaluations"].append(eval_result)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"results_{model_name}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ConvergeBench evaluation")
    parser.add_argument("--model", type=str, default="gemini",
                        help="Model to evaluate (gemini, nano-banana)")
    parser.add_argument("--images-dir", type=Path, default=Path("data/images"),
                        help="Directory containing source images")
    parser.add_argument("--edit-pairs", type=Path, default=Path("data/edit_pairs.json"),
                        help="Path to edit pairs JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Output directory for results")
    parser.add_argument("--max-rounds", type=int, default=10,
                        help="Maximum number of round-trips")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for model")
    parser.add_argument("--no-lpips", action="store_true",
                        help="Skip LPIPS computation (faster)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test cases (for debugging)")
    
    args = parser.parse_args()
    
    run_benchmark(
        model_name=args.model,
        images_dir=args.images_dir,
        edit_pairs_path=args.edit_pairs,
        output_dir=args.output_dir,
        max_rounds=args.max_rounds,
        api_key=args.api_key,
        include_lpips=not args.no_lpips,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
