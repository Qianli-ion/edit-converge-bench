#!/usr/bin/env python3
"""
Smart benchmark runner for ConvergeBench.

Matches images with appropriate edit pairs based on type:
- portraits → accessories
- scenes → scene_objects  
- all → geometric

Supports --resume to skip already-completed evaluations.
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import Optional, Set, Tuple
from PIL import Image
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base import BaseEditModel
from metrics.quality import compute_quality_metrics
from metrics.semantic import compute_semantic_metrics


# Mapping: image type → allowed edit categories
IMAGE_EDIT_MAPPING = {
    "portrait": ["accessories", "geometric"],
    "scene": ["scene_objects", "geometric"]
}


def load_model(model_name: str, api_key: Optional[str] = None) -> BaseEditModel:
    """Load an editing model by name."""
    # Gemini (direct Google API)
    if model_name in ["gemini", "nano-banana", "gemini-flash"]:
        from models.gemini import GeminiEditModel
        return GeminiEditModel(api_key=api_key)
    
    # FLUX Kontext variants
    elif model_name in ["flux", "flux-dev", "flux-kontext-dev"]:
        from models.flux import FluxKontextDev
        return FluxKontextDev(api_key=api_key)
    elif model_name in ["flux-pro", "flux-kontext-pro"]:
        from models.flux import FluxKontextPro
        return FluxKontextPro(api_key=api_key)
    elif model_name in ["flux-max", "flux-kontext-max"]:
        from models.flux import FluxKontextMax
        return FluxKontextMax(api_key=api_key)
    
    # Seedream v4.5 (ByteDance via FAL)
    elif model_name in ["seedream", "seedream-v4.5", "bytedance"]:
        from models.fal_models import SeedreamModel
        return SeedreamModel(api_key=api_key)
    
    # Grok Imagine (xAI via FAL)
    elif model_name in ["grok", "grok-imagine"]:
        from models.fal_models import GrokImagineModel
        return GrokImagineModel(api_key=api_key)
    
    # Qwen Edit (Alibaba via FAL)
    elif model_name in ["qwen", "qwen-edit"]:
        from models.fal_models import QwenEditModel
        return QwenEditModel(api_key=api_key)
    
    # Nano Banana Edit (Gemini via FAL)
    elif model_name in ["nano-banana-edit", "nano-banana-fal", "gemini-fal"]:
        from models.fal_models import NanoBananaEditModel
        return NanoBananaEditModel(api_key=api_key)
    
    else:
        available = [
            "gemini", "flux-dev", "flux-pro", "flux-max",
            "seedream", "grok", "qwen", "nano-banana-edit"
        ]
        raise ValueError(f"Unknown model: {model_name}. Available: {', '.join(available)}")


def load_edit_pairs(path: Path) -> list[dict]:
    """Load edit pairs from JSON file."""
    with open(path) as f:
        return json.load(f)


def load_images_metadata(path: Path) -> list[dict]:
    """Load image metadata from JSON file."""
    with open(path) as f:
        return json.load(f)


def get_valid_pairs(image_meta: dict, edit_pairs: list[dict]) -> list[dict]:
    """Get edit pairs valid for this image type."""
    image_type = image_meta.get("type", "scene")
    allowed_categories = IMAGE_EDIT_MAPPING.get(image_type, ["geometric"])
    
    return [ep for ep in edit_pairs if ep.get("category") in allowed_categories]


def load_completed_evaluations(
    output_dir: Path, 
    max_rounds: int
) -> Tuple[Set[Tuple[str, str]], list[dict], Optional[Path]]:
    """
    Load completed evaluations from existing results files.
    
    Returns:
        - Set of (image_filename, edit_pair_name) tuples that are complete
        - List of all existing evaluation results (for merging)
        - Path to the most recent results file (or None)
    """
    completed = set()
    all_results = []
    latest_file = None
    latest_time = None
    
    # Find all results JSON files in output directory
    pattern = str(output_dir / "results_*.json")
    result_files = glob.glob(pattern)
    
    for filepath in result_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            # Track the latest file by timestamp in filename
            file_time = Path(filepath).stem.split("_")[-1]
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
                latest_file = Path(filepath)
            
            for eval_result in data.get("evaluations", []):
                img = eval_result.get("image_filename", "")
                edit = eval_result.get("edit_pair_name", "")
                rounds = eval_result.get("rounds", [])
                
                # Check if this evaluation is complete (all rounds successful)
                successful_rounds = sum(1 for r in rounds if r.get("success", False))
                
                if successful_rounds >= max_rounds:
                    completed.add((img, edit))
                    all_results.append(eval_result)
                    
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {filepath}: {e}")
            continue
    
    return completed, all_results, latest_file


def run_single_evaluation(
    model: BaseEditModel,
    image_path: Path,
    forward_prompt: str,
    backward_prompt: str,
    max_rounds: int = 5,
    output_dir: Optional[Path] = None,
    save_intermediates: bool = True,
    include_lpips: bool = False,
    retry_count: int = 2,
    retry_delay: float = 5.0
) -> dict:
    """
    Run round-trip evaluation on a single image/edit pair.
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
        
        # Retry logic for API calls
        for attempt in range(retry_count + 1):
            try:
                # Forward edit
                print(f"    Round {n}: Forward edit...", end=" ", flush=True)
                forward_image = model.edit(current_image, forward_prompt)
                print("✓")
                
                if output_dir and save_intermediates:
                    forward_image.save(output_dir / f"I{n}_forward.png")
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
                # Backward edit
                print(f"    Round {n}: Backward edit...", end=" ", flush=True)
                backward_image = model.edit(forward_image, backward_prompt)
                print("✓")
                
                if output_dir and save_intermediates:
                    backward_image.save(output_dir / f"I{n}_backward.png")
                
                # Compute metrics
                print(f"    Round {n}: Computing metrics...", end=" ", flush=True)
                quality = compute_quality_metrics(
                    source_image, backward_image, include_lpips=include_lpips
                )
                semantic = compute_semantic_metrics(source_image, backward_image)
                print("✓")
                
                round_result["quality"] = quality.to_dict()
                round_result["semantic"] = semantic.to_dict()
                round_result["success"] = True
                
                # Update current image for next round
                current_image = backward_image
                
                # Small delay between rounds
                time.sleep(1)
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < retry_count:
                    print(f"⚠ (retry {attempt + 1}/{retry_count})")
                    time.sleep(retry_delay)
                else:
                    print(f"✗ Error: {e}")
                    round_result["success"] = False
                    round_result["error"] = str(e)
        
        results["rounds"].append(round_result)
        
        # Stop if this round failed
        if not round_result.get("success", False):
            break
    
    return results


def run_smart_benchmark(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    max_rounds: int = 3,
    api_key: Optional[str] = None,
    include_lpips: bool = False,
    max_pairs_per_image: Optional[int] = None,
    resume: bool = False
) -> dict:
    """
    Run the smart benchmark with proper image-edit pairing.
    
    Args:
        resume: If True, skip evaluations that are already complete in existing results.
    """
    images_dir = data_dir / "images"
    edit_pairs_path = data_dir / "edit_pairs.json"
    images_meta_path = data_dir / "images.json"
    
    print(f"Loading model: {model_name}")
    model = load_model(model_name, api_key)
    
    print(f"Loading data from: {data_dir}")
    edit_pairs = load_edit_pairs(edit_pairs_path)
    images_meta = load_images_metadata(images_meta_path)
    
    # Check for existing results if resuming
    completed_evals = set()
    existing_results = []
    if resume:
        print(f"Checking for existing results in: {output_dir}")
        completed_evals, existing_results, latest_file = load_completed_evaluations(
            output_dir, max_rounds
        )
        if completed_evals:
            print(f"Found {len(completed_evals)} completed evaluations to skip")
            if latest_file:
                print(f"  Latest results file: {latest_file.name}")
        else:
            print("No completed evaluations found, starting fresh")
    
    # Prepare results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "metadata": {
            "model": model_name,
            "timestamp": timestamp,
            "max_rounds": max_rounds,
            "resumed": resume,
            "resumed_from_count": len(existing_results) if resume else 0,
        },
        "evaluations": list(existing_results)  # Start with existing complete results
    }
    
    # Build evaluation plan
    total_evals = 0
    eval_plan = []
    skipped = 0
    for img_meta in images_meta:
        valid_pairs = get_valid_pairs(img_meta, edit_pairs)
        if max_pairs_per_image:
            valid_pairs = valid_pairs[:max_pairs_per_image]
        for ep in valid_pairs:
            total_evals += 1
            eval_key = (img_meta["filename"], ep["name"])
            if resume and eval_key in completed_evals:
                skipped += 1
                continue
            eval_plan.append((img_meta, ep))
    
    remaining = len(eval_plan)
    print(f"\nTotal evaluations: {total_evals}")
    if resume:
        print(f"Already complete: {skipped}")
        print(f"Remaining to run: {remaining}")
    print(f"Max rounds per evaluation: {max_rounds}")
    print("-" * 50)
    
    if remaining == 0:
        print("\nAll evaluations already complete!")
        results_path = output_dir / f"results_{model_name}_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        return results
    
    # Run evaluations
    for idx, (img_meta, edit_pair) in enumerate(eval_plan, 1):
        image_path = images_dir / img_meta["filename"]
        print(f"\n[{idx}/{remaining}] {img_meta['filename']} × {edit_pair['name']}")
        print(f"  Type: {img_meta['type']} | Category: {edit_pair['category']}")
        
        # Create output subdirectory
        eval_output_dir = output_dir / f"{Path(img_meta['filename']).stem}_{edit_pair['name']}"
        
        eval_result = run_single_evaluation(
            model=model,
            image_path=image_path,
            forward_prompt=edit_pair["forward"],
            backward_prompt=edit_pair["backward"],
            max_rounds=max_rounds,
            output_dir=eval_output_dir,
            include_lpips=include_lpips
        )
        eval_result["image_filename"] = img_meta["filename"]
        eval_result["image_type"] = img_meta["type"]
        eval_result["edit_pair_name"] = edit_pair["name"]
        eval_result["edit_category"] = edit_pair["category"]
        
        results["evaluations"].append(eval_result)
        
        # Save intermediate results
        results_path = output_dir / f"results_{model_name}_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 50}")
    print(f"Benchmark complete!")
    print(f"Results saved to: {results_path}")
    print(f"Total evaluations in file: {len(results['evaluations'])}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ConvergeBench smart evaluation")
    parser.add_argument("--model", type=str, default="gemini",
                        help="Model to evaluate")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Data directory (contains images/, edit_pairs.json, images.json)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Output directory for results")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Maximum number of round-trips")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for model")
    parser.add_argument("--include-lpips", action="store_true",
                        help="Include LPIPS computation (slower)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Max edit pairs per image (for quick testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results, skipping completed evaluations")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    run_smart_benchmark(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_rounds=args.max_rounds,
        api_key=args.api_key,
        include_lpips=args.include_lpips,
        max_pairs_per_image=args.max_pairs,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
