#!/usr/bin/env python3
"""
Aggregate and compute summary metrics from benchmark results.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all evaluations."""
    model: str
    num_evaluations: int
    max_rounds: int
    
    # Per-round averages
    psnr_by_round: list[float]
    ssim_by_round: list[float]
    lpips_by_round: list[float]
    clip_by_round: list[float]
    
    # Headline metrics
    half_life_psnr: Optional[float]  # Rounds until PSNR drops by 50%
    half_life_ssim: Optional[float]  # Rounds until SSIM drops by 50%
    auc_psnr: float  # Area under PSNR curve
    auc_ssim: float  # Area under SSIM curve


def compute_half_life(values: list[float], threshold: float = 0.5) -> Optional[float]:
    """
    Compute the half-life: number of rounds until metric degrades by threshold.
    
    For PSNR/SSIM (higher is better), we find when value drops below threshold * initial.
    
    Args:
        values: List of metric values by round
        threshold: Fraction of initial value to trigger half-life
        
    Returns:
        Half-life in rounds, or None if never reached
    """
    if not values:
        return None
    
    initial = values[0]
    target = initial * threshold
    
    for i, v in enumerate(values):
        if v < target:
            # Interpolate between rounds
            if i > 0:
                prev = values[i-1]
                frac = (prev - target) / (prev - v) if prev != v else 0
                return i + frac
            return float(i + 1)
    
    return None  # Never dropped below threshold


def compute_auc(values: list[float], normalize: bool = True) -> float:
    """
    Compute area under the curve using trapezoidal rule.
    
    Args:
        values: List of metric values by round
        normalize: Whether to normalize by number of rounds
        
    Returns:
        AUC value
    """
    if len(values) < 2:
        return values[0] if values else 0.0
    
    auc = np.trapezoid(values)
    
    if normalize:
        auc /= len(values)
    
    return auc


def aggregate_results(results: dict) -> AggregateMetrics:
    """
    Aggregate results from multiple evaluations.
    
    Args:
        results: Full benchmark results dictionary
        
    Returns:
        AggregateMetrics with averaged values
    """
    model = results["metadata"]["model"]
    max_rounds = results["metadata"]["max_rounds"]
    evaluations = results["evaluations"]
    
    # Collect metrics by round
    psnr_by_round = [[] for _ in range(max_rounds)]
    ssim_by_round = [[] for _ in range(max_rounds)]
    lpips_by_round = [[] for _ in range(max_rounds)]
    clip_by_round = [[] for _ in range(max_rounds)]
    
    for eval_result in evaluations:
        for round_data in eval_result["rounds"]:
            if not round_data.get("success", False):
                continue
            
            n = round_data["round"] - 1  # 0-indexed
            if n < max_rounds:
                psnr_by_round[n].append(round_data["quality"]["psnr"])
                ssim_by_round[n].append(round_data["quality"]["ssim"])
                lpips_by_round[n].append(round_data["quality"]["lpips"])
                clip_by_round[n].append(round_data["semantic"]["clip_similarity"])
    
    # Average per round
    avg_psnr = [np.mean(vals) if vals else 0 for vals in psnr_by_round]
    avg_ssim = [np.mean(vals) if vals else 0 for vals in ssim_by_round]
    avg_lpips = [np.mean(vals) if vals else 0 for vals in lpips_by_round]
    avg_clip = [np.mean(vals) if vals else 0 for vals in clip_by_round]
    
    # Compute headline metrics
    half_life_psnr = compute_half_life(avg_psnr)
    half_life_ssim = compute_half_life(avg_ssim)
    auc_psnr = compute_auc(avg_psnr)
    auc_ssim = compute_auc(avg_ssim)
    
    return AggregateMetrics(
        model=model,
        num_evaluations=len(evaluations),
        max_rounds=max_rounds,
        psnr_by_round=avg_psnr,
        ssim_by_round=avg_ssim,
        lpips_by_round=avg_lpips,
        clip_by_round=avg_clip,
        half_life_psnr=half_life_psnr,
        half_life_ssim=half_life_ssim,
        auc_psnr=auc_psnr,
        auc_ssim=auc_ssim
    )


def print_summary(metrics: AggregateMetrics):
    """Print a summary of the metrics."""
    print(f"\n{'='*60}")
    print(f"ConvergeBench Results: {metrics.model}")
    print(f"{'='*60}")
    print(f"Evaluations: {metrics.num_evaluations}")
    print(f"Max rounds: {metrics.max_rounds}")
    print()
    
    print("Headline Metrics:")
    print(f"  PSNR Half-Life: {metrics.half_life_psnr:.1f} rounds" 
          if metrics.half_life_psnr else "  PSNR Half-Life: >max rounds")
    print(f"  SSIM Half-Life: {metrics.half_life_ssim:.1f} rounds"
          if metrics.half_life_ssim else "  SSIM Half-Life: >max rounds")
    print(f"  PSNR AUC: {metrics.auc_psnr:.2f}")
    print(f"  SSIM AUC: {metrics.auc_ssim:.4f}")
    print()
    
    print("Degradation by Round:")
    print(f"{'Round':<8} {'PSNR':<12} {'SSIM':<12} {'LPIPS':<12} {'CLIP':<12}")
    print("-" * 56)
    for i in range(len(metrics.psnr_by_round)):
        print(f"{i+1:<8} {metrics.psnr_by_round[i]:<12.2f} "
              f"{metrics.ssim_by_round[i]:<12.4f} "
              f"{metrics.lpips_by_round[i]:<12.4f} "
              f"{metrics.clip_by_round[i]:<12.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compute aggregate metrics")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing result JSON files")
    parser.add_argument("--results-file", type=Path, default=None,
                        help="Specific results file to process")
    
    args = parser.parse_args()
    
    if args.results_file:
        files = [args.results_file]
    else:
        files = list(args.results_dir.glob("results_*.json"))
    
    for file_path in files:
        print(f"\nProcessing: {file_path}")
        with open(file_path) as f:
            results = json.load(f)
        
        metrics = aggregate_results(results)
        print_summary(metrics)
        
        # Save aggregate metrics
        output_path = file_path.with_suffix(".aggregate.json")
        with open(output_path, "w") as f:
            json.dump({
                "model": metrics.model,
                "num_evaluations": metrics.num_evaluations,
                "max_rounds": metrics.max_rounds,
                "psnr_by_round": metrics.psnr_by_round,
                "ssim_by_round": metrics.ssim_by_round,
                "lpips_by_round": metrics.lpips_by_round,
                "clip_by_round": metrics.clip_by_round,
                "half_life_psnr": metrics.half_life_psnr,
                "half_life_ssim": metrics.half_life_ssim,
                "auc_psnr": metrics.auc_psnr,
                "auc_ssim": metrics.auc_ssim
            }, f, indent=2)
        print(f"Saved aggregate metrics to: {output_path}")


if __name__ == "__main__":
    main()
