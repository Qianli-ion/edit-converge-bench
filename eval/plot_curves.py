#!/usr/bin/env python3
"""
Generate degradation curve plots from benchmark results.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_degradation_curves(
    aggregate_files: list[Path],
    output_path: Path,
    title: str = "ConvergeBench: Quality Degradation Over Round-Trips"
):
    """
    Plot degradation curves for multiple models.
    
    Args:
        aggregate_files: List of .aggregate.json files
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10.colors
    
    for i, file_path in enumerate(aggregate_files):
        with open(file_path) as f:
            data = json.load(f)
        
        model = data["model"]
        color = colors[i % len(colors)]
        rounds = list(range(1, len(data["psnr_by_round"]) + 1))
        
        # PSNR
        axes[0, 0].plot(rounds, data["psnr_by_round"], 
                        marker='o', label=model, color=color)
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("PSNR (dB)")
        axes[0, 0].set_title("PSNR Degradation (higher = better)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # SSIM
        axes[0, 1].plot(rounds, data["ssim_by_round"],
                        marker='o', label=model, color=color)
        axes[0, 1].set_xlabel("Round")
        axes[0, 1].set_ylabel("SSIM")
        axes[0, 1].set_title("SSIM Degradation (higher = better)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # LPIPS
        axes[1, 0].plot(rounds, data["lpips_by_round"],
                        marker='o', label=model, color=color)
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("LPIPS")
        axes[1, 0].set_title("LPIPS Distance (lower = better)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # CLIP Similarity
        axes[1, 1].plot(rounds, data["clip_by_round"],
                        marker='o', label=model, color=color)
        axes[1, 1].set_xlabel("Round")
        axes[1, 1].set_ylabel("CLIP Similarity")
        axes[1, 1].set_title("Semantic Similarity (higher = better)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to: {output_path}")


def plot_model_comparison(
    aggregate_files: list[Path],
    output_path: Path
):
    """
    Create a bar chart comparing models on headline metrics.
    
    Args:
        aggregate_files: List of .aggregate.json files
        output_path: Path to save the plot
    """
    models = []
    half_life_ssim = []
    auc_ssim = []
    
    for file_path in aggregate_files:
        with open(file_path) as f:
            data = json.load(f)
        
        models.append(data["model"])
        # Use max_rounds + 1 if half-life not reached
        hl = data["half_life_ssim"]
        half_life_ssim.append(hl if hl else data["max_rounds"] + 1)
        auc_ssim.append(data["auc_ssim"])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ConvergeBench: Model Comparison", fontsize=14, fontweight='bold')
    
    x = np.arange(len(models))
    width = 0.6
    
    # Half-life (higher is better)
    bars1 = axes[0].bar(x, half_life_ssim, width, color='steelblue')
    axes[0].set_ylabel("SSIM Half-Life (rounds)")
    axes[0].set_title("Convergence Half-Life (higher = more robust)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].bar_label(bars1, fmt='%.1f')
    
    # AUC (higher is better for SSIM)
    bars2 = axes[1].bar(x, auc_ssim, width, color='seagreen')
    axes[1].set_ylabel("SSIM AUC")
    axes[1].set_title("Area Under SSIM Curve (higher = better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].bar_label(bars2, fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate degradation plots")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing .aggregate.json files")
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"),
                        help="Output directory for plots")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all aggregate files
    aggregate_files = list(args.results_dir.glob("*.aggregate.json"))
    
    if not aggregate_files:
        print("No aggregate files found. Run compute_metrics.py first.")
        return
    
    print(f"Found {len(aggregate_files)} aggregate files")
    
    # Plot degradation curves
    plot_degradation_curves(
        aggregate_files,
        args.output_dir / "degradation_curves.png"
    )
    
    # Plot model comparison
    if len(aggregate_files) > 1:
        plot_model_comparison(
            aggregate_files,
            args.output_dir / "model_comparison.png"
        )


if __name__ == "__main__":
    main()
