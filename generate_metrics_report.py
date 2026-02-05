#!/usr/bin/env python3
"""Generate quantitative metrics comparison across models."""

import json
import glob
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent

MODELS = {
    'FLUX': 'results_flux_10rounds',
    'Gemini': 'results_gemini_10rounds', 
    'Grok': 'results_grok_10rounds',
    'Qwen': 'results_qwen_10rounds',
}

def load_model_metrics(result_dir):
    """Load all metrics from a model's results."""
    pattern = str(BASE_DIR / result_dir / "results_*.json")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Use the most recent file
    files.sort()
    d = json.load(open(files[-1]))
    
    # Aggregate metrics by round
    rounds_data = {i: {'psnr': [], 'ssim': [], 'lpips': [], 'clip': []} for i in range(1, 11)}
    
    for eval_result in d.get('evaluations', []):
        rounds = eval_result.get('rounds', [])
        for r in rounds:
            if r.get('success'):
                round_num = r['round']
                quality = r.get('quality', {})
                semantic = r.get('semantic', {})
                
                if 'psnr' in quality:
                    rounds_data[round_num]['psnr'].append(quality['psnr'])
                if 'ssim' in quality:
                    rounds_data[round_num]['ssim'].append(quality['ssim'])
                if 'lpips' in quality:
                    rounds_data[round_num]['lpips'].append(quality['lpips'])
                if 'clip_similarity' in semantic:
                    rounds_data[round_num]['clip'].append(semantic['clip_similarity'])
    
    return rounds_data

def compute_stats(values):
    """Compute mean and std."""
    if not values:
        return None, None
    return np.mean(values), np.std(values)

def main():
    print("=" * 70)
    print("CONVERGEBENCH - Quantitative Metrics Report")
    print("=" * 70)
    print()
    
    all_metrics = {}
    
    for model_name, result_dir in MODELS.items():
        metrics = load_model_metrics(result_dir)
        if metrics:
            all_metrics[model_name] = metrics
            print(f"✓ Loaded {model_name}")
        else:
            print(f"✗ No data for {model_name}")
    
    print()
    
    # Print metrics table for each round
    for metric_name, metric_key, higher_better in [
        ('PSNR (dB)', 'psnr', True),
        ('SSIM', 'ssim', True),
        ('LPIPS', 'lpips', False),
        ('CLIP Similarity', 'clip', True),
    ]:
        print("-" * 70)
        print(f"{metric_name} {'(↑ better)' if higher_better else '(↓ better)'}")
        print("-" * 70)
        
        # Header
        print(f"{'Round':<8}", end="")
        for model in all_metrics.keys():
            print(f"{model:<15}", end="")
        print()
        
        # Data rows
        for round_num in [1, 2, 3, 5, 10]:
            print(f"R{round_num:<7}", end="")
            for model_name, metrics in all_metrics.items():
                values = metrics[round_num][metric_key]
                mean, std = compute_stats(values)
                if mean is not None:
                    print(f"{mean:>6.3f}±{std:<6.3f} ", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            print()
        print()
    
    # Summary ranking at round 10
    print("=" * 70)
    print("RANKING AT ROUND 10 (lower degradation = better)")
    print("=" * 70)
    
    rankings = []
    for model_name, metrics in all_metrics.items():
        r10 = metrics[10]
        psnr_mean, _ = compute_stats(r10['psnr'])
        ssim_mean, _ = compute_stats(r10['ssim'])
        lpips_mean, _ = compute_stats(r10['lpips'])
        clip_mean, _ = compute_stats(r10['clip'])
        
        # Composite score (normalized)
        if all(v is not None for v in [psnr_mean, ssim_mean, lpips_mean]):
            # Higher PSNR/SSIM = better, lower LPIPS = better
            score = psnr_mean/30 + ssim_mean - lpips_mean
            rankings.append((model_name, score, psnr_mean, ssim_mean, lpips_mean, clip_mean))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<6}{'Model':<12}{'PSNR':<10}{'SSIM':<10}{'LPIPS':<10}{'CLIP':<10}")
    print("-" * 58)
    for i, (model, score, psnr, ssim, lpips, clip) in enumerate(rankings, 1):
        psnr_s = f"{psnr:.2f}" if psnr else "N/A"
        ssim_s = f"{ssim:.3f}" if ssim else "N/A"
        lpips_s = f"{lpips:.3f}" if lpips else "N/A"
        clip_s = f"{clip:.3f}" if clip else "N/A"
        print(f"{i:<6}{model:<12}{psnr_s:<10}{ssim_s:<10}{lpips_s:<10}{clip_s:<10}")

if __name__ == "__main__":
    main()
