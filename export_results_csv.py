#!/usr/bin/env python3
"""Export all benchmark results to CSV files for paper artifacts."""

import json
import glob
import csv
import os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "results_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS = {
    'flux': 'results_flux_10rounds',
    'gemini': 'results_gemini_10rounds',
    'grok': 'results_grok_10rounds',
    'qwen': 'results_qwen_10rounds',
    'nano_banana': 'results_nano-banana-edit_10rounds',
}

def load_all_results():
    """Load all results from all models."""
    all_data = []
    
    for model_name, result_dir in MODELS.items():
        pattern = str(BASE_DIR / result_dir / "results_*.json")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"No results for {model_name}")
            continue
        
        # Use most recent file
        d = json.load(open(files[-1]))
        
        for eval_result in d.get('evaluations', []):
            image_file = eval_result.get('image_filename', '')
            edit_pair = eval_result.get('edit_pair_name', '')
            image_type = eval_result.get('image_type', '')
            edit_category = eval_result.get('edit_category', '')
            
            for r in eval_result.get('rounds', []):
                if r.get('success'):
                    quality = r.get('quality', {})
                    semantic = r.get('semantic', {})
                    
                    row = {
                        'model': model_name,
                        'image': image_file,
                        'edit_pair': edit_pair,
                        'image_type': image_type,
                        'edit_category': edit_category,
                        'round': r['round'],
                        'psnr': quality.get('psnr'),
                        'ssim': quality.get('ssim'),
                        'lpips': quality.get('lpips'),
                        'clip_similarity': semantic.get('clip_similarity'),
                    }
                    all_data.append(row)
    
    return all_data

def export_detailed_csv(all_data):
    """Export detailed per-round results."""
    output_path = OUTPUT_DIR / "benchmark_results_detailed.csv"
    
    fieldnames = ['model', 'image', 'edit_pair', 'image_type', 'edit_category', 
                  'round', 'psnr', 'ssim', 'lpips', 'clip_similarity']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"Saved: {output_path} ({len(all_data)} rows)")
    return output_path

def export_summary_by_round(all_data):
    """Export summary statistics by model and round."""
    import numpy as np
    
    # Aggregate by model and round
    summary = {}
    for row in all_data:
        key = (row['model'], row['round'])
        if key not in summary:
            summary[key] = {'psnr': [], 'ssim': [], 'clip': []}
        if row['psnr'] is not None:
            summary[key]['psnr'].append(row['psnr'])
        if row['ssim'] is not None:
            summary[key]['ssim'].append(row['ssim'])
        if row['clip_similarity'] is not None:
            summary[key]['clip'].append(row['clip_similarity'])
    
    output_path = OUTPUT_DIR / "benchmark_summary_by_round.csv"
    
    fieldnames = ['model', 'round', 'n_samples', 
                  'psnr_mean', 'psnr_std', 
                  'ssim_mean', 'ssim_std',
                  'clip_mean', 'clip_std']
    
    rows = []
    for (model, round_num), metrics in sorted(summary.items()):
        rows.append({
            'model': model,
            'round': round_num,
            'n_samples': len(metrics['psnr']),
            'psnr_mean': np.mean(metrics['psnr']) if metrics['psnr'] else None,
            'psnr_std': np.std(metrics['psnr']) if metrics['psnr'] else None,
            'ssim_mean': np.mean(metrics['ssim']) if metrics['ssim'] else None,
            'ssim_std': np.std(metrics['ssim']) if metrics['ssim'] else None,
            'clip_mean': np.mean(metrics['clip']) if metrics['clip'] else None,
            'clip_std': np.std(metrics['clip']) if metrics['clip'] else None,
        })
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Saved: {output_path} ({len(rows)} rows)")
    return output_path

def export_summary_by_model(all_data):
    """Export final summary by model (round 10 only)."""
    import numpy as np
    
    # Filter to round 10
    r10_data = [r for r in all_data if r['round'] == 10]
    
    # Aggregate by model
    summary = {}
    for row in r10_data:
        model = row['model']
        if model not in summary:
            summary[model] = {'psnr': [], 'ssim': [], 'clip': [], 'n_evals': 0}
        summary[model]['n_evals'] += 1
        if row['psnr'] is not None:
            summary[model]['psnr'].append(row['psnr'])
        if row['ssim'] is not None:
            summary[model]['ssim'].append(row['ssim'])
        if row['clip_similarity'] is not None:
            summary[model]['clip'].append(row['clip_similarity'])
    
    output_path = OUTPUT_DIR / "benchmark_summary_final.csv"
    
    fieldnames = ['model', 'n_evaluations',
                  'psnr_mean', 'psnr_std', 
                  'ssim_mean', 'ssim_std',
                  'clip_mean', 'clip_std']
    
    rows = []
    for model, metrics in sorted(summary.items()):
        rows.append({
            'model': model,
            'n_evaluations': metrics['n_evals'],
            'psnr_mean': round(np.mean(metrics['psnr']), 4) if metrics['psnr'] else None,
            'psnr_std': round(np.std(metrics['psnr']), 4) if metrics['psnr'] else None,
            'ssim_mean': round(np.mean(metrics['ssim']), 4) if metrics['ssim'] else None,
            'ssim_std': round(np.std(metrics['ssim']), 4) if metrics['ssim'] else None,
            'clip_mean': round(np.mean(metrics['clip']), 4) if metrics['clip'] else None,
            'clip_std': round(np.std(metrics['clip']), 4) if metrics['clip'] else None,
        })
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Saved: {output_path} ({len(rows)} rows)")
    return output_path

def export_experiment_summary(all_data):
    """Export summary by experiment (aggregated across rounds)."""
    import numpy as np
    
    # Aggregate by model, image, edit_pair for round 10
    r10_data = [r for r in all_data if r['round'] == 10]
    
    output_path = OUTPUT_DIR / "benchmark_by_experiment.csv"
    
    fieldnames = ['model', 'image', 'edit_pair', 'image_type', 'edit_category',
                  'psnr_r10', 'ssim_r10', 'clip_r10']
    
    rows = []
    for row in r10_data:
        rows.append({
            'model': row['model'],
            'image': row['image'],
            'edit_pair': row['edit_pair'],
            'image_type': row['image_type'],
            'edit_category': row['edit_category'],
            'psnr_r10': round(row['psnr'], 4) if row['psnr'] else None,
            'ssim_r10': round(row['ssim'], 4) if row['ssim'] else None,
            'clip_r10': round(row['clip_similarity'], 4) if row['clip_similarity'] else None,
        })
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Saved: {output_path} ({len(rows)} rows)")
    return output_path

def export_timing_estimates():
    """Export estimated timing data."""
    output_path = OUTPUT_DIR / "model_timing_estimates.csv"
    
    # From our earlier analysis
    timing_data = [
        {'model': 'flux', 'sec_per_image': 4.1, 'relative_speed': 1.0},
        {'model': 'gemini', 'sec_per_image': 6.1, 'relative_speed': 1.5},
        {'model': 'qwen', 'sec_per_image': 7.2, 'relative_speed': 1.8},
        {'model': 'grok', 'sec_per_image': 10.5, 'relative_speed': 2.6},
        {'model': 'nano_banana', 'sec_per_image': 28.5, 'relative_speed': 7.0},
    ]
    
    fieldnames = ['model', 'sec_per_image', 'relative_speed']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_data)
    
    print(f"Saved: {output_path}")
    return output_path

def main():
    print("=" * 60)
    print("Exporting ConvergeBench Results to CSV")
    print("=" * 60)
    print()
    
    # Load all data
    all_data = load_all_results()
    print(f"Loaded {len(all_data)} total data points")
    print()
    
    # Export various CSV files
    export_detailed_csv(all_data)
    export_summary_by_round(all_data)
    export_summary_by_model(all_data)
    export_experiment_summary(all_data)
    export_timing_estimates()
    
    print()
    print("=" * 60)
    print(f"All files saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
