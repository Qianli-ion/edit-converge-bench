#!/usr/bin/env python3
"""Plot degradation curves across rounds for all models."""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "results_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS = {
    'FLUX': ('results_flux_10rounds', '#E74C3C'),      # Red
    'Gemini': ('results_gemini_10rounds', '#F39C12'),  # Orange
    'Grok': ('results_grok_10rounds', '#3498DB'),      # Blue
    'Qwen': ('results_qwen_10rounds', '#2ECC71'),      # Green
}

def load_metrics_by_round(result_dir, common_exps=None):
    """Load metrics averaged by round."""
    pattern = str(BASE_DIR / result_dir / "results_*.json")
    files = sorted(glob.glob(pattern))
    
    if not files:
        return None
    
    d = json.load(open(files[-1]))
    
    rounds_data = {i: {'ssim': [], 'psnr': [], 'clip': []} for i in range(1, 11)}
    
    for eval_result in d.get('evaluations', []):
        # Filter to common experiments if specified
        if common_exps is not None:
            key = (eval_result.get('image_filename'), eval_result.get('edit_pair_name'))
            if key not in common_exps:
                continue
        
        rounds = eval_result.get('rounds', [])
        for r in rounds:
            if r.get('success'):
                round_num = r['round']
                quality = r.get('quality', {})
                semantic = r.get('semantic', {})
                
                if 'ssim' in quality:
                    rounds_data[round_num]['ssim'].append(quality['ssim'])
                if 'psnr' in quality:
                    rounds_data[round_num]['psnr'].append(quality['psnr'])
                if 'clip_similarity' in semantic:
                    rounds_data[round_num]['clip'].append(semantic['clip_similarity'])
    
    # Compute means
    result = {'rounds': list(range(1, 11)), 'ssim': [], 'psnr': [], 'clip': []}
    for i in range(1, 11):
        result['ssim'].append(np.mean(rounds_data[i]['ssim']) if rounds_data[i]['ssim'] else np.nan)
        result['psnr'].append(np.mean(rounds_data[i]['psnr']) if rounds_data[i]['psnr'] else np.nan)
        result['clip'].append(np.mean(rounds_data[i]['clip']) if rounds_data[i]['clip'] else np.nan)
    
    return result

def find_common_experiments():
    """Find experiments common to all models."""
    model_exps = {}
    for model_name, (result_dir, _) in MODELS.items():
        pattern = str(BASE_DIR / result_dir / "results_*.json")
        files = sorted(glob.glob(pattern))
        if files:
            d = json.load(open(files[-1]))
            evals = d.get('evaluations', [])
            complete = [e for e in evals if e.get('rounds') and len(e['rounds']) == 10 and e['rounds'][-1].get('success')]
            model_exps[model_name] = set((e['image_filename'], e['edit_pair_name']) for e in complete)
    
    common = set.intersection(*model_exps.values()) if model_exps else set()
    return common

def plot_curves():
    """Create degradation curve plots."""
    # Find common experiments for fair comparison
    common_exps = find_common_experiments()
    print(f"Using {len(common_exps)} common experiments for fair comparison")
    
    # Load data for all models
    all_data = {}
    for model_name, (result_dir, color) in MODELS.items():
        data = load_metrics_by_round(result_dir, common_exps)
        if data:
            all_data[model_name] = (data, color)
            print(f"Loaded {model_name}")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Image Quality Degradation Over Round-Trip Edits', fontsize=14, fontweight='bold')
    
    rounds = list(range(1, 11))
    
    # Plot SSIM
    ax = axes[0]
    for model_name, (data, color) in all_data.items():
        ax.plot(rounds, data['ssim'], marker='o', color=color, label=model_name, linewidth=2, markersize=6)
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('SSIM (↑ better)', fontsize=11)
    ax.set_title('Structural Similarity', fontsize=12)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1)
    
    # Plot PSNR
    ax = axes[1]
    for model_name, (data, color) in all_data.items():
        ax.plot(rounds, data['psnr'], marker='s', color=color, label=model_name, linewidth=2, markersize=6)
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('PSNR in dB (↑ better)', fontsize=11)
    ax.set_title('Peak Signal-to-Noise Ratio', fontsize=12)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 10.5)
    
    # Plot CLIP Similarity
    ax = axes[2]
    for model_name, (data, color) in all_data.items():
        ax.plot(rounds, data['clip'], marker='^', color=color, label=model_name, linewidth=2, markersize=6)
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('CLIP Similarity (↑ better)', fontsize=11)
    ax.set_title('Semantic Similarity', fontsize=12)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 1)
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / "degradation_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    return output_path

if __name__ == "__main__":
    plot_curves()
