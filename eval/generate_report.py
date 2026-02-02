#!/usr/bin/env python3
"""
Generate HTML report from ConvergeBench results.

Creates a visual report with:
- Summary statistics
- Degradation curves
- Visual examples of round-trips
"""

import json
import base64
from pathlib import Path
from datetime import datetime
import argparse


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 data URI."""
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def generate_html_report(results_path: Path, output_dir: Path) -> Path:
    """Generate HTML report from results JSON."""
    
    with open(results_path) as f:
        results = json.load(f)
    
    metadata = results.get("metadata", {})
    evaluations = results.get("evaluations", [])
    
    # Compute summary statistics
    total_evals = len(evaluations)
    successful_evals = sum(1 for e in evaluations if e.get("rounds") and e["rounds"][0].get("success"))
    
    # Aggregate metrics by round
    round_metrics = {}
    for eval_result in evaluations:
        for round_data in eval_result.get("rounds", []):
            if not round_data.get("success"):
                continue
            n = round_data["round"]
            if n not in round_metrics:
                round_metrics[n] = {"psnr": [], "ssim": [], "clip": []}
            
            quality = round_data.get("quality", {})
            semantic = round_data.get("semantic", {})
            
            if quality.get("psnr"):
                round_metrics[n]["psnr"].append(quality["psnr"])
            if quality.get("ssim"):
                round_metrics[n]["ssim"].append(quality["ssim"])
            if semantic.get("clip_similarity"):
                round_metrics[n]["clip"].append(semantic["clip_similarity"])
    
    # Compute averages
    avg_metrics = {}
    for n, metrics in sorted(round_metrics.items()):
        avg_metrics[n] = {
            "psnr": sum(metrics["psnr"]) / len(metrics["psnr"]) if metrics["psnr"] else 0,
            "ssim": sum(metrics["ssim"]) / len(metrics["ssim"]) if metrics["ssim"] else 0,
            "clip": sum(metrics["clip"]) / len(metrics["clip"]) if metrics["clip"] else 0,
        }
    
    # Group evaluations by category
    by_category = {}
    for eval_result in evaluations:
        cat = eval_result.get("edit_category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(eval_result)
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ConvergeBench Report - {metadata.get('timestamp', 'Unknown')}</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
            --warning: #d29922;
            --danger: #f85149;
            --border: #30363d;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--accent), #a371f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        h2 {{
            font-size: 1.5rem;
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }}
        
        h3 {{
            font-size: 1.2rem;
            color: var(--text-secondary);
            margin: 1.5rem 0 0.75rem;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            margin-bottom: 2rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }}
        
        .stat-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: 600;
            color: var(--accent);
        }}
        
        .stat-label {{
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .metrics-table th,
        .metrics-table td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        .metrics-table th {{
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.875rem;
        }}
        
        .metrics-table tr:last-child td {{
            border-bottom: none;
        }}
        
        .example-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }}
        
        .example-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .example-header {{
            padding: 1rem;
            border-bottom: 1px solid var(--border);
        }}
        
        .example-title {{
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}
        
        .example-meta {{
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
        
        .example-images {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2px;
            background: var(--border);
        }}
        
        .example-images img {{
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            display: block;
        }}
        
        .example-labels {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            text-align: center;
            font-size: 0.75rem;
            color: var(--text-secondary);
            padding: 0.5rem;
        }}
        
        .example-metrics {{
            padding: 0.75rem 1rem;
            font-size: 0.85rem;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 1rem;
        }}
        
        .metric-badge {{
            background: var(--bg-tertiary);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
        }}
        
        .chart-container {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1.5rem 0;
        }}
        
        .bar-chart {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        
        .bar-row {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .bar-label {{
            width: 80px;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }}
        
        .bar-container {{
            flex: 1;
            height: 24px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .bar {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        
        .bar.psnr {{ background: linear-gradient(90deg, #58a6ff, #388bfd); }}
        .bar.ssim {{ background: linear-gradient(90deg, #3fb950, #2ea043); }}
        .bar.clip {{ background: linear-gradient(90deg, #a371f7, #8957e5); }}
        
        .bar-value {{
            width: 60px;
            text-align: right;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        
        .category-section {{
            margin: 2rem 0;
            padding: 1.5rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
        }}
        
        .category-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        
        .category-badge {{
            background: var(--bg-tertiary);
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
        
        footer {{
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}
        
        footer a {{
            color: var(--accent);
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 ConvergeBench Report</h1>
        <p class="subtitle">
            Model: <strong>{metadata.get('model', 'Unknown')}</strong> | 
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
            Max Rounds: {metadata.get('max_rounds', 'N/A')}
        </p>
        
        <h2>📊 Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_evals}</div>
                <div class="stat-label">Total Evaluations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{successful_evals}</div>
                <div class="stat-label">Successful (Round 1+)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(by_category)}</div>
                <div class="stat-label">Edit Categories</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{max(round_metrics.keys()) if round_metrics else 0}</div>
                <div class="stat-label">Max Rounds Completed</div>
            </div>
        </div>
        
        <h2>📈 Degradation by Round</h2>
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
            Average metrics across all evaluations. Lower degradation = better convergence.
        </p>
        
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Round</th>
                    <th>PSNR (dB) ↑</th>
                    <th>SSIM ↑</th>
                    <th>CLIP Sim ↑</th>
                    <th>Samples</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for n in sorted(avg_metrics.keys()):
        m = avg_metrics[n]
        sample_count = len(round_metrics[n]["psnr"])
        html += f"""
                <tr>
                    <td><strong>Round {n}</strong></td>
                    <td>{m['psnr']:.2f}</td>
                    <td>{m['ssim']:.4f}</td>
                    <td>{m['clip']:.4f}</td>
                    <td>{sample_count}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
        
        <h2>🖼️ Visual Examples</h2>
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
            Round-trip examples: Original → Forward Edit → Backward Edit (Round 1)
        </p>
        
        <div class="example-grid">
"""
    
    # Add visual examples (up to 12)
    example_count = 0
    for eval_result in evaluations:
        if example_count >= 12:
            break
        if not eval_result.get("rounds") or not eval_result["rounds"][0].get("success"):
            continue
        
        # Get image paths
        img_stem = Path(eval_result["image_filename"]).stem
        edit_name = eval_result["edit_pair_name"]
        eval_dir = output_dir / f"{img_stem}_{edit_name}"
        
        source_img = eval_dir / "I0_source.png"
        forward_img = eval_dir / "I1_forward.png"
        backward_img = eval_dir / "I1_backward.png"
        
        if not all(p.exists() for p in [source_img, forward_img, backward_img]):
            continue
        
        round1 = eval_result["rounds"][0]
        quality = round1.get("quality", {})
        semantic = round1.get("semantic", {})
        
        html += f"""
            <div class="example-card">
                <div class="example-header">
                    <div class="example-title">{edit_name.replace('_', ' ').title()}</div>
                    <div class="example-meta">{eval_result['image_filename']} • {eval_result['edit_category']}</div>
                </div>
                <div class="example-images">
                    <img src="{image_to_base64(source_img)}" alt="Original">
                    <img src="{image_to_base64(forward_img)}" alt="Forward">
                    <img src="{image_to_base64(backward_img)}" alt="Backward">
                </div>
                <div class="example-labels">
                    <span>Original</span>
                    <span>+ Edit</span>
                    <span>- Edit</span>
                </div>
                <div class="example-metrics">
                    <span class="metric-badge">PSNR: {quality.get('psnr', 0):.1f}</span>
                    <span class="metric-badge">SSIM: {quality.get('ssim', 0):.3f}</span>
                    <span class="metric-badge">CLIP: {semantic.get('clip_similarity', 0):.3f}</span>
                </div>
            </div>
"""
        example_count += 1
    
    html += """
        </div>
        
        <h2>📂 Results by Category</h2>
"""
    
    # Category breakdown
    for category, cat_evals in sorted(by_category.items()):
        successful = sum(1 for e in cat_evals if e.get("rounds") and e["rounds"][0].get("success"))
        
        # Calculate average metrics for this category
        cat_psnr, cat_ssim, cat_clip = [], [], []
        for e in cat_evals:
            if e.get("rounds") and e["rounds"][0].get("success"):
                r1 = e["rounds"][0]
                if r1.get("quality", {}).get("psnr"):
                    cat_psnr.append(r1["quality"]["psnr"])
                if r1.get("quality", {}).get("ssim"):
                    cat_ssim.append(r1["quality"]["ssim"])
                if r1.get("semantic", {}).get("clip_similarity"):
                    cat_clip.append(r1["semantic"]["clip_similarity"])
        
        avg_psnr = sum(cat_psnr) / len(cat_psnr) if cat_psnr else 0
        avg_ssim = sum(cat_ssim) / len(cat_ssim) if cat_ssim else 0
        avg_clip = sum(cat_clip) / len(cat_clip) if cat_clip else 0
        
        html += f"""
        <div class="category-section">
            <div class="category-header">
                <h3 style="margin: 0;">{category.replace('_', ' ').title()}</h3>
                <span class="category-badge">{successful}/{len(cat_evals)} successful</span>
            </div>
            
            <div class="chart-container" style="padding: 1rem;">
                <div class="bar-chart">
                    <div class="bar-row">
                        <span class="bar-label">PSNR</span>
                        <div class="bar-container">
                            <div class="bar psnr" style="width: {min(avg_psnr / 40 * 100, 100):.1f}%"></div>
                        </div>
                        <span class="bar-value">{avg_psnr:.1f}</span>
                    </div>
                    <div class="bar-row">
                        <span class="bar-label">SSIM</span>
                        <div class="bar-container">
                            <div class="bar ssim" style="width: {avg_ssim * 100:.1f}%"></div>
                        </div>
                        <span class="bar-value">{avg_ssim:.3f}</span>
                    </div>
                    <div class="bar-row">
                        <span class="bar-label">CLIP</span>
                        <div class="bar-container">
                            <div class="bar clip" style="width: {avg_clip * 100:.1f}%"></div>
                        </div>
                        <span class="bar-value">{avg_clip:.3f}</span>
                    </div>
                </div>
            </div>
        </div>
"""
    
    html += f"""
        <footer>
            <p>Generated by <a href="https://github.com/Qianli-ion/edit-converge-bench">ConvergeBench</a> 🔄</p>
            <p>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Save report
    report_path = output_dir / "report.html"
    with open(report_path, "w") as f:
        f.write(html)
    
    print(f"Report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate ConvergeBench HTML report")
    parser.add_argument("--results", type=Path, required=True,
                        help="Path to results JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Output directory (should contain evaluation subdirs)")
    
    args = parser.parse_args()
    generate_html_report(args.results, args.output_dir)


if __name__ == "__main__":
    main()
