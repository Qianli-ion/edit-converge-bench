# ConvergeBench

A benchmark for evaluating **iterative robustness** of image editing models.

## Motivation

Real users don't edit once—they iterate: *"make it greener"* → *"a bit more"* → *"now move it left."* Current benchmarks test single-shot edit quality, but not **quality degradation over iterations**.

**Core Thesis**: A good editing system enables **convergence**—users can iteratively steer toward their goal without accumulated artifacts or unintended side effects.

## Approach: Round-Trip Testing

We test models by performing round-trip edits:

```
I₀ → edit(forward) → I₁ → edit(backward) → I₁'
   → edit(forward) → I₂ → edit(backward) → I₂'
   → ... (N round-trips)
```

The original image `I₀` serves as ground truth. We measure how much `Iₙ'` diverges from `I₀` as `n` increases.

## Metrics

**Quality Preservation:**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

**Semantic Consistency:**
- CLIP similarity

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run evaluation on a model
python eval/run_roundtrip.py --model gemini --max-rounds 10

# Compute aggregate metrics
python eval/compute_metrics.py --results-dir results/

# Generate plots
python eval/plot_curves.py --results-dir results/
```

## Project Structure

```
convergebench/
├── data/
│   ├── images/              # Source images
│   └── edit_pairs.json      # (forward, backward) prompt pairs
├── models/
│   ├── base.py              # Abstract model interface
│   ├── gemini.py            # Gemini/Nano Banana wrapper
│   └── ...
├── metrics/
│   ├── quality.py           # PSNR, SSIM, LPIPS
│   └── semantic.py          # CLIP similarity
├── eval/
│   ├── run_roundtrip.py     # Core evaluation loop
│   ├── compute_metrics.py   # Aggregate results
│   └── plot_curves.py       # Visualization
└── results/                 # Output directory
```

## Models Evaluated

| Model | Provider | Status |
|-------|----------|--------|
| Gemini 2.5 Flash Image (Nano Banana) | Google | ✅ Implemented |
| FLUX Kontext | Black Forest Labs | 🔲 Planned |
| Qwen-Image-Edit | Alibaba | 🔲 Planned |
| GPT-Image-1 | OpenAI | 🔲 Planned |
| Seedream | ByteDance | 🔲 Planned |

## Citation

```bibtex
@misc{convergebench2026,
  title={ConvergeBench: Measuring Iterative Robustness of Image Editing Models},
  author={Feng, Qianli},
  year={2026}
}
```

## License

MIT
