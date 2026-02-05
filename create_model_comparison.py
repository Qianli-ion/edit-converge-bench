#!/usr/bin/env python3
"""Create a comparison grid showing model outputs after round 10."""

from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "results_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

# Models to compare (with their result directories)
MODELS = [
    ("Original", None),  # Special case - use source image
    ("FLUX", "results_flux_10rounds"),
    ("Gemini", "results_gemini_10rounds"),
    ("Grok", "results_grok_10rounds"),
    ("Qwen", "results_qwen_10rounds"),
]

# Representative experiments to show (that exist in most/all models)
EXPERIMENTS = [
    "portrait_01_glasses_add_remove",
    "portrait_01_hat_add_remove",
    "portrait_02_glasses_add_remove",
    "portrait_02_hat_add_remove",
    "portrait_03_glasses_add_remove",
    "portrait_03_hat_add_remove",
]

# Grid settings
THUMB_SIZE = 200
PADDING = 8
HEADER_HEIGHT = 35
ROW_LABEL_WIDTH = 180
FONT_SIZE = 16

def get_font(size=FONT_SIZE):
    """Try to get a decent font."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                pass
    return ImageFont.load_default()

def create_model_comparison():
    """Create comparison grid: rows=experiments, cols=models."""
    n_cols = len(MODELS)
    n_rows = len(EXPERIMENTS)
    
    # Calculate dimensions
    grid_width = ROW_LABEL_WIDTH + n_cols * (THUMB_SIZE + PADDING) + PADDING
    grid_height = HEADER_HEIGHT + n_rows * (THUMB_SIZE + PADDING) + PADDING
    
    # Create canvas
    canvas = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(canvas)
    font = get_font()
    small_font = get_font(12)
    
    # Draw column headers (model names)
    for col_idx, (model_name, _) in enumerate(MODELS):
        x = ROW_LABEL_WIDTH + col_idx * (THUMB_SIZE + PADDING) + PADDING + THUMB_SIZE // 2
        y = HEADER_HEIGHT // 2
        draw.text((x, y), model_name, fill='black', font=font, anchor='mm')
    
    # Draw rows
    for row_idx, exp_name in enumerate(EXPERIMENTS):
        y_base = HEADER_HEIGHT + row_idx * (THUMB_SIZE + PADDING) + PADDING
        
        # Row label
        short_name = exp_name.replace("_add_remove", "").replace("_", " ")
        label_y = y_base + THUMB_SIZE // 2
        draw.text((PADDING, label_y), short_name, fill='black', font=small_font, anchor='lm')
        
        # Images for each model
        for col_idx, (model_name, result_dir) in enumerate(MODELS):
            x = ROW_LABEL_WIDTH + col_idx * (THUMB_SIZE + PADDING) + PADDING
            
            if result_dir is None:
                # Original image - get from any model's source
                for _, rd in MODELS[1:]:
                    img_path = BASE_DIR / rd / exp_name / "I0_source.png"
                    if img_path.exists():
                        break
            else:
                # Round 10 result
                img_path = BASE_DIR / result_dir / exp_name / "I10_backward.png"
            
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
                    offset_x = (THUMB_SIZE - img.width) // 2
                    offset_y = (THUMB_SIZE - img.height) // 2
                    canvas.paste(img, (x + offset_x, y_base + offset_y))
                except Exception as e:
                    draw.rectangle([x, y_base, x + THUMB_SIZE, y_base + THUMB_SIZE], 
                                  outline='red', width=2)
            else:
                draw.rectangle([x, y_base, x + THUMB_SIZE, y_base + THUMB_SIZE], 
                              outline='gray', width=1)
                draw.text((x + THUMB_SIZE//2, y_base + THUMB_SIZE//2), 
                         "N/A", fill='gray', font=small_font, anchor='mm')
    
    # Save
    output_path = OUTPUT_DIR / "model_comparison_round10.png"
    canvas.save(output_path, quality=95)
    print(f"Saved: {output_path}")
    return output_path

if __name__ == "__main__":
    create_model_comparison()
