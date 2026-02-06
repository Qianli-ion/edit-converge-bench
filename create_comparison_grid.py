#!/usr/bin/env python3
"""Create a comparison grid of round-trip editing results across all models."""

from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "results_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

# Models with their directories
MODELS = [
    ("FLUX", "results_flux_10rounds"),
    ("Gemini", "results_gemini_10rounds"),
    ("Grok", "results_grok_10rounds"),
    ("Qwen", "results_qwen_10rounds"),
    ("Nano Banana", "results_nano-banana-edit_10rounds"),
    ("Seedream", "results_seedream_10rounds"),
]

# Columns: Original, Round 1, 2, 5, 10
COLUMNS = [
    ("Original", "I0_source.png"),
    ("Round 1", "I1_backward.png"),
    ("Round 2", "I2_backward.png"),
    ("Round 5", "I5_backward.png"),
    ("Round 10", "I10_backward.png"),
]

# Experiments to include (portraits + scenes)
EXPERIMENTS = [
    "portrait_01_glasses_add_remove",
    "portrait_01_hat_add_remove",
    "portrait_02_hat_add_remove",
    "portrait_03_glasses_add_remove",
    "scene_01_chair_add_remove",
    "scene_01_red_ball_add_remove",
    "scene_02_chair_add_remove",
    "scene_03_chair_add_remove",
]

# Grid settings
THUMB_SIZE = 180
PADDING = 6
HEADER_HEIGHT = 30
ROW_LABEL_WIDTH = 140
FONT_SIZE = 14

def get_font(size=FONT_SIZE):
    """Try to get a decent font, fall back to default."""
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

def create_comparison_grid():
    """Create the full comparison grid."""
    n_cols = len(COLUMNS)
    n_rows = len(EXPERIMENTS) * len(MODELS)  # Each experiment × each model
    
    # Calculate dimensions
    grid_width = ROW_LABEL_WIDTH + n_cols * (THUMB_SIZE + PADDING) + PADDING
    grid_height = HEADER_HEIGHT + n_rows * (THUMB_SIZE + PADDING) + PADDING
    
    # Create canvas
    canvas = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(canvas)
    font = get_font()
    small_font = get_font(11)
    
    # Draw column headers
    for col_idx, (col_name, _) in enumerate(COLUMNS):
        x = ROW_LABEL_WIDTH + col_idx * (THUMB_SIZE + PADDING) + PADDING + THUMB_SIZE // 2
        y = HEADER_HEIGHT // 2
        draw.text((x, y), col_name, fill='black', font=font, anchor='mm')
    
    # Draw rows
    row_idx = 0
    for exp_idx, exp_name in enumerate(EXPERIMENTS):
        for model_idx, (model_name, model_dir) in enumerate(MODELS):
            y_base = HEADER_HEIGHT + row_idx * (THUMB_SIZE + PADDING) + PADDING
            
            # Row label
            short_exp = exp_name.replace("_add_remove", "").replace("_", " ")
            label = f"{short_exp}\n({model_name})"
            label_y = y_base + THUMB_SIZE // 2
            draw.text((PADDING, label_y), label, fill='black', font=small_font, anchor='lm')
            
            # Images for each column
            for col_idx, (_, filename) in enumerate(COLUMNS):
                img_path = BASE_DIR / model_dir / exp_name / filename
                x = ROW_LABEL_WIDTH + col_idx * (THUMB_SIZE + PADDING) + PADDING
                
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
                        # Center the thumbnail
                        offset_x = (THUMB_SIZE - img.width) // 2
                        offset_y = (THUMB_SIZE - img.height) // 2
                        canvas.paste(img, (x + offset_x, y_base + offset_y))
                    except Exception as e:
                        draw.rectangle([x, y_base, x + THUMB_SIZE, y_base + THUMB_SIZE], 
                                      outline='red', width=2)
                        draw.text((x + THUMB_SIZE//2, y_base + THUMB_SIZE//2), 
                                 "Error", fill='red', font=small_font, anchor='mm')
                else:
                    draw.rectangle([x, y_base, x + THUMB_SIZE, y_base + THUMB_SIZE], 
                                  outline='gray', width=1)
                    draw.text((x + THUMB_SIZE//2, y_base + THUMB_SIZE//2), 
                             "N/A", fill='gray', font=small_font, anchor='mm')
            
            row_idx += 1
    
    # Save
    output_path = OUTPUT_DIR / "comparison_grid_full.png"
    canvas.save(output_path, quality=95)
    print(f"Saved: {output_path}")
    return output_path

if __name__ == "__main__":
    create_comparison_grid()
