#!/usr/bin/env python3
"""Generate valid synthetic keyframe images for local dev and testing."""

import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def generate_keyframes():
    root = Path(__file__).resolve().parents[1]
    keyframes_dir = root / "tests" / "_ui_backend_fixtures" / "keyframes"
    index_frames = root / "tests" / "_ui_backend_fixtures" / "index" / "frames.ndjson"
    
    # Copy frames.ndjson
    shutil.copy(index_frames, keyframes_dir / "frames.ndjson")
    
    # Read all frame entries
    with open(index_frames, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Video colors for visually distinct thumbnails
    video_colors = {
        "L21_V001": (35, 60, 95),
        "L21_V002": (40, 85, 60),
        "L21_V003": (90, 45, 60),
    }

    images_dir = keyframes_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for r in records:
        vid = r["keyframe"]["video_id"]["value"]
        ord_num = r["keyframe"]["keyframe_ordinal"]
        frame_idx = r["frame_index"]
        pts = r["pts_time"]
        
        vid_dir = images_dir / vid
        vid_dir.mkdir(parents=True, exist_ok=True)
        img_path = vid_dir / f"{ord_num:06d}.jpg"
        
        # Create a 320x180 16:9 thumbnail
        bg_color = video_colors.get(vid, (45, 45, 55))
        img = Image.new("RGB", (320, 180), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw some gradient-like shapes
        draw.rectangle([10, 10, 310, 170], outline=(100, 140, 180), width=2)
        draw.rectangle([20, 20, 120, 80], fill=(bg_color[0] + 30, bg_color[1] + 30, bg_color[2] + 30))
        draw.ellipse([200, 40, 280, 120], fill=(bg_color[0] + 40, bg_color[1] + 20, bg_color[2] + 40))
        
        # Text annotations
        draw.text((25, 25), f"{vid}", fill=(255, 255, 255))
        draw.text((25, 45), f"Frame: {frame_idx}", fill=(220, 220, 220))
        draw.text((25, 60), f"PTS: {pts:.1f}s", fill=(180, 220, 255))
        draw.text((25, 140), f"Keyframe #{ord_num}", fill=(255, 200, 100))
        
        img.save(img_path, format="JPEG", quality=85)

    print(f"Successfully generated {len(records)} realistic keyframe thumbnails in {images_dir}")

if __name__ == "__main__":
    generate_keyframes()
