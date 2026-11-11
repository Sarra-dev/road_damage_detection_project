"""Image size statistics per country (reads only the file headers: fast).

Usage: python src/check_images.py
"""
from pathlib import Path

import pandas as pd
from PIL import Image

IMAGES = Path(__file__).resolve().parents[1] / "data" / "raw" / "rdd2022" / "data" / "images"

rows = []
for split in ["train", "valid", "test"]:
    for p in (IMAGES / split).rglob("*.jpg"):
        with Image.open(p) as im:
            w, h = im.size
        rows.append({"country": p.stem.rsplit("_", 1)[0], "w": w, "h": h,
                     "long_side": max(w, h), "kb": p.stat().st_size / 1024})

df = pd.DataFrame(rows)
summary = df.groupby("country").agg(
    images=("w", "size"),
    w_min=("w", "min"), w_median=("w", "median"), w_max=("w", "max"),
    h_median=("h", "median"),
    long_side_median=("long_side", "median"), long_side_max=("long_side", "max"),
    kb_mean=("kb", "mean"),
).round(0)
print(summary.to_string())

above = (df.long_side > 640).groupby(df.country).mean().mul(100).round(1)
print("\nShare of images whose longest side is above 640 px (these are downscaled at imgsz=640):")
print(above.to_string())
