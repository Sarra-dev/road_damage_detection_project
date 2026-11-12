"""Evaluate models/best.pt per split, per country and overall.

Usage:  python src/evaluate.py [params_file]
Writes: eval_metrics.json (nested metrics) and eval_table.csv (one row per split/group/class).
The country of an image is the prefix of its file name (Czech_000123.jpg -> Czech).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
DATA = ROOT / "data" / "raw" / "rdd2022" / "data"

params_file = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "params.yaml"
params = yaml.safe_load(params_file.read_text())
tp = params["train"]
ev = params.get("evaluate") or {}
countries = params["data"]["countries"]
splits = ev.get("splits", ["valid", "test"])
device = ev.get("device", tp.get("device", "cpu"))
imgsz = ev.get("imgsz", tp["imgsz"])
batch = ev.get("batch", 16)
workers = ev.get("workers", tp.get("workers", 0))
weights = ROOT / ev.get("weights", "models/best.pt")

lists_dir = ROOT / "runs" / "eval_lists"
lists_dir.mkdir(parents=True, exist_ok=True)


def country_of(img: Path) -> str:
    return img.stem.rsplit("_", 1)[0]


def label_path(img: Path, split: str) -> Path:
    return DATA / "labels" / split / img.relative_to(DATA / "images" / split).with_suffix(".txt")


def count_boxes(imgs, split):
    counts = dict.fromkeys(NAMES, 0)
    for img in imgs:
        lbl = label_path(img, split)
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                if line.strip():
                    counts[NAMES[int(line.split()[0])]] += 1
    return counts


model = YOLO(str(weights))
results, rows = {}, []

for split in splits:
    all_imgs = sorted((DATA / "images" / split).rglob("*.jpg"))
    groups = {"all": all_imgs}
    for c in countries:
        groups[c] = [p for p in all_imgs if country_of(p) == c]

    for group, imgs in groups.items():
        if not imgs:
            continue
        list_file = lists_dir / f"{split}_{group}.txt"
        list_file.write_text("\n".join(p.as_posix() for p in imgs))
        data_yaml = lists_dir / f"{split}_{group}.yaml"
        data_yaml.write_text(yaml.safe_dump({
            "path": DATA.as_posix(),
            "train": list_file.as_posix(),
            "val": list_file.as_posix(),
            "names": dict(enumerate(NAMES)),
        }))

        print(f"\n=== {split} / {group}: {len(imgs)} images ===")
        m = model.val(
            data=str(data_yaml), split="val", imgsz=imgsz, batch=batch,
            device=device, workers=workers, plots=False, verbose=False,
            project=str(ROOT / "runs" / "eval"), name=f"{split}_{group}", exist_ok=True,
        )
        box = m.box
        boxes = count_boxes(imgs, split)

        entry = {
            "images": len(imgs),
            "boxes": sum(boxes.values()),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
        }
        per_class = {}
        for i, c in enumerate(box.ap_class_index):
            p, r, ap50, ap = box.class_result(i)
            name = NAMES[int(c)]
            per_class[name] = {
                "boxes": boxes[name],
                "precision": float(p),
                "recall": float(r),
                "f1": float(2 * p * r / (p + r + 1e-9)),
                "mAP50": float(ap50),
                "mAP50-95": float(ap),
            }
        for name in NAMES:  # classes with no box in this group: only the count
            per_class.setdefault(name, {"boxes": boxes[name]})
        entry["per_class"] = per_class
        results[f"{split}/{group}"] = entry

        rows.append({"split": split, "group": group, "class": "ALL", "images": len(imgs),
                     "boxes": entry["boxes"], "precision": entry["precision"],
                     "recall": entry["recall"], "f1": None, "mAP50": entry["mAP50"],
                     "mAP50-95": entry["mAP50-95"]})
        for name in NAMES:
            pc = per_class[name]
            rows.append({"split": split, "group": group, "class": name, "images": len(imgs),
                         "boxes": pc["boxes"], "precision": pc.get("precision"),
                         "recall": pc.get("recall"), "f1": pc.get("f1"),
                         "mAP50": pc.get("mAP50"), "mAP50-95": pc.get("mAP50-95")})

(ROOT / "eval_metrics.json").write_text(json.dumps(results, indent=2))
table = pd.DataFrame(rows)
table.to_csv(ROOT / "eval_table.csv", index=False)
print("\n", table.round(3).to_string(index=False))
