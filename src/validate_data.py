from pathlib import Path
import pandas as pd

ROOT = Path("data/raw/rdd2022/data")
NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]

rows, missing = [], []
for split in ["train", "valid", "test"]:
    img_dir = ROOT / "images" / split
    for img in img_dir.rglob("*.jpg"):
        lbl = ROOT / "labels" / split / img.relative_to(img_dir).with_suffix(".txt")
        country = img.stem.rsplit("_", 1)[0]
        if not lbl.exists():
            missing.append(str(img))
            continue
        boxes = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        if not boxes:
            rows.append(dict(split=split, country=country, img=img.name, cls=None, w=None, h=None))
        for b in boxes:
            rows.append(dict(split=split, country=country, img=img.name,
                             cls=int(b[0]), w=float(b[3]), h=float(b[4])))

df = pd.DataFrame(rows)
print("Images without a label file:", len(missing))

print("\nImages per country and split")
print(df.groupby(["country", "split"])["img"].nunique().unstack(fill_value=0))

boxes = df.dropna(subset=["cls"]).copy()
boxes["cls"] = boxes["cls"].astype(int).map(dict(enumerate(NAMES)))
print("\nBoxes per class and country")
print(boxes.groupby(["country", "cls"]).size().unstack(fill_value=0))

empty = df[df.cls.isna()].groupby("country")["img"].nunique()
total = df.groupby("country")["img"].nunique()
print("\nShare of images with no damage")
print((empty / total).round(2))

print("\nBox size (normalized width/height)")
print(boxes.groupby("country")[["w", "h"]].describe().round(3).T)

Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv("data/processed/boxes.csv", index=False)