"""Helpers for the training data (used by train.py)."""
from pathlib import Path


def build_train_list(data_root: Path, names, oversample: dict, out_file: Path) -> dict:
    """Write a text file listing the train images, repeating those that contain a rare class.

    oversample = {"transverse_crack": 3} means every train image that contains at least one
    transverse crack appears 3 times in total (the original + 2 extra copies).
    """
    img_dir = data_root / "images" / "train"
    lbl_dir = data_root / "labels" / "train"
    name_to_id = {n: i for i, n in enumerate(names)}

    unknown = set(oversample) - set(name_to_id)
    if unknown:
        raise ValueError(f"Unknown class(es) in oversample: {sorted(unknown)}. Valid: {list(names)}")

    entries, repeated = [], 0
    images = sorted(img_dir.rglob("*.jpg"))
    for img in images:
        lbl = lbl_dir / img.relative_to(img_dir).with_suffix(".txt")
        copies = 1
        if lbl.exists():
            ids = {int(line.split()[0]) for line in lbl.read_text().splitlines() if line.strip()}
            for cls_name, k in oversample.items():
                if name_to_id[cls_name] in ids:
                    copies = max(copies, int(k))
        if copies > 1:
            repeated += 1
        entries += [img.as_posix()] * copies

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(entries))
    return {"unique_images": len(images), "listed_images": len(entries), "repeated_images": repeated}
