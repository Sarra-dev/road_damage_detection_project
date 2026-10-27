import json
import os
import re
import shutil
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from ultralytics import YOLO, settings

settings.update({"mlflow": False})  # we do the MLflow logging ourselves

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]

# Optional argument: python src/train.py params.kaggle.yaml
params_file = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "params.yaml"
params = yaml.safe_load(params_file.read_text())
tp = params["train"]
countries = params["data"]["countries"]
workers = tp.get("workers", 0)

(ROOT / "runs").mkdir(exist_ok=True)
data_yaml = ROOT / "runs" / "dataset.yaml"
data_yaml.write_text(yaml.safe_dump({
    "path": (ROOT / "data" / "raw" / "rdd2022" / "data").as_posix(),
    "train": "images/train",
    "val": "images/valid",
    "test": "images/test",
    "names": dict(enumerate(NAMES)),
}))

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("road-damage-detection")


def clean(name: str) -> str:
    """MLflow metric names cannot contain parentheses."""
    return re.sub(r"[^\w\-\. /]", "", name.replace("(B)", "").strip())


run_name = f"{Path(tp['model']).stem}_{'-'.join(countries)}"
with mlflow.start_run(run_name=run_name):
    mlflow.log_params({f"train.{k}": v for k, v in tp.items()})
    mlflow.log_param("data.countries", ",".join(countries))

    model = YOLO(tp["model"])
    results = model.train(
        data=str(data_yaml),
        epochs=tp["epochs"],
        imgsz=tp["imgsz"],
        batch=tp["batch"],
        fraction=tp["fraction"],
        device=tp["device"],
        seed=tp["seed"],
        workers=workers,
        project=str(ROOT / "runs" / "train"),
        name="exp",
        exist_ok=True,
    )
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"

    # Per-epoch curves -> MLflow
    res = pd.read_csv(save_dir / "results.csv")
    res.columns = [c.strip() for c in res.columns]
    for _, row in res.iterrows():
        for col in res.columns:
            if col != "epoch" and pd.notna(row[col]):
                mlflow.log_metric(clean(col), float(row[col]), step=int(row["epoch"]))

    # Final metrics on the validation split
    metrics = results
    if metrics is None or not hasattr(metrics, "box"):
        metrics = YOLO(str(best)).val(data=str(data_yaml), split="val",
                                      device=tp["device"], workers=workers, plots=False)
    box = metrics.box
    summary = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class[metrics.names[int(c)]] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(2 * p * r / (p + r + 1e-9)),
            "mAP50": float(ap50),
            "mAP50-95": float(ap),
        }
    summary["per_class"] = per_class
    (ROOT / "metrics.json").write_text(json.dumps(summary, indent=2))

    mlflow.log_metrics({k: v for k, v in summary.items() if k != "per_class"})
    for cls_name, m in per_class.items():
        for k, v in m.items():
            mlflow.log_metric(f"{cls_name}/{k}", v)

    # Artifacts: weights, curves, plots
    mlflow.log_artifact(str(best))
    mlflow.log_artifact(str(ROOT / "metrics.json"))
    for f in list(save_dir.glob("*.png")) + list(save_dir.glob("*.csv")):
        mlflow.log_artifact(str(f), artifact_path="train_run")

    # Copy the model to the tracked location (DVC output)
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "models" / "best.pt")

print("Done. Metrics:", json.dumps(summary, indent=2))