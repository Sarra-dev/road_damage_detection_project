"""Log a finished Kaggle run into MLflow, from the folder extracted out of results_<name>.zip.

Usage (from the project root):
  python src/log_kaggle_run.py kaggle_runs/baseline --name baseline_yolov8s --tag selected=v1
  python src/log_kaggle_run.py kaggle_runs/exp3_combo --name exp3_combo

Where the run goes: the MLFLOW_TRACKING_URI environment variable, otherwise the local sqlite:///mlflow.db.
For DagsHub set  MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow  together with
MLFLOW_TRACKING_USERNAME (your DagsHub user) and MLFLOW_TRACKING_PASSWORD (your DagsHub token).

Options:  --with-model  also upload best.pt as an MLflow artifact
          --again       log the run again even if a run with the same name already exists
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("--name", required=True)
parser.add_argument("--tag", action="append", default=[], help="key=value, can be repeated")
parser.add_argument("--with-model", action="store_true")
parser.add_argument("--again", action="store_true")
args = parser.parse_args()

folder = args.folder
if not (folder / "params.run.yaml").exists() or not (folder / "metrics.json").exists():
    sys.exit(f"{folder} does not look like an extracted results zip (params.run.yaml or metrics.json missing).")

run_info = {}
if (folder / "run_info.txt").exists():
    for line in (folder / "run_info.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            run_info[k.strip().replace(" ", "_")] = v.strip()
if run_info.get("quick_test", "").lower() == "true":
    sys.exit("This zip comes from a QUICK_TEST run (2 epochs): not logging it.")


def clean(name: str) -> str:
    """MLflow metric names cannot contain parentheses."""
    return re.sub(r"[^\w\-\. /]", "", name.replace("(B)", "").strip())


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(map(str, v))
        else:
            out[key] = v
    return out


def ok(x):
    return isinstance(x, (int, float)) and not math.isnan(x)


mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
experiment = mlflow.set_experiment("road-damage-detection")
client = MlflowClient()

if not args.again:
    existing = client.search_runs([experiment.experiment_id],
                                  filter_string=f"tags.mlflow.runName = '{args.name}'")
    if existing:
        sys.exit(f"A run named '{args.name}' already exists ({existing[0].info.run_id}). Use --again to log it anyway.")

params = yaml.safe_load((folder / "params.run.yaml").read_text())
n_metrics = 0

with mlflow.start_run(run_name=args.name) as run:
    mlflow.log_params({k: v for k, v in flatten(params).items()})
    tags = {"source": "kaggle", **run_info, **dict(t.split("=", 1) for t in args.tag)}
    mlflow.set_tags(tags)

    # per-epoch curves
    results_csv = folder / "train_run" / "results.csv"
    if results_csv.exists():
        res = pd.read_csv(results_csv)
        res.columns = [c.strip() for c in res.columns]
        for _, row in res.iterrows():
            for col in res.columns:
                if col != "epoch" and pd.notna(row[col]):
                    mlflow.log_metric(clean(col), float(row[col]), step=int(row["epoch"]))
                    n_metrics += 1

    # final validation metrics (same names as train.py uses)
    final = json.loads((folder / "metrics.json").read_text())
    mlflow.log_metrics({k: v for k, v in final.items() if k != "per_class" and ok(v)})
    for cls, m in final.get("per_class", {}).items():
        mlflow.log_metrics({f"{cls}/{k}": v for k, v in m.items() if ok(v)})

    # evaluation per split / country / class
    eval_json = folder / "eval_metrics.json"
    if eval_json.exists():
        for key, entry in json.loads(eval_json.read_text()).items():     # key like "test/Czech"
            for k in ["images", "boxes", "precision", "recall", "mAP50", "mAP50-95"]:
                if ok(entry.get(k)):
                    mlflow.log_metric(f"eval/{key}/{k}", entry[k])
                    n_metrics += 1
            for cls, m in entry.get("per_class", {}).items():
                for k, v in m.items():
                    if ok(v):
                        mlflow.log_metric(f"eval/{key}/{cls}/{k}", v)
                        n_metrics += 1

    # files
    try:
        for f in ["metrics.json", "eval_metrics.json", "eval_table.csv", "params.run.yaml", "run_info.txt", "train_log.txt"]:
            if (folder / f).exists():
                mlflow.log_artifact(str(folder / f))
        for f in list((folder / "train_run").glob("*.png")) + [results_csv, folder / "train_run" / "args.yaml"]:
            if Path(f).exists():
                mlflow.log_artifact(str(f), artifact_path="train_run")
        if args.with_model and (folder / "best.pt").exists():
            mlflow.log_artifact(str(folder / "best.pt"), artifact_path="model")
    except Exception as e:  # noqa: BLE001
        print("Warning: some files could not be uploaded:", e)

    print(f"Logged run '{args.name}' ({run.info.run_id}) to {mlflow.get_tracking_uri()}")
    print(f"  {len(flatten(params))} parameters, {len(tags)} tags, per-epoch + final + evaluation metrics")
