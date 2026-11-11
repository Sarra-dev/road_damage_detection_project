"""Compare several eval_table.csv files side by side.

Usage: python src/compare_evals.py baseline=kaggle_run1/eval_table.csv exp1=kaggle_exp1/eval_table.csv [--split valid]
The first run is the reference: columns d_<name> are the difference to it.
"""
import sys

import pandas as pd

args = sys.argv[1:]
split = "test"
if "--split" in args:
    i = args.index("--split")
    split = args[i + 1]
    del args[i:i + 2]
if len(args) < 2:
    sys.exit("Give at least two runs: name=path/to/eval_table.csv")

runs = dict(a.split("=", 1) for a in args)
names = list(runs)

for metric in ["mAP50", "mAP50-95"]:
    table = pd.DataFrame()
    for name, path in runs.items():
        df = pd.read_csv(path)
        df = df[df["split"] == split].set_index(["group", "class"])
        if "boxes" not in table.columns:
            table["boxes"] = df["boxes"]
        table[name] = df[metric]
    for n in names[1:]:
        table[f"d_{n}"] = table[n] - table[names[0]]
    print(f"\n=== {metric} on the {split} split (boxes = labelled boxes in that group) ===")
    print(table.round(3).to_string())
