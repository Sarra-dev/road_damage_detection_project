# Road Damage Detector

An automatic road damage detector based on deep learning, delivered through a reproducible MLOps workflow.
A YOLOv8 object detection model locates and classifies **cracks and potholes** in road images.

| Class | Description |
| :--- | :--- |
| `longitudinal_crack` | crack running along the road |
| `transverse_crack` | crack running across the road |
| `alligator_crack` | web-like network of cracks |
| `pothole` | hole in the road surface |

**Team:** Sarra-dev · ines-belkahla · weslatiwissem

---

## Project status

 **≈ 45%**

- ✅ **Data:** RDD2022 (Czech + India) downloaded, validated and versioned with DVC
- ✅ **Baseline:** YOLOv8s trained on a Kaggle T4 GPU: **mAP50 0.364 · mAP50-95 0.161** on the test split
- ✅ **Tracking:** MLflow logs params, training curves, per-class metrics and weights
- ✅ **Evaluation:** results per split, country and class, plus a run-comparison script
- ⏳ **Next:** FastAPI backend, React frontend, Docker, CI/CD (GitHub Actions), data-drift monitoring

---

## Project structure

```
road_damage_detection_project/
├── configs/
│   └── rdd.yaml              # dataset definition (paths + 4 class names)
├── data/
│   ├── raw/rdd2022.dvc       # DVC pointer to the raw dataset
│   └── processed/            # boxes.csv (output of the validate stage)
├── models/                   # best.pt (output of the train stage)
├── src/
│   ├── download_data.py      # download Czech + India from Hugging Face (with retries)
│   ├── validate_data.py      # label checks and dataset statistics
│   ├── check_images.py       # image resolution per country
│   ├── data_utils.py         # rare-class oversampling
│   ├── train.py              # YOLOv8 training + MLflow logging
│   ├── evaluate.py           # evaluation per split / country / class
│   └── compare_evals.py      # side-by-side comparison of runs
├── dvc.yaml / dvc.lock       # DVC pipeline (validate → train)
├── params.yaml               # quick-test config (CPU)
├── params.kaggle.yaml        # full baseline config (GPU)
├── params.exp1.yaml          # experiment: rare-class oversampling
├── params.exp2.yaml          # experiment: YOLOv8m
├── params.exp3.yaml          # experiment: longer training, cosine LR
├── metrics.json              # metrics of the last training run
└── requirements.txt
```

---

## Dataset

**RDD2022** (Road Damage Dataset 2022), Czech Republic and India subsets, downloaded from the Hugging Face mirror [`dronefreak/RDD2022`](https://huggingface.co/datasets/dronefreak/RDD2022) (license CC BY-SA 4.0).

| Split | Images | Share |
| :--- | ---: | ---: |
| Train | 7,330 | 70% |
| Validation | 1,603 | 15% |
| Test | 1,602 | 15% |
| **Total** | **10,535** | |

- Annotations are YOLO-format bounding boxes (one `.txt` label file per image).
- The country is read from the file name prefix (`Czech_…`, `India_…`).
- Transverse cracks are the rarest class (84 boxes in validation), and the Czech data has no potholes in validation or test.

> Arya et al., *RDD2022: A multi-national image dataset for automatic road damage detection*, Geoscience Data Journal, 2023.

---

## Installation

```bash
git clone https://github.com/Sarra-dev/road_damage_detection_project.git
cd road_damage_detection_project

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

### Get the data

```bash
dvc pull                          # dvc
# or
python src/download_data.py       # Hugging Face
```

### Check the data

```bash
python src/validate_data.py       # images per split/country, boxes per class -> data/processed/boxes.csv
python src/check_images.py        # image resolution per country
```

---

## How to run

### 1. Quick test (CPU, a few minutes)

Runs the whole pipeline on 5% of the data (YOLOv8n, 1 epoch) to check that everything works. The score is not meaningful.

```bash
dvc repro
```

### 2. Train the baseline (GPU, e.g. Kaggle T4, ≈ 1.3 h)

YOLOv8s, 50 epochs, 640 px, batch 16, seed 42.

```bash
python src/train.py params.kaggle.yaml
```

The best weights are saved to `models/best.pt` and the final metrics to `metrics.json`.

### 3. Run an experiment

Each experiment is defined by its own params file:

```bash
python src/train.py params.exp1.yaml
```

### 4. Test the baseline

```bash
# Evaluate models/best.pt per split, country and class -> eval_metrics.json, eval_table.csv
python src/evaluate.py params.kaggle.yaml

# Predict on one road image
yolo detect predict model=models/best.pt source=path/to/road_image.jpg imgsz=640

# Compare two runs (the first one is the reference)
python src/compare_evals.py baseline=path/to/baseline/eval_table.csv exp3=path/to/exp3/eval_table.csv --split test
```

### 5. View the runs in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.

---

## Results (test split)

| Run | Setup | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: |
| **Baseline** (kept) | YOLOv8s · 50 epochs · 640 px | **0.364** | **0.161** |
| Experiment 3 | YOLOv8m · 80 epochs + early stopping · cosine LR · transverse ×3 | 0.335 | 0.147 |

**mAP50 per class (baseline):** longitudinal 0.33 · transverse 0.16 · alligator 0.59 · pothole 0.37

A new run is kept only if overall mAP50-95 rises, transverse mAP50 rises, and no class loses more than 0.03 mAP50. The larger model did not beat the baseline, so the baseline is kept.

---

## Tech stack

- **Model:** YOLOv8 (Ultralytics), transfer learning from pre-trained weights
- **Data versioning:** DVC
- **Experiment tracking:** MLflow
- **Training:** Kaggle T4 GPU
- **Planned:** FastAPI, React, Docker, GitHub Actions
