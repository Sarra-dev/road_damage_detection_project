# Road Damage Detection (Deep Learning + MLOps)

YOLOv8 detector for road damage (D00, D10, D20, D40) with a reproducible MLOps workflow.

## Data
RDD2022, Czech Republic and India subsets, downloaded from the Hugging Face
mirror `dronefreak/RDD2022` (license CC BY-SA 4.0).
Please cite: Arya et al., "RDD2022: A multi-national image dataset for automatic
road damage detection", Geoscience Data Journal, 2023.

## Setup
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python src/download_data.py
    python src/validate_data.py