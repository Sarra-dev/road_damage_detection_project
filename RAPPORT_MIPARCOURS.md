# 📊 Rapport d'Avancement de Mi-Parcours — Projet Deep Learning Avancé
> **Cycle Ingénieur en IA & Data Science — 3ème Année**  
> **Module :** Deep Learning 2 / Deep Learning Avancé  
> **Enseignant :** Haythem Ghazouani  
> **Année Universitaire :** 2026–2027  

---

## 📌 Section 1 : Fiche d'Identité et État Global du Projet

| Champ | Informations du Groupe |
| :--- | :--- |
| **Intitulé du Sujet** | Road Damage Detector: automatic detection and classification of road damage (cracks and potholes) in road images with YOLOv8, built as a reproducible MLOps workflow |
| **Track Métier** | Track 2 : Smart City - Industry |
| **Identifiant du Groupe** | WorkingBees — Road Damage Detector |
| **URL du Dépôt Git** | `https://github.com/Sarra-dev/road_damage_detection_project` |
| **Workspace MLflow / W&B** | Local MLflow server (SQLite backend `sqlite:///mlflow.db`, experiment `road-damage-detection`; can be redirected with `MLFLOW_TRACKING_URI`). DVC remote on DagsHub: `https://dagshub.com/Sarra-dev/road_damage_detection_project.dvc` |
| **Pourcentage d'Avancement Global** | **`≈ 45%`** *(Data, DVC versioning, baseline training, MLflow tracking and evaluation are done (3 of 6 components largely complete: ~90% + 100% + 80%). FastAPI, frontend and Docker/CI-CD have not been started yet (0%). Average of the 6 components ≈ 45%.)* |

### Membres de l'Équipe & Rôles
1. **Ben Bettaieb Sarra** (`srrbettaieb@gmail.com`) — *Role : Data pipeline & DVC, Deep Learning modelling & training, MLflow tracking, evaluation*
2. **Belkahla Ines** (`blkahlaines@gmail.com`) — *Role : Backend FastAPI & inference, Docker containerisation, data-drift monitoring*
3. **Ouelati Ouissem** (`wissemoueslati94@gmail.com`) — *Role : Frontend React, CI/CD (GitHub Actions + pytest), shared DVC remote*

---

## 🛠️ Section 2 : Travail Déjà Réalisé (Fait & Validé)

### 2.1. Données & Versioning (DVC)
* **Dataset source :** RDD2022 (Road Damage Dataset 2022), Czech Republic + India subsets, from the Hugging Face mirror `https://huggingface.co/datasets/dronefreak/RDD2022` (license CC BY-SA 4.0). Reference: Arya et al., *RDD2022: A multi-national image dataset for automatic road damage detection*, Geoscience Data Journal, 2023.
* **Volume :** 10,535 annotated images (7,330 train · 1,603 validation · 1,602 test), YOLO-format bounding boxes, 4 classes (longitudinal crack, transverse crack, alligator crack, pothole). ≈ 725 MB / 21,071 files tracked by DVC. Labelled boxes in validation: 370 longitudinal · 84 transverse · 322 alligator · 202 pothole. Native resolution: varies by country `[TO FILL: output of src/check_images.py (median width × height per country)]`.
* **Pipeline de prétraitement :**
  * `src/download_data.py`: downloads only the Czech and India files, with 10 automatic retries and resume.
  * `src/validate_data.py` (DVC `validate` stage): checks for images without a label file (none found), counts images per country/split and boxes per class/country, computes box-size statistics, and writes `data/processed/boxes.csv`.
  * Images resized to 640 px by YOLOv8 (`imgsz: 640`), with YOLOv8's built-in default augmentations (mosaic, HSV colour jitter, horizontal flip, scale/translate).
  * Rare-class oversampling (`src/data_utils.py`): train images containing a transverse crack are listed 3× (checked: 7,824 images listed = 7,824 scanned).
* **Preuve DVC :** Fichier `data/raw/rdd2022.dvc` créé et versionné sous Git, plus `dvc.yaml` (stages `validate` → `train`) and `dvc.lock`. Stockage distant configuré sur DagsHub `[TO FILL: push the DagsHub remote config — .dvc/config in the repo still points to a local folder]`.

### 2.2. Modélisation & Entraînement Deep Learning (Baseline)
* **Architecture testée :** YOLOv8s (Ultralytics), pre-trained on COCO and fine-tuned on the 4 road-damage classes (transfer learning). Settings: 50 epochs, 640 px, batch 16, seed 42. Experiment 3 also tested: YOLOv8m, 80 epochs with early stopping (patience 30), cosine LR schedule, transverse oversampling ×3.
* **Fonction de perte :** YOLOv8's default detection loss = CIoU loss + Distribution Focal Loss (box regression) + Binary Cross-Entropy (classification). It is the standard loss for the anchor-free YOLOv8 head and was kept unchanged for the baseline. The class imbalance is handled with oversampling instead of a loss change.
* **Environnement d'entraînement :** GPU Kaggle T4 (16 GB); ≈ 1.3 h for the baseline, ≈ 5.6 h for Experiment 3. A CPU quick-test mode (`params.yaml`: YOLOv8n, 1 epoch, 5% of the data) checks the whole pipeline locally.
* **Performances quantitatives obtenues (Validation Set) :**

| Métrique | Baseline Initiale | Objectif Final Visé |
| :--- | :---: | :---: |
| **mAP50** | `0.364` *(test split)* | `[TO FILL: team target, e.g. > 0.40]` |
| **mAP50-95** | `0.161` *(test split)* | `[TO FILL: team target, e.g. > 0.18]` |
| **Temps d'inférence (CPU)** | `[TO FILL: not measured yet]` | `[TO FILL]` |

*Values above are on the test split. Validation-split values: `[TO FILL: from eval_table.csv]`. Baseline mAP50 per class (test): longitudinal 0.33 · transverse 0.16 · alligator 0.59 · pothole 0.37. Experiment 3 (test): mAP50 0.335 · mAP50-95 0.147 → baseline kept, using a decision rule fixed in advance (keep a new run only if mAP50-95 rises, transverse mAP50 rises, and no class loses more than 0.03 mAP50).*

### 2.3. Traçabilité MLOps (MLflow / Weights & Biases)
* Nombre total d'expériences / runs enregistrés : **`[TO FILL: count in the MLflow UI; at least 3: quick test, baseline, experiment 3]`**
* Hyperparamètres tracés : `train.model`, `train.epochs`, `train.imgsz`, `train.batch`, `train.fraction`, `train.device`, `train.seed`, `train.workers`, `data.countries`, `sampling.*` (oversampling statistics), plus the per-experiment `train.extra` options (e.g. `patience`, `cos_lr`).
* Artefacts sauvegardés : checkpoint du meilleur modèle (`best.pt`), `metrics.json` (overall and per-class precision, recall, F1, mAP50, mAP50-95), per-epoch train/val curves (box, cls and DFL losses, precision, recall, mAP), Ultralytics plots (`results.png`, confusion matrix, PR curves) and `results.csv`. Evaluation per split, country and class (`src/evaluate.py` → `eval_metrics.json`, `eval_table.csv`); runs compared side by side with `src/compare_evals.py`.

### 2.4. Services Logiciels (FastAPI & Interface Utilisateur)
* **Backend FastAPI :**
  * `GET /health` : Not implemented yet.
  * `POST /predict` : Not implemented yet.
  * Validation des schémas d'entrée/sortie via Pydantic : Not included yet.
* **Frontend (React ou Angular) :**
  * React project not initialised yet.
  * Image upload component: not included yet.
  * Visualisation du résultat (bounding boxes drawn on the image) : Not ready.

---

## 📊 Tableau Synthétique d'Avancement par Composant

| Composant Technique | Statut (%) | Éléments Concrètement Prêts & Testés | Prochaine Action Immédiate (Semaine +1) |
| :--- | :---: | :--- | :--- |
| **1. Dataset & DVC** | `[90 %]` | 10,535 images downloaded and validated (no image without a label file), 70/15/15 split, `rdd2022.dvc` + `validate` stage | Push the DagsHub remote config so all members can `dvc pull` |
| **2. Modèle DL (Baseline)** | `[100 %]` | YOLOv8s trained for 50 epochs on Kaggle T4: mAP50 0.364 / mAP50-95 0.161 (test); Experiment 3 (YOLOv8m) compared and rejected | 960 px resolution experiment + error analysis with the confusion matrix |
| **3. Tracking MLOps** | `[80 %]` | MLflow runs with params, per-epoch curves, per-class metrics and weights; evaluation per split/country/class | Add the `evaluate` stage to `dvc.yaml`; register the best model in MLflow |
| **4. API FastAPI** | `[0 %]` | — | Create `/health` and `/predict` with the baseline model |
| **5. Frontend UI** | `[0 %]` | — | Initialise the React app with an image-upload page |
| **6. Docker & CI/CD** | `[0 %]` | — | Write the API Dockerfile; first pytest tests for the data helpers |

---

## ⚠️ Section 3 : Difficultés Rencontrées, Verrous & Solutions Envisagées

| N° | Verrou / Difficulté Rencontrée | Impact sur le Projet | Solution Technique Retenue |
| :---: | :--- | :--- | :--- |
| **1** | Class imbalance: transverse cracks are rare (84 boxes in validation vs 202–370 for the other classes) | Lowest score of all classes (transverse mAP50 0.16 on test) | Oversampling ×3 of the train images that contain a transverse crack: tested and checked, no clear gain. Next lever: higher input resolution (960 px) |
| **2** | No local GPU; long training runs (≈ 1.3 h baseline, ≈ 5.6 h for Experiment 3) | Full training impossible on our laptops; slow iterations | Kaggle T4 notebook driven by one params file per experiment, plus a CPU quick-test mode that runs the whole chain on 5% of the data first |
| **3** | Dataset too large for Git (≈ 725 MB, 21,071 files) | Data cannot be shared or versioned with the code | DVC for data versioning, with DagsHub as remote storage |
| **4** | Unstable Hugging Face downloads | Interrupted downloads of the dataset | Download script with 10 automatic retries that resumes where it stopped, filtered to the Czech + India files only |

---

## 🎯 Section 4 : Travail Restant à Accomplir (Ce qu'il reste à faire)

1. **Modélisation Avancée :**
   * [ ] Train and compare a higher-resolution model (960 px) against the baseline, using the same decision rule.
   * [ ] Error analysis with the confusion matrix and per-country results; repeat the best setting with several seeds.
2. **Ingénierie MLOps & Production :**
   * [ ] FastAPI backend (`/health`, `/predict`) with Pydantic schemas, and a React interface (upload a photo → boxes drawn).
   * [ ] Conteneurisation complète multi-conteneurs via `docker-compose.yml` *(API + Frontend + MLflow)*.
   * [ ] Complete the DVC pipeline (`evaluate` stage) and register the best model in MLflow.
   * [ ] Automatisation des tests unitaires *(pytest on the data helpers and the API endpoints)*.
   * [ ] Configuration du pipeline GitHub Actions *(lint + unit tests)*.
3. **Module de Surveillance (Monitoring) :**
   * [ ] Data-drift / out-of-distribution check on incoming images (e.g. Kolmogorov-Smirnov test on image statistics, or Evidently AI).
4. **Documentation & Validation Finale :**
   * [ ] Rédiger le `README.md` exhaustif avec instructions de lancement en une seule commande (`docker compose up`).
   * [ ] Répétition générale de la démonstration technique en direct.

---

## 📅 Section 5 : Planning Prévisionnel Ajusté & Répartition des Tâches

### Planning Gantt Hebdomadaire (Jusqu'à la Soutenance Finale)

| Tâche / Sous-Système | Semaine +1 | Semaine +2 | Semaine +3 | Semaine +4 (Livraison) | Responsable Attitré |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Modèle Avancé & Tuning** | 🟩🟩🟩 | 🟩 | | | Sarra Ben Bettaieb |
| **Finalisation API FastAPI** | 🟩🟩 | 🟩 | | | Ines Belkahla |
| **Développement Frontend** | 🟩🟩 | 🟩🟩 | 🟩 | | Ouissem Ouelati |
| **Docker & Docker-Compose** | | 🟩🟩 | 🟩 | | Ines Belkahla |
| **CI/CD GitHub Actions** | | 🟩 | 🟩🟩 | | Ouissem Ouelati |
| **Module Data Drift / Monitoring** | | | 🟩🟩 | | Ines Belkahla |
| **Tests d'Intégration & Répétition** | | | 🟩🟩🟩 | | Whole team |

*Three weeks remain before the final delivery, so the Semaine +4 column is empty.*

### Matrice RACI (Responsabilités au sein du Groupe)
* **R (Responsible) :** Réalise la tâche.
* **A (Accountable) :** Valide et garantit la qualité finale.
* **C (Consulted) :** Apporte son aide / expertise.
* **I (Informed) :** Tenu au courant de l'avancement.

| Tâche | Sarra Ben Bettaieb | Ines Belkahla | Ouissem Ouelati |
| :--- | :---: | :---: | :---: |
| Pipeline Données & DVC | **R / A** | **I** | **C** |
| Entraînement DL & Tracking MLflow | **R / A** | **I** | **I** |
| API FastAPI & Inférence | **C** | **R / A** | **C** |
| Interface Utilisateur (React/Angular) | **I** | **C** | **R / A** |
| Conteneurisation Docker & CI/CD | **C** | **R** | **A** |
| Tests & Démonstration Live | **R** | **R** | **R / A** |

---

## 📷 Annexes : Captures d'Écran et Preuves d'Exécution
*(Insérez ci-dessous 2 ou 3 captures réelles de votre travail actuel)*

1. **Capture d'écran du Tracking MLflow :**
   `[TO FILL: screenshot of the MLflow run list and the train/val loss curves of the baseline run]`
   
2. **Capture d'écran de l'inférence visuelle (Baseline) :**
   `[TO FILL: one test image showing the original, the ground-truth boxes and the baseline prediction, e.g. from Ultralytics' val_batch*_labels.jpg / val_batch*_pred.jpg]`

3. **Capture d'écran de l'API / Swagger (`/docs`) ou de l'Interface UI :**
   Not available at mid-term: the FastAPI backend and the UI have not been started yet (planned for week +1).
