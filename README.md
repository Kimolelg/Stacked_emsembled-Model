# Teacher Mental Health Risk — End-to-End MLOps Platform

**Project:** Local production-style ML platform for **Kenyan TSC teacher depression risk screening**  
**Target:** `High_Risk` = PHQ-9 total ≥ 15 (moderate-to-severe symptom range)  
**Role:** Early-warning / institutional screening support — **not** clinical diagnosis  
**Stack:** Experiment tracking → feature store → data validation → monitoring → serving (local end-to-end MLOps)

**Audience of this document:** Next ML engineer who must understand, operate, extend, and hand over this system in one read.

---

## Table of contents

1. [What this system does](#1-what-this-system-does)
2. [How the platform was built (iterative story)](#2-how-the-platform-was-built-iterative-story)
3. [Architecture at a glance](#3-architecture-at-a-glance)
4. [Full project map (every folder & file)](#4-full-project-map-every-folder--file)
5. [Layer 0 — Data processing](#5-layer-0--data-processing)
6. [Layer 1 — Meta-models 1–4](#6-layer-1--meta-models-14)
7. [Layer 2 — Training CLI](#7-layer-2--training-cli)
8. [Layer 3 — MLflow Tracking + Registry](#8-layer-3--mlflow-tracking--registry)
9. [Layer 4 — FastAPI serving](#9-layer-4--fastapi-serving)
10. [Layer 5 — Feast feature store](#10-layer-5--feast-feature-store)
11. [Layer 6 — Great Expectations validation](#11-layer-6--great-expectations-validation)
12. [Layer 7 — Evidently monitoring](#12-layer-7--evidently-monitoring)
13. [How the layers complement each other](#13-how-the-layers-complement-each-other)
14. [Environment setup](#14-environment-setup)
15. [Day-1 runbook (full pipeline)](#15-day-1-runbook-full-pipeline)
16. [API reference](#16-api-reference)
17. [Artifacts, data stores & registry names](#17-artifacts-data-stores--registry-names)
18. [Tests](#18-tests)
19. [Ethics & prohibited uses](#19-ethics--prohibited-uses)
20. [Troubleshooting](#20-troubleshooting)
21. [Handover checklist for the next engineer](#21-handover-checklist-for-the-next-engineer)

---

## 1. What this system does

Given a teacher survey (or a compact questionnaire / known `teacher_id`), the system:

1. **Engineers** a fixed-width feature vector (~222 columns after cleaning, domain means, OHE, leakage drop).
2. **Trains** four equal meta-model stacks (plus optional RF) and logs each run to MLflow.
3. **Registers** models as peers in the Model Registry (`@latest` only from code).
4. **Serves** risk probability + binary flag via FastAPI, with ethical disclaimers on every response.
5. **Stores features** in Feast so train and serve share one contract.
6. **Validates** inputs (runtime gatekeeper) and batch tables (Great Expectations).
7. **Monitors** data + prediction drift with Evidently against a training reference.

Promotion to production is **manual**: set the `@champion` alias in the MLflow UI after comparing runs. No model is auto-crowned in code.

---

## 2. How the platform was built (iterative story)

This is the order the system was assembled. Read top → bottom to understand *why* files exist.

| Step | What we built | Why | Key paths |
|------|---------------|-----|-----------|
| **0** | Notebook → production processing | Stop leakage, fix Fin_D4, Sub_County OHE, multicollinearity; one pipeline for train + serve | `src/data/processing.py` |
| **1** | Meta-Models 1–4 as first-class peers | All stacks from `mental_health_final_model.py`, no preferred baseline in code | `src/models/meta_models.py` |
| **2** | Training CLI + local artifacts | Reproducible train, joblib packages, comparison JSON | `src/models/train.py`, `artifacts/` |
| **3** | MLflow Tracking + Registry | Params, metrics, versions, UI promotion | `src/utils/mlflow_tracking.py`, `scripts/start_mlflow_server.ps1` |
| **4** | FastAPI + feature builder | Compact questionnaire → full feature vector; health, predict, model-info | `src/api/*` |
| **5** | Feast offline + online | Kill train/serve skew via shared feature views | `feature_repo/`, `src/data/feast_*` |
| **6** | Great Expectations + gatekeeper | Garbage-in → error-out before model | `src/data/data_validation.py` |
| **7** | Evidently drift + prediction log | Reference (train) vs current (prod traffic) | `src/monitoring/*` |

Each layer **does not replace** the previous one — it wraps or consumes it. Processing remains the single source of feature engineering truth; Feast is the *serving contract* over already-engineered features; MLflow is the *experiment and promotion* system; the API is the *live entry point* that wires validation, Feast, model load, and monitoring together.

---

## 3. Architecture at a glance

```
                    ┌─────────────────────────────────────┐
                    │  TSC survey Excel (raw)              │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │  processing.py                       │
                    │  clean → engineer → High_Risk target │
                    └─────────────┬───────────┬───────────┘
                                  │           │
              train path          │           │  feast path
                                  │           │
         ┌────────────────────────▼┐   ┌──────▼──────────────────┐
         │ train.py + meta_models  │   │ feast_pipeline          │
         │ Meta 1–4 equal train    │   │ parquet + schema.json   │
         └────────────┬────────────┘   └──────┬──────────────────┘
                      │                       │
         ┌────────────▼────────────┐   ┌──────▼──────────────────┐
         │ MLflow Tracking         │   │ Feast apply + materialize│
         │ Registry @latest only   │   │ offline + online SQLite  │
         └────────────┬────────────┘   └──────┬──────────────────┘
                      │                       │
                      │    ┌──────────────────┘
                      │    │
         ┌────────────▼────▼────────────────────────────────────┐
         │  FastAPI (src/api/main.py)                            │
         │  load: @champion → max version → local artifacts      │
         │  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
         │  │ GE gatekeeper│  │ feature_    │  │ feast_client │  │
         │  │ + batch GE   │  │ builder     │  │ teacher_id   │  │
         │  └──────────────┘  └─────────────┘  └──────────────┘  │
         │  predict → prediction_logger → data/monitoring/       │
         └────────────────────────┬─────────────────────────────┘
                                  │
         ┌────────────────────────▼─────────────────────────────┐
         │  Evidently drift_monitor                              │
         │  reference = Feast/train  |  current = prod log       │
         │  HTML report + last_drift_summary.json                │
         └──────────────────────────────────────────────────────┘
```

**Decision policy**

| Concern | Who owns it |
|---------|-------------|
| Feature engineering logic | `processing.py` only |
| Model definitions | `meta_models.py` |
| Experiment compare + register | MLflow |
| Which model is “live” | Human in MLflow UI (`@champion`) |
| Feature serving contract | Feast (`feature_repo` + parquet) |
| Bad request rejection | `data_validation` on `/predict` |
| Drift detection | Evidently + production log |

---

## 4. Full project map (every folder & file)

```
Teachers Mental Health/
│
├── README.md                          # This handover document
├── requirements.txt                   # Pinned stack (Python 3.11)
├── Dockerfile                         # Container entry for API (basic)
├── .gitignore
│
├── src/                               # All application code
│   ├── __init__.py
│   │
│   ├── data/                          # Data plane
│   │   ├── processing.py              # Clean, engineer, target, prepare_for_modeling
│   │   ├── feast_pipeline.py          # Raw → Feast parquet + apply + materialize
│   │   ├── feast_client.py            # Online get + offline training matrix
│   │   └── data_validation.py         # Runtime gatekeeper + GE batch suite
│   │
│   ├── models/                        # Model plane
│   │   ├── meta_models.py             # Meta 1–4 defs, metrics, registry names
│   │   ├── meta_model4.py             # Thin compat shim for old pickles/imports
│   │   └── train.py                   # CLI: train meta1|2|3|4|rf|all + MLflow
│   │
│   ├── api/                           # Serving plane
│   │   ├── main.py                    # FastAPI app, routes, model load, hooks
│   │   ├── schemas.py                 # Pydantic request/response models
│   │   └── feature_builder.py         # Questionnaire → full feature_order vector
│   │
│   ├── monitoring/                    # Observability plane
│   │   ├── prediction_logger.py       # Append prod predictions for drift
│   │   └── drift_monitor.py           # Evidently (+ KS fallback)
│   │
│   └── utils/
│       └── mlflow_tracking.py         # Configure, log, register, load by alias
│
├── feature_repo/                      # Feast repository (apply from here)
│   ├── feature_store.yaml             # Local provider, file offline, sqlite online
│   ├── features.py                    # Entity + FeatureViews (domain + survey)
│   └── __init__.py
│
├── scripts/                           # Operator entry points (PowerShell / bash)
│   ├── start_mlflow_server.ps1        # MLflow UI + registry backend
│   ├── start_mlflow_server.sh
│   ├── feast_materialize.ps1          # Build Feast table + materialize
│   ├── run_full_pipeline.ps1          # Train (+ register) after MLflow is up
│   └── run_drift_check.ps1            # Evidently reference vs production
│
├── artifacts/                         # Local trained packages (gitignored typically)
│   ├── meta1.joblib … meta4.joblib
│   ├── meta_model4.joblib             # Alias path for older loaders
│   ├── feature_order.joblib           # Column order contract (~222)
│   ├── cal_*.joblib, final_xgb.joblib # Meta4 component dumps
│   ├── model_metadata_meta*.json      # Per-model train metadata
│   └── meta_models_comparison.json    # Side-by-side metrics when --model all
│
├── data/
│   ├── feast/                         # Feast offline + online stores
│   │   ├── teacher_survey_features.parquet
│   │   ├── teacher_labels.parquet
│   │   ├── feature_schema.json
│   │   ├── registry.db
│   │   └── online_store.db
│   └── monitoring/                    # Evidently I/O
│       ├── production_predictions.parquet / .csv
│       ├── reference_snapshot.parquet
│       ├── last_drift_summary.json
│       └── reports/*.html
│
├── mlflow.db                          # SQLite backend for tracking + registry
├── mlruns/                            # MLflow artifacts (runs + registered models)
│
├── notebooks/
│   └── meta_model4_baseline.ipynb     # Early baseline exploration (historical)
│
├── tests/                             # pytest suite
│   ├── test_processing.py
│   ├── test_meta_model4.py
│   ├── test_meta_models_123.py
│   ├── test_mlflow_tracking.py
│   ├── test_feature_builder.py
│   ├── test_feast_pipeline.py
│   ├── test_data_validation.py
│   └── test_monitoring.py
│
└── docs/
    └── ethical_guidelines.md          # Mandatory use / non-use policy
```

### What each top-level area is for

| Path | Complements | Role |
|------|-------------|------|
| `src/data` | Training, Feast, API | Truth for cleaning & validation |
| `src/models` | MLflow, artifacts | Train equal stacks |
| `src/utils` | train.py, API load | MLflow glue |
| `src/api` | All layers | Live inference + ops endpoints |
| `src/monitoring` | API, Feast reference | Drift after go-live |
| `feature_repo` | feast_pipeline | Feast definitions only |
| `scripts` | Operators | One-command workflows |
| `artifacts` | Offline serve / debug | Local models if MLflow down |
| `data/feast` | Train `--use-feast`, `/predict` teacher_id | Feature store files |
| `data/monitoring` | Evidently | Drift inputs/outputs |
| `mlflow.db` + `mlruns` | Train, promote, API | Experiment + registry |

---

## 5. Layer 0 — Data processing

**File:** `src/data/processing.py`

Faithful production port of the validated notebook pipeline (`mental_health_final_model.py`). Same **order of operations** as development so metrics stay comparable.

### Pipeline stages

1. **`load_and_clean_raw_data(path)`** — read Excel, standardize columns, coerce types.
2. **`engineer_features_and_target(df)`**
   - Likert maps (Strongly Disagree…Strongly Agree, Never…Always).
   - Domain means: workload, learners, WLB, performance, time, emotional, financial, social.
   - PHQ total + **`High_Risk = (PHQ_Total >= 15)`**.
   - Drop multicollinear columns (`MULTICOLLINEAR_DROP`).
   - Drop **label leakage** clinical/self-report diagnosis columns (`LEAKAGE_COLS`) — critical for *early screening* (we must not use known diagnosis to predict risk).
   - OHE demographics (`OHE_COLS`); binary yes/no fields handled separately (`Fin_D4` forced numeric).
3. **`prepare_for_modeling(df)`** — return `X`, `y`, `feature_order` (the column contract every downstream layer uses).

### Why this layer matters to everyone else

- **train.py** calls it for non-Feast training.
- **feast_pipeline** runs the same engineering before writing parquet (so Feast never invents features).
- **feature_builder** expands questionnaire scores into the **same** `feature_order` names.
- Changing feature logic here without re-materializing Feast and re-training **breaks** serve parity.

---

## 6. Layer 1 — Meta-models 1–4

**Files:**  
- `src/models/meta_models.py` — definitions, `evaluate_binary_model`, registry map  
- `src/models/meta_model4.py` — compatibility shim (old `from meta_model4 import …` / pickles)

### Equal peers (no production favorite in code)

| Key | Registry name | Stack idea |
|-----|---------------|------------|
| `meta1` | `teacher-mental-health-risk-meta1` | Calibrated RF + XGB + CatBoost → XGB stack (BorderlineSMOTE prepared) |
| `meta2` | `teacher-mental-health-risk-meta2` | RF + XGB + Cat + LGBM → XGB (BorderlineSMOTE) |
| `meta3` | `teacher-mental-health-risk-meta3` | Cal RF/XGB/Cat + cal LGBM → XGB (SMOTETomek + threshold opt via `ThresholdedModel`) |
| `meta4` | `teacher-mental-health-risk-meta4` | Cal RF/XGB/Cat → XGB meta with **PHQ sample weights** (custom, not pure sklearn Stacking) |
| `rf` | `teacher-mental-health-risk-rf` | Tuned RandomForest + BorderlineSMOTE (optional baseline) |

### Shared evaluation (logged to MLflow)

Via `evaluate_binary_model` / `metrics_for_mlflow`:

- accuracy, precision, recall, F1 (binary + macro where applicable)
- ROC-AUC, average precision (PR-AUC)
- classification report text as artifact/log

### ThresholdedModel

Meta3 may learn a decision threshold ≠ 0.5. The wrapper keeps `predict_proba` and applies the threshold in `predict`.

---

## 7. Layer 2 — Training CLI

**File:** `src/models/train.py`

### Responsibilities

1. Load features either from **raw Excel** (`processing`) or **Feast offline** (`--use-feast`).
2. Stratified train/test split.
3. Train the chosen model(s).
4. Write **local** joblib + metadata under `artifacts/`.
5. Log params/metrics/artifacts to MLflow and **register `@latest` only** (never auto-set `@champion`).

### CLI

```powershell
# Single model from Excel
python src/models/train.py --data_path "C:\path\to\survey.xlsx" --model meta4

# All meta models (comparison JSON + equal registry entries)
python src/models/train.py --data_path "C:\path\to\survey.xlsx" --model all

# Train from Feast offline table (after feast materialize)
python src/models/train.py --use-feast --model meta4

# Skip MLflow (local artifacts only)
python src/models/train.py --data_path survey.xlsx --model meta2 --no-mlflow
```

| Flag | Meaning |
|------|---------|
| `--data_path` | Required unless `--use-feast` |
| `--model` | `meta1` \| `meta2` \| `meta3` \| `meta4` \| `rf` \| `all` |
| `--use-feast` | `get_training_matrix` from Feast/parquet |
| `--no-mlflow` | Local only |
| `--registered-model-name` | Override registry name for one-off runs |
| `--output_dir` | Default `artifacts` |

### Outputs per model

- `artifacts/meta{N}.joblib` — full inference package
- `artifacts/feature_order.joblib` — shared column list
- `artifacts/model_metadata_meta{N}.json`
- `artifacts/meta_models_comparison.json` when `--model all`
- Meta4 also writes `meta_model4.joblib` + default `model_metadata.*` for older paths

---

## 8. Layer 3 — MLflow Tracking + Registry

| Piece | Path / command |
|-------|----------------|
| Helpers | `src/utils/mlflow_tracking.py` |
| Start server | `.\scripts\start_mlflow_server.ps1` |
| Backend | `sqlite:///mlflow.db` |
| Artifacts | `./mlruns` |
| UI | http://127.0.0.1:5000 |

### What training logs

- **Params:** model key, random_state, test_size, resampling notes, key hyperparams
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, …
- **Artifacts:** joblib package, feature_order, metadata JSON, comparison JSON
- **Tags:** model family, project name

### Registration policy (important)

- Code sets **`@latest`** (or registers a new version; never force-promotes production).
- **`@champion` / `@challenger`** are assigned **only in the MLflow UI** after human review.
- API load order (when registry enabled): **`@champion` → highest version → local artifacts**.

### Configure client

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME = "teacher-mental-health-risk"
```

If the HTTP server is down, `configure_mlflow` can fall back to file store under `./mlruns` so local training still works.

### Schema upgrade note

After upgrading MLflow packages against an older `mlflow.db`, you may need:

```powershell
# backup first
Copy-Item mlflow.db "mlflow.db.bak_$(Get-Date -Format yyyyMMdd_HHmmss)"
mlflow db upgrade sqlite:///mlflow.db
```

---

## 9. Layer 4 — FastAPI serving

**Files:**

| File | Role |
|------|------|
| `src/api/main.py` | App lifecycle, model load, all routes |
| `src/api/schemas.py` | Request/response Pydantic models + disclaimers |
| `src/api/feature_builder.py` | Expand ~15 questionnaire fields → full `feature_order` |

### Model load strategy

1. If `MLFLOW_LOAD_REGISTRY=1` and server reachable: try alias (`MLFLOW_MODEL_ALIAS`, default `champion`).
2. Else load highest registry version for `MLFLOW_REGISTERED_MODEL_NAME`.
3. Else load local `artifacts/meta_model4.joblib` (or meta*.joblib) + `feature_order.joblib`.

### Two prediction paths (same endpoint family)

| Path | Input | Feature source |
|------|--------|----------------|
| **Questionnaire** | Domain Likert averages + demographics | `feature_builder.expand_questionnaire_to_features` fills defaults for missing cols |
| **Feast `teacher_id`** | `{ "teacher_id": 1 }` | Online (then offline) Feast vector; questionnaire ignored |

This is intentional: live screening for new teachers uses the questionnaire; batch re-score of known survey respondents uses Feast IDs.

### Start API

```powershell
$env:MLFLOW_LOAD_REGISTRY = "1"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_REGISTERED_MODEL_NAME = "teacher-mental-health-risk-meta4"
$env:MLFLOW_MODEL_ALIAS = "champion"

uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
# Swagger: http://127.0.0.1:8000/docs
```

Without registry:

```powershell
# unset registry load; uses artifacts/
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Swagger `additionalProp1` note

Pydantic/OpenAPI sometimes shows free-form dict props as `additionalProp*`. Prefer the **examples** on each field in `schemas.py` and the concrete payloads in [§16](#16-api-reference).

---

## 10. Layer 5 — Feast feature store

Train/serve feature consistency via a shared feature store.

| File | Role |
|------|------|
| `feature_repo/feature_store.yaml` | Local provider; file offline; SQLite online |
| `feature_repo/features.py` | Entity `teacher` (`teacher_id`); domain FV + survey FV |
| `src/data/feast_pipeline.py` | Build parquet/schema; `feast apply`; materialize |
| `src/data/feast_client.py` | Historical matrix for train; online vectors for API |
| `scripts/feast_materialize.ps1` | Operator wrapper |

### Design choice

Feature **engineering still lives in `processing.py`**. Feast stores and serves the *already engineered* matrix so:

- Offline historical features for training (`--use-feast`)
- Online features for `POST /predict` with `teacher_id`
- Schema baked from `feature_schema.json` (names match `feature_order.joblib`)

### Feature views

1. **`teacher_domain_fv`** — stable domain means + key binaries (good for monitoring).
2. **`teacher_survey_fv`** — full wide vector for model inference.

### Build & materialize

```powershell
.\scripts\feast_materialize.ps1 -DataPath "C:\path\to\survey.xlsx"
# or
python -m src.data.feast_pipeline --data_path "C:\path\to\survey.xlsx"
```

Produces under `data/feast/`:

- `teacher_survey_features.parquet` — offline wide table
- `feature_schema.json` — column contract for FV fields
- `registry.db`, `online_store.db`

Then train with shared contract:

```powershell
python src/models/train.py --use-feast --model all
```

### Feast CLI note (Windows)

Use `feast.exe` / `python -m src.data.feast_pipeline` (subprocess to Feast CLI). Plain `python -m feast` may fail if package has no `__main__`.

---

## 11. Layer 6 — Great Expectations validation

**File:** `src/data/data_validation.py`

### Two tiers

| Tier | When | Mechanism |
|------|------|-----------|
| **Runtime gatekeeper** | Every `/predict` (and `/validate/questionnaire`) | Fast custom Python rules — Likert 1–5, demographic fragments, teacher_id int, no NaN explosion |
| **Batch auditor** | Feast parquet / training matrices (`/validate/feast`) | Great Expectations suite on columns, null rates, ranges |

Philosophy: **garbage in → HTTP 400 out** before the model runs. This protects both model quality and ethical misuse of nonsense inputs.

### API hooks

- Failures on predict return structured validation errors (not 500).
- `POST /validate/questionnaire` — dry-run validation without scoring.
- `GET /validate/feast` — batch check on materialised Feast table.

### Dependency

`great-expectations==0.18.19` in `requirements.txt`. First GE import can be slow; tests cover happy path without full GE cloud.

---

## 12. Layer 7 — Evidently monitoring

Data + prediction drift monitoring.

| File | Role |
|------|------|
| `src/monitoring/prediction_logger.py` | Append each production prediction (domain features + proba + class) |
| `src/monitoring/drift_monitor.py` | Build reference from Feast/train; run Evidently; KS fallback |
| `scripts/run_drift_check.ps1` | CLI drift job |
| `data/monitoring/` | Logs, reference snapshot, HTML reports |

### Flow

1. API predictions → append to `production_predictions.parquet` (and CSV).
2. Build **reference** from Feast offline domain features (training world).
3. Run **Evidently** DataDrift / prediction drift vs production log.
4. Write `reports/latest_drift_report.html` + `last_drift_summary.json`.

### Drift columns (stable domain set)

`workload_mean`, `learners_mean`, `wlb_mean`, `perf_mean`, `time_mean`, `emot_mean`, `fin_mean`, `soc_mean`, `tech_awareness`, plus `prediction_proba` / `prediction`.

### Operator commands

```powershell
.\scripts\run_drift_check.ps1 -BuildReference
.\scripts\run_drift_check.ps1 -MinRows 30
```

Or via API: `/monitoring/build-reference`, `/monitoring/run-drift`, `/monitoring/status`, `/monitoring/report`.

---

## 13. How the layers complement each other

Think of the platform as a **pipeline of contracts**, not a pile of tools:

```
processing  ──contracts──►  feature_order + High_Risk definition
     │
     ├──► meta_models + train  ──metrics──►  MLflow (compare equals)
     │         │
     │         └──► artifacts/*.joblib  ──fallback──►  API
     │
     ├──► feast_pipeline  ──same X──►  Feast offline/online  ──teacher_id──►  API
     │
     └──► GE / gatekeeper  ──valid I/O──►  API  ──log──►  Evidently
```

| If you change… | You must also… |
|----------------|----------------|
| `processing.py` feature logic | Re-run feast pipeline, retrain all models, refresh `feature_order` |
| Meta-model code | Retrain + register new version; re-promote `@champion` if desired |
| Registry champion | Restart or reload API env pointing at that alias |
| Feast schema | Re-materialize; ensure API feature_order still aligns |
| Validation rules | Update tests + document new 400 cases for product owners |
| Drift feature set | Rebuild reference snapshot |

**Train–serve skew is the #1 failure mode.** Feast + shared `feature_order.joblib` + feature_builder defaults exist specifically to prevent silent column drift.

---

## 14. Environment setup

### Python version

**Use Python 3.11.** The pinned stack (`xgboost`, `catboost`, `feast`, `mlflow`, …) does not reliably install on 3.14.

```powershell
cd "C:\Users\USER\Desktop\Teachers Mental Health"

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python --version   # 3.11.x
```

### Key dependency pins (see `requirements.txt`)

| Area | Packages |
|------|----------|
| ML | pandas, sklearn, imbalanced-learn, xgboost, lightgbm, catboost |
| Serving | fastapi, uvicorn, pydantic |
| Tracking | mlflow 2.17.2 |
| Features | feast 0.40.1, pyarrow 16.1.0 |
| Validation | great-expectations 0.18.19 |
| Monitoring | evidently 0.4.33 |

`pyarrow` is pinned for Feast/Dask offline store compatibility with MLflow 2.17.

---

## 15. Day-1 runbook (full pipeline)

Assume venv activated and survey Excel path known.

### Terminal A — MLflow

```powershell
.\scripts\start_mlflow_server.ps1
# UI: http://127.0.0.1:5000
```

### Terminal B — Feast + train + promote + serve

```powershell
# 1) Feature store
.\scripts\feast_materialize.ps1 -DataPath "C:\path\to\TSC_survey.xlsx"

# 2) Train all meta models + register @latest
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
.\scripts\run_full_pipeline.ps1 -DataPath "C:\path\to\TSC_survey.xlsx" -Model all
# or Feast-backed:
# python src/models/train.py --use-feast --model all

# 3) In MLflow UI → Models → pick version → Add Alias: champion
#    (e.g. on teacher-mental-health-risk-meta4 or whichever wins comparison)

# 4) API
$env:MLFLOW_LOAD_REGISTRY = "1"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_REGISTERED_MODEL_NAME = "teacher-mental-health-risk-meta4"
$env:MLFLOW_MODEL_ALIAS = "champion"
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# 5) Smoke
# GET  http://127.0.0.1:8000/health
# GET  http://127.0.0.1:8000/model-info
# POST http://127.0.0.1:8000/predict  (see §16)

# 6) Drift baseline
.\scripts\run_drift_check.ps1 -BuildReference
```

---

## 16. API reference

Base URL: `http://127.0.0.1:8000` — interactive docs at `/docs`.

### Health & model

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Root / service name |
| GET | `/health` | Liveness + model loaded flag |
| GET | `/model-info` | Loaded model source, registry, feature count |
| GET | `/features/schema` | Expected feature names / questionnaire fields |

### Prediction

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/predict` | Compact questionnaire or `teacher_id` |
| POST | `/predict/full` | Partial engineered dict + defaults |

**Example — questionnaire**

```json
{
  "age_category": "25-34",
  "gender": "Female",
  "years_in_service": "5-10 years",
  "school_type": "Primary School",
  "education_qualification": "Bachelor's Degree",
  "ict_skills": "Basic",
  "workload": 4.0,
  "learners": 3.5,
  "work_life_balance": 4.0,
  "performance": 3.0,
  "time_pressure": 4.0,
  "emotional_impact": 3.5,
  "financial": 4.0,
  "social_support": 3.0,
  "tech_awareness": true
}
```

**Example — Feast entity**

```json
{ "teacher_id": 1 }
```

**Typical response fields**

- `prediction` (0/1), `probability` / risk score  
- `risk_label` (e.g. High/Low)  
- Ethical disclaimer string (always present)  
- Optional model version / source metadata  

Invalid questionnaire → **400** with validation details (not 500).

### Feast

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/feast/teachers?limit=20` | Sample teacher_ids present in store |

### Validation

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/validate/questionnaire` | Validate body without predicting |
| GET | `/validate/feast` | Batch GE-style audit of Feast parquet |

### Monitoring

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/monitoring/status` | Log row counts, reference present, last summary |
| POST | `/monitoring/build-reference` | Snapshot reference from Feast/train |
| POST | `/monitoring/run-drift` | Run Evidently (optional min rows) |
| GET | `/monitoring/report` | Latest HTML report path / content info |

---

## 17. Artifacts, data stores & registry names

### Local artifacts (`artifacts/`)

| File | Use |
|------|-----|
| `meta1.joblib` … `meta4.joblib` | Full stacks |
| `meta_model4.joblib` | Legacy alias for meta4 |
| `feature_order.joblib` | **Must match** serve matrix columns |
| `model_metadata_meta*.json` | Metrics, MLflow run ids, notes |
| `meta_models_comparison.json` | Cross-model table from `--model all` |

### Feast (`data/feast/`)

| File | Use |
|------|-----|
| `teacher_survey_features.parquet` | Offline training / reference |
| `feature_schema.json` | FV field definitions |
| `online_store.db` | Online serving for `teacher_id` |
| `registry.db` | Feast registry |

### MLflow registered models

- `teacher-mental-health-risk-meta1`
- `teacher-mental-health-risk-meta2`
- `teacher-mental-health-risk-meta3`
- `teacher-mental-health-risk-meta4`
- `teacher-mental-health-risk-rf` (if trained)

Aliases: `@latest` (code), `@champion` / `@challenger` (UI only).

### Monitoring (`data/monitoring/`)

| File | Use |
|------|-----|
| `production_predictions.*` | Live traffic log |
| `reference_snapshot.parquet` | Train-world baseline |
| `last_drift_summary.json` | Machine-readable last run |
| `reports/*.html` | Human-readable Evidently report |

---

## 18. Tests

```powershell
pytest tests/ -q
```

| Test module | Covers |
|-------------|--------|
| `test_processing.py` | Cleaning, target, leakage drop, feature matrix |
| `test_meta_model4.py` / `test_meta_models_123.py` | Stack training on tiny synthetic data |
| `test_mlflow_tracking.py` | Logging helpers (file store) |
| `test_feature_builder.py` | Questionnaire expansion + defaults |
| `test_feast_pipeline.py` | Table build shape / schema |
| `test_data_validation.py` | Gatekeeper accept/reject |
| `test_monitoring.py` | Logger + drift helpers |

Full end-to-end train on the real Excel is **slow** (CatBoost/XGB stacks). Use unit tests for CI; reserve full train for release candidates.

---

## 19. Ethics & prohibited uses

See full policy: [`docs/ethical_guidelines.md`](docs/ethical_guidelines.md).

**Mandatory framing on every product surface:**

> This is an AI screening support tool only. It is **not** a medical diagnosis. A high-risk result should prompt professional support, not automated HR action.

| Allowed (institutional) | Prohibited |
|-------------------------|------------|
| Voluntary wellness outreach | Individual discipline / firing |
| Counseling resource planning | Insurance discrimination |
| Aggregated needs assessment | Public labeling of teachers |
| Research with consent | Fully automated decisions without human review |

Human oversight is **required** for any high-risk flag.

---

## 20. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `pip` hangs / fails on 3.14 | Wrong Python | Recreate venv with **3.11** |
| API **503** model not loaded | No artifacts + registry down / no champion | Train once; or set alias; or load local artifacts |
| API **503** feature_order missing | Never trained / path wrong | Run train; ensure `artifacts/feature_order.joblib` |
| MLflow schema / alembic errors | DB older than package | Backup + `mlflow db upgrade sqlite:///mlflow.db` |
| `@latest` reserved alias errors | Older code used reserved name wrong | Use version resolution helpers in `mlflow_tracking.py` |
| Feast apply fails | Not from repo / pyarrow mismatch | Run `feast_pipeline`; pin pyarrow 16.1 + mlflow 2.17.2 |
| `teacher_id` predict empty | Not materialized | Run `feast_materialize.ps1` |
| Predict **400** | Gatekeeper rejection | Check Likert 1–5 and demographic strings |
| Drift “not enough rows” | Sparse prod log | Generate traffic; lower `min_current_rows` carefully |
| Old MetaModel4 unpickle | Class path moved | Use `meta_model4.py` shim + retrain preferred |
| GE / dask noisy stderr on Windows | Import side effects | Usually harmless if suite returns; check tests |

---

## 21. Checklist To Run on your Own

**First hour**

- [ ] Read this README + `docs/ethical_guidelines.md`
- [ ] Create Python 3.11 venv, `pip install -r requirements.txt`
- [ ] Start MLflow; open UI; list registered models and aliases
- [ ] Confirm `artifacts/feature_order.joblib` and at least one `meta*.joblib`
- [ ] `pytest tests/ -q`

**First day**

- [ ] Trace one questionnaire request through: validate → feature_builder → model → logger
- [ ] Trace one `teacher_id` request through: feast_client → model
- [ ] Run or review a full train (`--model meta4` is enough for smoke)
- [ ] Open latest Evidently HTML under `data/monitoring/reports/`
- [ ] Confirm who is authorized to set `@champion` in your org

**Before changing production**

- [ ] Never auto-promote in code without product sign-off
- [ ] Re-materialize Feast if processing changes
- [ ] Retrain **all** peers if you want a fair comparison
- [ ] Rebuild drift reference after major feature shifts
- [ ] Keep disclaimers on API responses and any UI

**Out of current scope (future work)**

- Full CI/CD promotion gates
- Hardened Docker multi-service compose (MLflow + API + Feast)
- SHAP explainability endpoint for case-level review
- AuthN/AuthZ on API and PII handling beyond anonymized `teacher_id`

---

## Quick command cheat sheet

```powershell
# Env
.\venv\Scripts\Activate.ps1

# MLflow
.\scripts\start_mlflow_server.ps1

# Feast
.\scripts\feast_materialize.ps1 -DataPath "SURVEY.xlsx"

# Train
python src/models/train.py --data_path "SURVEY.xlsx" --model all
python src/models/train.py --use-feast --model meta4

# API
$env:MLFLOW_LOAD_REGISTRY="1"
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:MLFLOW_REGISTERED_MODEL_NAME="teacher-mental-health-risk-meta4"
$env:MLFLOW_MODEL_ALIAS="champion"
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# Drift
.\scripts\run_drift_check.ps1 -BuildReference

# Tests
pytest tests/ -q
```

---

*Built as a local, complete MLOps learning + delivery stack for Kenyan teacher mental health screening support. Technology must serve people with humility, privacy, and human oversight.*
