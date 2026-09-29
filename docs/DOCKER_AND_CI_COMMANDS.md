# Docker & CI/CD — commands cheat sheet

## Environment (serving)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MLFLOW_REGISTERED_MODEL_NAME` | `teacher-mental-health-risk` | **Base** registry family name |
| `MLFLOW_MODEL_VARIANT` | *(empty)* | Optional pin. If empty, API **discovers** which `*-metaN` has `@champion` |
| `MLFLOW_MODEL_ALIAS` | `champion` | Only this alias is loaded (no `@latest`) |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | Tracking server |
| `MLFLOW_LOAD_REGISTRY` | `1` | `0` = local `artifacts/` only |

**Serving rule:** put `@champion` on **exactly one** of `teacher-mental-health-risk-meta1…meta4` (or `-rf`).  
API + frontend both use that model. Switch production by moving the alias in MLflow UI (or `.\scripts\set_mlflow_champion.ps1 -Variant meta2`).

---

## MLflow experiments in Docker (important)

Host training and Docker MLflow share **one** folder: `./mlflow_data/`  
(`mlflow_data/mlflow.db` + `mlflow_data/mlruns`).

1. Use only **one** MLflow on port 5000 (host script **or** compose, not both).
2. `docker compose up -d mlflow` (or full stack).
3. Train with `$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"`.
4. Open http://127.0.0.1:5000 — experiment runs + registered models appear.

The old separate `docker_data/mlflow` store is unused now (it was empty and looked like “no experiments”).

---

## Docker (full local stack)

```powershell
cd "C:\Users\USER\Desktop\Teachers Mental Health"

# Build + start: MLflow (:5000) + API (:8000) + Frontend (:3000)
docker compose up --build

# Detached
docker compose up --build -d

# Logs
docker compose logs -f api
docker compose logs -f frontend

# Stop
docker compose down
```

| Service | URL |
|---------|-----|
| UI | http://127.0.0.1:3000 |
| API docs | http://127.0.0.1:8000/docs |
| MLflow | http://127.0.0.1:5000 |

**Notes**
- Mount `./artifacts` into API — train on the host first (or copy joblibs in).
- To serve from MLflow champion in compose, set on `api` service:
  - `MLFLOW_LOAD_REGISTRY=1`
  - `MLFLOW_REGISTERED_MODEL_NAME=teacher-mental-health-risk`
  - `MLFLOW_MODEL_VARIANT=meta4`
  - `MLFLOW_MODEL_ALIAS=champion`
- Feast / Evidently jobs stay on the host (`scripts\schedule_*.ps1`).

Build API image only:

```powershell
docker build -t teacher-mh-api:local -f Dockerfile .
```

Build frontend image only:

```powershell
docker build -t teacher-mh-ui:local -f Dockerfile.frontend --build-arg VITE_API_BASE_URL=http://127.0.0.1:8000 .
```

---

## CI/CD (GitHub Actions — already in repo)

Workflow: `.github/workflows/ci.yml`

### What CI does
1. Python 3.11 → install `requirements-dev.txt`
2. flake8 (critical errors fail; style is warn)
3. `pytest -m "not slow and not integration"`
4. `docker build` API image (no push)
5. `docker compose config`

### Test CI locally (same spirit)

```powershell
cd "C:\Users\USER\Desktop\Teachers Mental Health"
.\venv\Scripts\Activate.ps1

# Fast tests (CI equivalent)
$env:MLFLOW_LOAD_REGISTRY = "0"
pytest tests/ -q -m "not slow and not integration" --tb=short

# Lint (critical)
flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics

# Docker build smoke
docker build -t teacher-mh-api:ci -f Dockerfile .
docker compose config
```

### Push to trigger real Actions

```powershell
git add -A
git status
git commit -m "ci: docker frontend + registry base/variant env"
git push
```

Then open GitHub → **Actions** tab.

---

## Host API with new env names

```powershell
$env:MLFLOW_LOAD_REGISTRY = "1"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_REGISTERED_MODEL_NAME = "teacher-mental-health-risk"
$env:MLFLOW_MODEL_VARIANT = "meta4"
$env:MLFLOW_MODEL_ALIAS = "champion"
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Switch stack without renaming the base:

```powershell
$env:MLFLOW_MODEL_VARIANT = "meta2"   # or meta1 / meta3 / rf
```
