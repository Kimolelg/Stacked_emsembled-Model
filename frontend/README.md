# Teacher Mental Health — Frontend (React + Vite)

Talks to the FastAPI backend (`VITE_API_BASE_URL`, default `http://127.0.0.1:8000`).

## Pages

- **Screening** — questionnaire or Feast `teacher_id` → `POST /predict`
- **Monitoring** — drift status, alerts, latest Evidently HTML report

## Run (dev)

```powershell
# Terminal A — API (venv activated, project root)
$env:MLFLOW_LOAD_REGISTRY="0"
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# Terminal B — UI
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (usually http://127.0.0.1:5173).

## Run (Docker)

From project root:

```powershell
docker compose up --build
```

UI: http://127.0.0.1:3000 · API: http://127.0.0.1:8000

## Build

```powershell
cd frontend
npm run build
npm run preview
```

**Monitoring:** the Evidently HTML report is **not** auto-loaded when you open the page — click **Load latest report**.
