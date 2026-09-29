# Teacher Mental Health Risk — FastAPI serving image
# Python 3.11 (pinned stack does not install cleanly on 3.14)
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_LOAD_REGISTRY=0 \
    PIP_NO_CACHE_DIR=1

# curl for HEALTHCHECK (tutorial-style); build tools for some native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Slimmer serve deps (full train/ops stack stays in requirements.txt)
COPY requirements-serve.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements-serve.txt

# Application code only — mount ./artifacts at runtime (compose)
COPY src/ ./src/

# Placeholder so path exists if volume is empty
RUN mkdir -p /app/artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
