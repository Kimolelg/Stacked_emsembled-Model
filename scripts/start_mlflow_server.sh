#!/usr/bin/env bash
# Start local MLflow Tracking Server + Model Registry backend
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p mlruns

echo "Starting MLflow server..."
echo "  tracking UI : http://127.0.0.1:5000"
echo "  backend     : sqlite:///mlflow.db"
echo "  artifacts   : ./mlruns"

mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
