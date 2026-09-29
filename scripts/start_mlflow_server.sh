#!/usr/bin/env bash
# Start local MLflow Tracking Server — shared store with Docker: ./mlflow_data
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STORE="$ROOT/mlflow_data"
DB="$STORE/mlflow.db"
ART="$STORE/mlruns"
mkdir -p "$ART"

echo "Starting MLflow server..."
echo "  tracking UI : http://127.0.0.1:5000"
echo "  backend     : sqlite:///$DB"
echo "  artifacts   : $ART"
echo "  (same store as Docker Compose service 'mlflow')"

mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri "sqlite:///$DB" \
  --default-artifact-root "$ART"
