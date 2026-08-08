# Start local MLflow Tracking Server + Model Registry backend
# sqlite backend + ./mlruns artifact root
#
# Usage (from project root, with venv activated):
#   .\scripts\start_mlflow_server.ps1
#
# UI: http://localhost:5000

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

New-Item -ItemType Directory -Force -Path "mlruns" | Out-Null

Write-Host "Starting MLflow server..."
Write-Host "  tracking UI : http://127.0.0.1:5000"
Write-Host "  backend     : sqlite:///mlflow.db"
Write-Host "  artifacts   : ./mlruns"
Write-Host ""

mlflow server `
  --host 127.0.0.1 `
  --port 5000 `
  --backend-store-uri "sqlite:///mlflow.db" `
  --default-artifact-root "./mlruns"
