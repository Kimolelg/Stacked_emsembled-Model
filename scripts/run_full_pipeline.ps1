# Full pipeline: MLflow server notes + train all meta models
# Usage (from project root, venv activated):
#   .\scripts\run_full_pipeline.ps1 -DataPath "C:\path\to\survey.xlsx"
#
# Keep MLflow server running in a SEPARATE terminal first.

param(
    [Parameter(Mandatory = $true)]
    [string]$DataPath,

    [string]$Model = "all",   # meta1|meta2|meta3|meta4|all|rf
    [string]$OutputDir = "artifacts"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "Project root: $Root"
Write-Host "Data: $DataPath"
Write-Host "Model: $Model"
Write-Host ""
Write-Host "Prerequisite: MLflow server must be running in another terminal:"
Write-Host "  .\scripts\start_mlflow_server.ps1"
Write-Host "  UI: http://127.0.0.1:5000"
Write-Host ""

if (-not (Test-Path $DataPath)) {
    throw "Data file not found: $DataPath"
}

$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

Write-Host "=== Training ($Model) + MLflow register ==="
python src/models/train.py `
  --data_path $DataPath `
  --model $Model `
  --output_dir $OutputDir

Write-Host ""
Write-Host "=== Next steps ==="
Write-Host "1. Open http://127.0.0.1:5000 → Models → pick a model → Add Alias: champion"
Write-Host "2. Start API:"
Write-Host '   $env:MLFLOW_LOAD_REGISTRY="1"'
Write-Host '   $env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"'
Write-Host '   $env:MLFLOW_REGISTERED_MODEL_NAME="teacher-mental-health-risk-meta4"'
Write-Host '   $env:MLFLOW_MODEL_ALIAS="champion"'
Write-Host "   uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000"
Write-Host "3. Check: http://127.0.0.1:8000/health  and  /model-info"
Write-Host "4. Predict: POST http://127.0.0.1:8000/predict"
