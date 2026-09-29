# Start local MLflow Tracking Server + Model Registry backend
# Uses SHARED store with Docker Compose: ./mlflow_data
#
# Usage (from project root, with venv activated):
#   .\scripts\start_mlflow_server.ps1
#
# UI: http://localhost:5000
#
# Do NOT run this at the same time as `docker compose` MLflow on port 5000.
# If UI errors with FieldDescriptor.label / protobuf AttributeError:
#   pip install "protobuf==4.25.3"

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Store = Join-Path $Root "mlflow_data"
$Db = Join-Path $Store "mlflow.db"
$Artifacts = Join-Path $Store "mlruns"
New-Item -ItemType Directory -Force -Path $Artifacts | Out-Null

# Soft check: MLflow 2.17 needs protobuf 4.x (not 5+/6+/7+)
try {
  $pb = & python -c "import google.protobuf as p; print(p.__version__)" 2>$null
  if ($pb -and ([version]($pb.Split('.')[0]) -ge 5)) {
    Write-Host "WARNING: protobuf $pb is too new for MLflow 2.17 UI."
    Write-Host "  Fix: pip install `"protobuf==4.25.3`""
    Write-Host ""
  }
} catch { }

# Windows path → sqlite URI (forward slashes)
$DbUri = ("sqlite:///" + ($Db -replace "\\", "/"))

Write-Host "Starting MLflow server..."
Write-Host "  tracking UI : http://127.0.0.1:5000"
Write-Host "  backend     : $DbUri"
Write-Host "  artifacts   : $Artifacts"
Write-Host "  (same store as Docker Compose service 'mlflow')"
Write-Host ""

mlflow server `
  --host 127.0.0.1 `
  --port 5000 `
  --backend-store-uri $DbUri `
  --default-artifact-root $Artifacts
