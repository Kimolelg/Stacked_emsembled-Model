# Assign @champion on ONE registered model and remove it from the others.
# API + frontend both serve whatever currently holds @champion.
#
# Usage (venv on, MLflow up on :5000):
#   .\scripts\set_mlflow_champion.ps1 -Variant meta1
#   .\scripts\set_mlflow_champion.ps1 -Variant meta4 -Version 1

param(
    [string]$TrackingUri = "http://127.0.0.1:5000",
    [Parameter(Mandatory = $true)]
    [ValidateSet("meta1", "meta2", "meta3", "meta4", "rf")]
    [string]$Variant,
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:MLFLOW_TRACKING_URI = $TrackingUri

& python -c @"
from mlflow.tracking import MlflowClient
import os

c = MlflowClient(os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
family = 'teacher-mental-health-risk'
variants = ['meta1', 'meta2', 'meta3', 'meta4', 'rf']
chosen = '$Variant'.strip().lower()
want = '$Version'.strip()
name = f'{family}-{chosen}'

versions = c.search_model_versions(\"name='\" + name + \"'\")
if not versions:
    raise SystemExit(f'No versions for {name}. Train + register first.')
ver = want or str(max(versions, key=lambda v: int(v.version)).version)

for v in variants:
    other = f'{family}-{v}'
    try:
        c.delete_registered_model_alias(other, 'champion')
        print(f'cleared @champion from {other}')
    except Exception:
        pass

c.set_registered_model_alias(name, 'champion', ver)
print(f'Set {name} @champion -> version {ver}')
print('Restart API: docker compose up -d api')
"@
