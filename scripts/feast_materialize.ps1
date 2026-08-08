# Build Feast offline table from TSC survey and materialize online store
# Prerequisite: venv activated, feast installed (pip install feast pyarrow)
#
# Usage:
#   .\scripts\feast_materialize.ps1 -DataPath "C:\path\to\survey.xlsx"

param(
    [Parameter(Mandatory = $true)]
    [string]$DataPath,
    [switch]$SkipMaterialize
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path $DataPath)) {
    throw "Survey file not found: $DataPath"
}

Write-Host "=== Feast pipeline (processing → parquet → apply → materialize) ==="
Write-Host "Data: $DataPath"

$argsList = @(
    "-m", "src.data.feast_pipeline",
    "--data_path", $DataPath
)
if ($SkipMaterialize) {
    $argsList += "--skip-materialize"
}

python @argsList

Write-Host ""
Write-Host "Done. Offline table: data\feast\teacher_survey_features.parquet"
Write-Host "Train with Feast features:"
Write-Host "  python src/models/train.py --use-feast --model meta4"
Write-Host "API: POST /predict {`"teacher_id`": 1}  or questionnaire fields"
