# Build reference (from Feast) and run Evidently drift check vs production log
#
# Usage (venv activated, from project root):
#   .\scripts\run_drift_check.ps1
#   .\scripts\run_drift_check.ps1 -BuildReference
#   .\scripts\run_drift_check.ps1 -MinRows 20

param(
    [switch]$BuildReference,
    [int]$MinRows = 30
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "=== Evidently drift check (reference=Feast, current=production log) ==="

$py = @"
from src.monitoring.drift_monitor import run_drift_check
import json
s = run_drift_check(build_reference=$($BuildReference.IsPresent | ForEach-Object { if ($_) { 'True' } else { 'False' } }), min_current_rows=$MinRows)
print(json.dumps(s, indent=2, default=str))
"@

# Fix boolean for Python
$buildPy = if ($BuildReference) { "True" } else { "False" }
python -c @"
from src.monitoring.drift_monitor import run_drift_check
import json
s = run_drift_check(build_reference=$buildPy, min_current_rows=$MinRows)
print(json.dumps(s, indent=2, default=str))
if s.get('evidently', {}).get('report_path'):
    print('Report:', s['evidently']['report_path'])
"@
