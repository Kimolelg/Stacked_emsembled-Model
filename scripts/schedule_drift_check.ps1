# Host-scheduled Evidently drift check (stays on host)
#
# Manual:
#   .\scripts\schedule_drift_check.ps1
#   .\scripts\schedule_drift_check.ps1 -BuildReference
#
# Prefer running after Feast materialize so reference features stay fresh.

param(
    [switch]$BuildReference,
    [int]$MinRows = 30
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$LogDir = Join-Path $Root "data\monitoring\job_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "drift_check_$Stamp.log"

Write-Host "=== Scheduled Evidently drift check ==="
Write-Host "BuildReference: $BuildReference  MinRows: $MinRows"
Write-Host "Log: $LogFile"

$argsList = @()
if ($BuildReference) { $argsList += "-BuildReference" }
$argsList += "-MinRows"
$argsList += $MinRows

& "$Root\scripts\run_drift_check.ps1" @argsList *>&1 |
  Tee-Object -FilePath $LogFile

if ($LASTEXITCODE -ne 0) {
    throw "Drift check failed with exit $LASTEXITCODE (see $LogFile)"
}

$Summary = Join-Path $Root "data\monitoring\last_drift_summary.json"
if (Test-Path $Summary) {
    Write-Host "Summary: $Summary"
    Get-Content $Summary -Raw
}

Write-Host "Report: data\monitoring\reports\latest_drift_report.html"
Write-Host "View in UI: Monitoring page → Latest Evidently report"
