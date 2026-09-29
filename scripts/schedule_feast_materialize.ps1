# Host-scheduled Feast materialize (Feast stays on host — not in Docker compose)
#
# Manual:
#   .\scripts\schedule_feast_materialize.ps1 -DataPath "C:\path\to\survey.xlsx"
#
# Task Scheduler (daily example — see register_scheduled_jobs.ps1):
#   Program: powershell.exe
#   Arguments: -File "...\scripts\schedule_feast_materialize.ps1" -DataPath "...\survey.xlsx"

param(
    [Parameter(Mandatory = $true)]
    [string]$DataPath
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$LogDir = Join-Path $Root "data\monitoring\job_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "feast_materialize_$Stamp.log"

Write-Host "=== Scheduled Feast materialize ==="
Write-Host "Data: $DataPath"
Write-Host "Log:  $LogFile"

& "$Root\scripts\feast_materialize.ps1" -DataPath $DataPath *>&1 |
  Tee-Object -FilePath $LogFile

if ($LASTEXITCODE -ne 0) {
    throw "Feast materialize failed with exit $LASTEXITCODE (see $LogFile)"
}

Write-Host "Done. Offline/online Feast stores updated under data\feast\"
