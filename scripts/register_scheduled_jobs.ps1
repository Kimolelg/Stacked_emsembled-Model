# Optional helper: create Windows Task Scheduler tasks for Feast + Evidently
# Requires: run PowerShell as Administrator for Register-ScheduledTask
#
# Usage:
#   .\scripts\register_scheduled_jobs.ps1 -DataPath "C:\path\to\survey.xlsx" -WhatIf
#   .\scripts\register_scheduled_jobs.ps1 -DataPath "C:\path\to\survey.xlsx"
#
# Or print instructions only (no admin):
#   .\scripts\register_scheduled_jobs.ps1 -DataPath "..." -PrintOnly

param(
    [Parameter(Mandatory = $true)]
    [string]$DataPath,
    [switch]$WhatIf,
    [switch]$PrintOnly,
    [string]$FeastTime = "02:00",
    [string]$DriftTime = "03:00"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$FeastScript = Join-Path $Root "scripts\schedule_feast_materialize.ps1"
$DriftScript = Join-Path $Root "scripts\schedule_drift_check.ps1"

Write-Host "Project root: $Root"
Write-Host "Survey path:  $DataPath"
Write-Host ""
Write-Host "Recommended schedule (host jobs — Feast/Evidently NOT in Docker):"
Write-Host "  Daily $FeastTime  Feast materialize"
Write-Host "  Daily $DriftTime  Evidently drift (-BuildReference)"
Write-Host ""

$feastArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$FeastScript`" -DataPath `"$DataPath`""
$driftArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$DriftScript`" -BuildReference"

Write-Host "Manual Task Scheduler setup:"
Write-Host "  Program: powershell.exe"
Write-Host "  Feast arguments: $feastArgs"
Write-Host "  Drift arguments: $driftArgs"
Write-Host ""

if ($PrintOnly) {
    Write-Host "PrintOnly set — no tasks registered."
    exit 0
}

function Register-DailyTask {
    param([string]$Name, [string]$Arguments, [string]$Time)
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $Arguments
    $trigger = New-ScheduledTaskTrigger -Daily -At $Time
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    if ($WhatIf) {
        Write-Host "[WhatIf] Would register $Name at $Time"
        return
    }
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
    Write-Host "Registered: $Name ($Time)"
}

Register-DailyTask -Name "TeacherMH-FeastMaterialize" -Arguments $feastArgs -Time $FeastTime
Register-DailyTask -Name "TeacherMH-EvidentlyDrift" -Arguments $driftArgs -Time $DriftTime

Write-Host ""
Write-Host "Done. Open Task Scheduler to confirm. Monitoring UI reads data\monitoring\ after jobs run."
