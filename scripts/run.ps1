<#
.SYNOPSIS
    Start the Puzzle Copilot server on Windows.

.DESCRIPTION
    Run scripts\setup.ps1 first. Then:

        powershell -ExecutionPolicy Bypass -File scripts\run.ps1

    The server listens on http://127.0.0.1:8765 and does not exit until you
    stop it with Ctrl+C. Open the browser on monitor 2 and drag it there; the
    HUD is at /hud and settings at /settings.

.PARAMETER SkipChecks
    Start without verifying that the UI has been built.
#>

param(
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Dist = Join-Path $Root "apps\web\dist"

Write-Host "Puzzle Copilot"
Write-Host "  repository:  $Root"
Write-Host "  interpreter: $Python"

if (-not (Test-Path $Python)) {
    throw "No venv at $Python. Run scripts\setup.ps1 first."
}

if (-not $SkipChecks) {
    # FastAPI serves the built UI from apps\web\dist. Without it the server
    # starts and every page 404s, which looks like a backend fault and is not.
    if (-not (Test-Path (Join-Path $Dist "index.html"))) {
        throw "The web UI is not built ($Dist). Run scripts\setup.ps1, or 'npm run build' in apps\web."
    }
}

# UTF-8 so the console never mangles a model name or a rule quotation. The
# environment variable covers any child process the server spawns as well.
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "  url:         http://127.0.0.1:8765   (HUD at /hud, settings at /settings)"
Write-Host "  stop:        Ctrl+C"
Write-Host ""
Write-Host "Drag the browser window to monitor 2 before the run starts."
Write-Host ""

& $Python -m services.core.main
exit $LASTEXITCODE
