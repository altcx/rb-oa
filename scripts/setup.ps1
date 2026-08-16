<#
.SYNOPSIS
    One-time setup on Windows: Python venv, dependencies, and the web UI build.

.DESCRIPTION
    Run this once per checkout, from the repository root or from anywhere:

        powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

    Then start the tool with scripts\run.ps1.
#>

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$Web = Join-Path $Root "apps\web"

Write-Host "Puzzle Copilot setup"
Write-Host "  repository: $Root"
Write-Host ""

# -- 1. Python 3.12 virtual environment --------------------------------------

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not on PATH. Install it first: winget install --id=astral-sh.uv"
}

if (Test-Path $Python) {
    Write-Host "[1/3] venv already present at $Venv"
} else {
    Write-Host "[1/3] creating the Python 3.12 venv at $Venv"
    uv venv --python 3.12 $Venv
}

Write-Host "      installing dependencies (editable, with dev extras)"
uv pip install --python $Python -e "$Root[dev]"

# -- 2. Web UI ---------------------------------------------------------------
#
# FastAPI serves apps\web\dist, so the UI must be built before the server is
# any use. There is no dev server in the production flow.

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm is not on PATH. Install Node.js LTS: winget install --id=OpenJS.NodeJS.LTS"
}

Write-Host ""
Write-Host "[2/3] installing web dependencies in $Web"
Push-Location $Web
try {
    npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed with exit code $LASTEXITCODE" }

    Write-Host "[3/3] building the UI (FastAPI serves apps\web\dist)"
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

# -- 3. Report ---------------------------------------------------------------

Write-Host ""
Write-Host "Setup complete."
Write-Host "  interpreter: $Python"
Write-Host "  next:        .\scripts\run.ps1"
Write-Host "  pre-flight:  & '$Python' scripts\acceptance.py"
