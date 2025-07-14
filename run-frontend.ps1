# Check if venv is activated, if not, create and activate it
if (-not (Test-Path .venv)) {
    Write-Host "[INFO] Python venv not found. Creating with 'uv venv'..."
    uv venv
}

# Activate venv if not already active
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[INFO] Activating Python venv..."
    .\.venv\Scripts\Activate.ps1
}

# Check if frontend/app.py exists
if (-not (Test-Path frontend/app.py)) {
    Write-Host "[ERROR] frontend/app.py is missing. Please ensure the repository is fully cloned."
    exit 1
}

streamlit run .\frontend\app.py
