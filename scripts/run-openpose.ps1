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
if (-not (Test-Path cli/detect.py)) {
    Write-Host "[ERROR] cli/detect.py is missing. Please ensure the repository is fully cloned."
    exit 1
}

$backend = 'openpose'
$model_pose = 'COCO'

python cli/detect.py `
    --video data/video/video.avi `
    --output-dir "outputs/video-$backend" `
    --overlay-video "outputs/video-$backend/overlay.mp4" `
    --backend $backend `
    --min-confidence 0.5 `
    --net-resolution 656x368 `
    --model-pose $model_pose `
    --toronto-gait-format `
    --extract-comprehensive-frames `
    --verbose