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

# Check if cli/detect.py exists
if (-not (Test-Path cli/detect.py)) {
    Write-Host "[ERROR] cli/detect.py is missing. Please ensure the repository is fully cloned."
    exit 1
}

$backend = 'ultralytics'
$model_pose = 'models/yolov8n-pose.pt'
# $video_path = '../videos/OAW09-top.mp4'
$video_path = 'data/video/video.avi'

python cli/detect.py `
    --video $video_path `
    --output-dir "outputs/video-$backend" `
    --overlay-video "outputs/video-$backend/overlay.mp4" `
    --backend $backend `
    --min-confidence 0.5 `
    --net-resolution 672x384 `
    --model-pose $model_pose `
    --toronto-gait-format `
    --extract-comprehensive-frames `
    --verbose
    # --net-resolution 656x368 `