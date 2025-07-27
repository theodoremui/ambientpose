param(
    [Parameter(Position=0)]
    [string]$VideoPath
)

# Function to show usage
function Show-Usage {
    Write-Host ""
    Write-Host "OpenPose Single Video Processing Script" -ForegroundColor Green
    Write-Host "=======================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Cyan
    Write-Host "  .\scripts\run-openpose.ps1 <video_path>" -ForegroundColor White
    Write-Host ""
    Write-Host "DESCRIPTION:" -ForegroundColor Cyan
    Write-Host "  Processes a single video file using the OpenPose backend for pose detection."
    Write-Host "  Generates pose data in JSON/CSV formats, overlay video, and Toronto gait analysis."
    Write-Host ""
    Write-Host "PARAMETERS:" -ForegroundColor Cyan
    Write-Host "  video_path    Path to the input video file (required)" -ForegroundColor White
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Cyan
    Write-Host "  .\scripts\run-openpose.ps1 data\videos\OAW01-top.mp4" -ForegroundColor Yellow
    Write-Host "  .\scripts\run-openpose.ps1 C:\path\to\my_video.mp4" -ForegroundColor Yellow
    Write-Host "  .\scripts\run-openpose.ps1 demo.avi" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "OUTPUT:" -ForegroundColor Cyan
    Write-Host "  Results will be saved to: outputs\video-openpose\" -ForegroundColor White
    Write-Host "  - pose_points.json     (Raw pose data)"
    Write-Host "  - poses.csv           (CSV format)"
    Write-Host "  - toronto_gait.json   (Toronto dataset format)"
    Write-Host "  - overlay.mp4         (Video with pose overlay)"
    Write-Host "  - frames\             (Individual frames)"
    Write-Host "  - overlays\           (Overlay frames)"
    Write-Host ""
    Write-Host "REQUIREMENTS:" -ForegroundColor Cyan
    Write-Host "  - OpenPose installation with openpose_path environment variable set"
    Write-Host "  - Python virtual environment (will be created automatically)"
    Write-Host ""
    Write-Host "NOTES:" -ForegroundColor Cyan
    Write-Host "  - For batch processing multiple videos, use: .\scripts\run-openpose-batch.ps1"
    Write-Host "  - See README.md for detailed setup instructions"
    Write-Host ""
}

# Check for help flags or missing arguments
if (-not $VideoPath -or $VideoPath -eq "-h" -or $VideoPath -eq "--help" -or $VideoPath -eq "help") {
    Show-Usage
    exit 0
}

# Validate video file exists
if (-not (Test-Path $VideoPath)) {
    Write-Host "[ERROR] Video file not found: $VideoPath" -ForegroundColor Red
    Write-Host "[INFO] Use -h or --help for usage instructions" -ForegroundColor Yellow
    exit 1
}

# Get absolute path and file info
$VideoPath = Resolve-Path $VideoPath
$VideoFileName = [System.IO.Path]::GetFileNameWithoutExtension($VideoPath)
Write-Host "[INFO] Processing video: $VideoPath" -ForegroundColor Cyan
Write-Host "[INFO] Output will be saved as: video-openpose" -ForegroundColor Cyan

# Check if venv is activated, if not, create and activate it
if (-not (Test-Path .venv)) {
    Write-Host "[INFO] Python venv not found. Creating with 'uv venv'..." -ForegroundColor Yellow
    uv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate venv if not already active
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[INFO] Activating Python venv..." -ForegroundColor Yellow
    .\.venv\Scripts\Activate.ps1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to activate virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Check if CLI exists
if (-not (Test-Path cli/detect.py)) {
    Write-Host "[ERROR] cli/detect.py is missing. Please ensure the repository is fully cloned." -ForegroundColor Red
    exit 1
}

# Check OpenPose environment variable
$openpose_path = $env:OPENPOSEPATH
if (-not $openpose_path) {
    $openpose_path = $env:OPENPOSE_HOME
}

if (-not $openpose_path) {
    Write-Host "[WARNING] Neither OPENPOSE_HOME nor OPENPOSEPATH environment variable is set." -ForegroundColor Yellow
    Write-Host "[WARNING] OpenPose backend may fall back to other available backends." -ForegroundColor Yellow
} else {
    Write-Host "[INFO] OpenPose installation found: $openpose_path" -ForegroundColor Green
}

# Configuration
$backend = 'openpose'
$model_pose = 'BODY_25'

Write-Host "[START] Running OpenPose pose detection..." -ForegroundColor Green

# Execute pose detection
python cli/detect.py `
    --video $VideoPath `
    --output-dir "outputs\video-$backend\$VideoFileName" `
    --overlay-video "outputs\video-$backend\$VideoFileName\overlay.mp4" `
    --backend $backend `
    --min-confidence 0.5 `
    --net-resolution 656x368 `
    --model-pose $model_pose `
    --toronto-gait-format `
    --extract-comprehensive-frames `
    --verbose

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] OpenPose processing completed successfully!" -ForegroundColor Green
    Write-Host "[OUTPUT] Results saved to: outputs\video-$backend\" -ForegroundColor Cyan
} else {
    Write-Host "[ERROR] OpenPose processing failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}