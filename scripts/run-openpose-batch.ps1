# OpenPose Batch Processing Script
# Processes all videos in data/videos directory using OpenPose backend
# Author: Theodore Mui
# Date: 2025-01-25

param(
    [switch]$Resume = $false,
    [switch]$DryRun = $false,
    [string]$OutputBaseDir = "outputs\openpose-batch",
    [string]$MinConfidence = "0.5",
    [string]$NetResolution = "656x368",
    [string]$ModelPose = "BODY_25",
    [switch]$TorontoGaitFormat = $true,
    [switch]$ExtractComprehensiveFrames = $true,
    [switch]$Verbose = $true
)

# Colors for output
$Green = @{ForegroundColor = "Green"}
$Yellow = @{ForegroundColor = "Yellow"}
$Red = @{ForegroundColor = "Red"}
$Cyan = @{ForegroundColor = "Cyan"}
$Magenta = @{ForegroundColor = "Magenta"}

Write-Host "[OPENPOSE BATCH] Starting OpenPose batch processing..." @Green
Write-Host "[INFO] Processing all videos in data\videos directory" @Cyan

# Validate environment
Write-Host "[CHECK] Validating environment..." @Cyan

# Check if venv is activated, if not, create and activate it
if (-not (Test-Path .venv)) {
    Write-Host "[INFO] Python venv not found. Creating with 'uv venv'..." @Yellow
    uv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment" @Red
        exit 1
    }
}

# Activate venv if not already active
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[INFO] Activating Python venv..." @Yellow
    .\.venv\Scripts\Activate.ps1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to activate virtual environment" @Red
        exit 1
    }
}

# Check if CLI exists
if (-not (Test-Path cli/detect.py)) {
    Write-Host "[ERROR] cli/detect.py is missing. Please ensure the repository is fully cloned." @Red
    exit 1
}

# Check if videos directory exists
if (-not (Test-Path data/videos)) {
    Write-Host "[ERROR] data/videos directory is missing." @Red
    exit 1
}

# Check OpenPose environment variable (support both OPENPOSE_HOME and OPENPOSEPATH)
$openpose_home = $env:OPENPOSE_HOME
if (-not $openpose_home) {
    $openpose_home = $env:OPENPOSEPATH
}

if (-not $openpose_home) {
    Write-Host "[WARNING] Neither OPENPOSE_HOME nor OPENPOSEPATH environment variable is set." @Yellow
    Write-Host "[WARNING] OpenPose backend may fall back to other available backends." @Yellow
} else {
    Write-Host "[INFO] OpenPose installation found: $openpose_home" @Green
}

# Get all video files
$videoFiles = Get-ChildItem -Path "data\videos\*.mp4" | Sort-Object Name
$totalVideos = $videoFiles.Count

if ($totalVideos -eq 0) {
    Write-Host "[ERROR] No MP4 files found in data\videos directory." @Red
    exit 1
}

Write-Host "[INFO] Found $totalVideos video files to process" @Green

# Create base output directory
if (-not (Test-Path $OutputBaseDir)) {
    New-Item -Path $OutputBaseDir -ItemType Directory -Force | Out-Null
}

# Create progress tracking file
$progressFile = Join-Path $OutputBaseDir "processing_progress.txt"
$completedVideos = @()

if ($Resume -and (Test-Path $progressFile)) {
    $completedVideos = Get-Content $progressFile -ErrorAction SilentlyContinue
    Write-Host "[INFO] Resume mode enabled. Found $($completedVideos.Count) previously completed videos." @Yellow
}

# Dry run mode
if ($DryRun) {
    Write-Host "[DRY RUN] Would process the following videos:" @Magenta
    foreach ($video in $videoFiles) {
        if ($Resume -and ($completedVideos -contains $video.Name)) {
            Write-Host "  [SKIP] $($video.Name) (already completed)" @Yellow
        } else {
            Write-Host "  [PROCESS] $($video.Name)" @Cyan
        }
    }
    Write-Host "[DRY RUN] Use without -DryRun to actually process videos." @Magenta
    exit 0
}

# Process each video
$processedCount = 0
$skippedCount = 0
$failedCount = 0
$startTime = Get-Date

Write-Host "[START] Beginning batch processing at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" @Green
Write-Host "========================================" @Green

foreach ($video in $videoFiles) {
    $videoName = $video.BaseName
    $videoPath = $video.FullName
    $currentIndex = $videoFiles.IndexOf($video) + 1
    
    Write-Host "[$currentIndex/$totalVideos] Processing: $($video.Name)" @Cyan
    
    # Skip if already completed in resume mode
    if ($Resume -and ($completedVideos -contains $video.Name)) {
        Write-Host "  [SKIP] Already completed in previous run" @Yellow
        $skippedCount++
        continue
    }
    
    # Create output directory for this video
    $videoOutputDir = Join-Path $OutputBaseDir $videoName
    
    try {
        # Build command arguments
        $args = @(
            "cli/detect.py"
            "--video", $videoPath
            "--output-dir", $videoOutputDir
            "--overlay-video", (Join-Path $videoOutputDir "overlay.mp4")
            "--backend", "openpose"
            "--min-confidence", $MinConfidence
            "--net-resolution", $NetResolution
            "--model-pose", $ModelPose
        )
        
        if ($TorontoGaitFormat) {
            $args += "--toronto-gait-format"
        }
        
        if ($ExtractComprehensiveFrames) {
            $args += "--extract-comprehensive-frames"
        }
        
        if ($Verbose) {
            $args += "--verbose"
        }
        
        Write-Host "  [RUN] Executing OpenPose detection..." @Cyan
        
        # Execute the command
        $process = Start-Process -FilePath "python" -ArgumentList $args -Wait -PassThru -NoNewWindow
        
        if ($process.ExitCode -eq 0) {
            Write-Host "  [SUCCESS] $($video.Name) completed successfully" @Green
            
            # Add to completed list
            Add-Content -Path $progressFile -Value $video.Name
            $processedCount++
        } else {
            Write-Host "  [ERROR] $($video.Name) failed with exit code $($process.ExitCode)" @Red
            $failedCount++
        }
        
    } catch {
        Write-Host "  [ERROR] Exception processing $($video.Name): $($_.Exception.Message)" @Red
        $failedCount++
    }
    
    # Show progress
    $elapsed = (Get-Date) - $startTime
    $avgTimePerVideo = if ($processedCount -gt 0) { $elapsed.TotalMinutes / $processedCount } else { 0 }
    $remainingVideos = $totalVideos - $currentIndex
    $estimatedTimeRemaining = if ($avgTimePerVideo -gt 0) { [TimeSpan]::FromMinutes($avgTimePerVideo * $remainingVideos) } else { [TimeSpan]::Zero }
    
    Write-Host "  [PROGRESS] Processed: $processedCount | Skipped: $skippedCount | Failed: $failedCount | Remaining: $remainingVideos" @Yellow
    if ($avgTimePerVideo -gt 0) {
        Write-Host "  [TIME] Avg: $([math]::Round($avgTimePerVideo, 1)) min/video | ETA: $($estimatedTimeRemaining.ToString('hh\:mm\:ss'))" @Yellow
    }
    Write-Host ""
}

# Final summary
$endTime = Get-Date
$totalTime = $endTime - $startTime

Write-Host "========================================" @Green
Write-Host "[COMPLETE] Batch processing finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" @Green
Write-Host "[SUMMARY] Total time: $($totalTime.ToString('hh\:mm\:ss'))" @Green
Write-Host "[SUMMARY] Videos processed: $processedCount" @Green
Write-Host "[SUMMARY] Videos skipped: $skippedCount" @Yellow
if ($failedCount -gt 0) {
    Write-Host "[SUMMARY] Videos failed: $failedCount" @Red
} else {
    Write-Host "[SUMMARY] Videos failed: $failedCount" @Green
}
Write-Host "[SUMMARY] Success rate: $([math]::Round(($processedCount / ($processedCount + $failedCount)) * 100, 1))%" @Green

if ($processedCount -gt 0) {
    $avgTime = $totalTime.TotalMinutes / $processedCount
    Write-Host "[SUMMARY] Average time per video: $([math]::Round($avgTime, 1)) minutes" @Green
}

Write-Host "[OUTPUT] All results saved in: $OutputBaseDir" @Cyan

if ($failedCount -gt 0) {
    Write-Host "[NOTE] Some videos failed to process. Check the logs for details." @Yellow
    Write-Host "[NOTE] You can use -Resume to retry failed videos." @Yellow
}

Write-Host "[DONE] OpenPose batch processing complete!" @Green 