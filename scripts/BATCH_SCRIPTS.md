# Batch Processing Scripts

This directory contains scripts for batch processing multiple videos with different pose detection backends.

## OpenPose Batch Processing

### Quick Start

**Windows (PowerShell):**
```powershell
.\scripts\run-openpose-batch.ps1
```

**Linux/macOS (Bash):**
```bash
./scripts/run-openpose-batch.sh
```

### Prerequisites

1. **OpenPose Installation**: Set the `OPENPOSE_HOME` environment variable to your OpenPose installation directory
2. **Python Environment**: The script will automatically create and activate a virtual environment using `uv`
3. **Video Files**: Place your videos in the `data/videos` directory

### Features

- **Automatic Environment Setup**: Creates and activates Python virtual environment
- **Progress Tracking**: Shows real-time progress with time estimates
- **Resume Capability**: Continue from where you left off if interrupted
- **Dry Run Mode**: Preview what will be processed without running
- **Comprehensive Output**: Generates JSON, CSV, overlay videos, and Toronto gait format
- **Error Handling**: Graceful handling of failed videos with detailed reporting

### Usage Examples

**Basic usage:**
```powershell
# Windows
.\scripts\run-openpose-batch.ps1

# Linux/macOS
./scripts/run-openpose-batch.sh
```

**Preview what will be processed:**
```powershell
# Windows
.\scripts\run-openpose-batch.ps1 -DryRun

# Linux/macOS
./scripts/run-openpose-batch.sh --dry-run
```

**Resume from previous run:**
```powershell
# Windows
.\scripts\run-openpose-batch.ps1 -Resume

# Linux/macOS
./scripts/run-openpose-batch.sh --resume
```

**Custom configuration:**
```powershell
# Windows
.\scripts\run-openpose-batch.ps1 -MinConfidence "0.3" -NetResolution "832x480" -ModelPose "COCO"

# Linux/macOS
./scripts/run-openpose-batch.sh --min-confidence "0.3" --net-resolution "832x480" --model-pose "COCO"
```

### Parameters

#### PowerShell Script Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Resume` | `false` | Resume from previous run |
| `-DryRun` | `false` | Show what would be processed without running |
| `-OutputBaseDir` | `"outputs\openpose-batch"` | Base output directory |
| `-MinConfidence` | `"0.5"` | Minimum confidence threshold |
| `-NetResolution` | `"656x368"` | Network resolution |
| `-ModelPose` | `"BODY_25"` | Pose model (BODY_25, COCO, MPI, MPI_4_layers) |
| `-TorontoGaitFormat` | `true` | Enable Toronto gait format output |
| `-ExtractComprehensiveFrames` | `true` | Enable comprehensive frame extraction |
| `-Verbose` | `true` | Enable verbose output |

#### Bash Script Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--resume` | `false` | Resume from previous run |
| `--dry-run` | `false` | Show what would be processed without running |
| `--output-dir` | `"outputs/openpose-batch"` | Base output directory |
| `--min-confidence` | `"0.5"` | Minimum confidence threshold |
| `--net-resolution` | `"656x368"` | Network resolution |
| `--model-pose` | `"BODY_25"` | Pose model (BODY_25, COCO, MPI, MPI_4_layers) |
| `--no-toronto-gait-format` | N/A | Disable Toronto gait format output |
| `--no-extract-comprehensive-frames` | N/A | Disable comprehensive frame extraction |
| `--no-verbose` | N/A | Disable verbose output |

### Output Structure

```
outputs/openpose-batch/
├── processing_progress.txt           # Progress tracking
├── OAW01-top/                       # Individual video results
│   ├── pose_OAW01-top_openpose_20250125_143022/
│   │   ├── poses.json               # Raw pose data
│   │   ├── poses.csv                # CSV format
│   │   ├── toronto_gait.json        # Toronto gait format
│   │   ├── overlay.mp4              # Video with pose overlay
│   │   ├── run.json                 # Run metadata
│   │   ├── run.log                  # Processing logs
│   │   └── frames/                  # Individual frames (if enabled)
│   └── ...
├── OAW01-bottom/
│   └── ...
└── ...
```

### Environment Variables

The script supports both environment variables for OpenPose:
- `OPENPOSE_HOME` (preferred, standard)
- `OPENPOSEPATH` (legacy support)

### Error Recovery

If the script fails or is interrupted:

1. **Check the logs**: Each video has its own log file in the output directory
2. **Use resume mode**: Run with `-Resume` (PowerShell) or `--resume` (Bash) to continue
3. **Check OpenPose setup**: Verify `OPENPOSE_HOME` is set correctly
4. **Manual retry**: Process individual videos using the single video scripts

### Performance Notes

- Processing time varies by video length, resolution, and hardware
- OpenPose typically processes 2-10 FPS depending on configuration
- Each video (1-2 minutes) may take 5-15 minutes to process
- Total batch time for 28 videos: approximately 3-7 hours

### Troubleshooting

**Common Issues:**

1. **OpenPose not found**: Set `OPENPOSE_HOME` environment variable
2. **Permission denied**: Ensure scripts are executable (`chmod +x` on Linux/macOS)
3. **Virtual environment issues**: Delete `.venv` folder and let script recreate it
4. **Out of memory**: Reduce `net-resolution` or process fewer videos simultaneously

For more detailed information, see the main project documentation. 