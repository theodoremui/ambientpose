# Advanced CLI Reference for AmbientPose

This document describes all command-line options for `cli/detect.py`, backend-specific support, and output formats.

## CLI Arguments

| Argument                        | Type      | Description                                                                                 |
|----------------------------------|-----------|---------------------------------------------------------------------------------------------|
| --video <path>                   | str       | Path to input video file (mutually exclusive with --image-dir)                              |
| --image-dir <path>               | str       | Path to directory of input images (mutually exclusive with --video)                         |
| --output <path>                  | str       | Path to output JSON file (default: outputs/pose_<timestamp>.json)                           |
| --output-dir <path>              | str       | Directory for output files (default: outputs)                                               |
| --backend <name>                 | str       | Backend to use: auto, mediapipe, ultralytics, openpose, alphapose (default: auto)           |
| --min-confidence <float>         | float     | Minimum confidence threshold for detections (default: 0.5)                                  |
| --confidence-threshold <float>   | float     | Alias for --min-confidence (OpenPose compatibility)                                         |
| --net-resolution <WxH>           | str       | Network input resolution (e.g., 656x368, 832x512)                                           |
| --model-pose <name>              | str       | Pose model to use (backend-specific: COCO, BODY_25, MPI, etc.)                              |
| --overlay-video <path>           | str       | Path to save overlay video file (MP4 format)                                                |
| --toronto-gait-format            | flag      | Output results in Toronto gait analysis format                                              |
| --extract-comprehensive-frames   | flag      | Extract comprehensive frame metadata and analysis                                           |
| --debug                          | flag      | Enable debug mode                                                                           |
| --verbose                        | flag      | Enable verbose logging with detailed information                                            |

## Backend-Specific Support

| Option                   | MediaPipe | Ultralytics | OpenPose         | AlphaPose        |
|-------------------------|-----------|-------------|------------------|------------------|
| --net-resolution        | (auto)    | yes         | yes              | yes (limited)    |
| --model-pose            | (fixed)   | yes         | yes              | yes              |
| --overlay-video         | yes       | yes         | yes              | yes              |
| --toronto-gait-format   | yes       | yes         | yes              | yes              |
| --extract-comprehensive-frames | yes | yes         | yes              | yes              |

- MediaPipe: `--net-resolution` and `--model-pose` are not directly configurable, but `--net-resolution` adjusts model complexity.
- Ultralytics: Supports custom YOLOv8 pose models and resolutions.
- OpenPose: Fully supports `--net-resolution` and `--model-pose`.
- AlphaPose: Supports a limited set of resolutions and models.

## Example Commands

### MediaPipe (default)
```sh
python cli/detect.py --video input.mp4 --backend mediapipe --output outputs/mp.json
```

### Ultralytics
```sh
python cli/detect.py --video input.mp4 --backend ultralytics --net-resolution 1280x1280 --model-pose yolov8l-pose.pt --output outputs/ultra.json
```

### OpenPose (with advanced options)
```sh
python cli/detect.py --video input.mp4 --backend openpose --net-resolution 832x512 --model-pose BODY_25 --overlay-video outputs/openpose_overlay.mp4 --toronto-gait-format --extract-comprehensive-frames --output outputs/openpose.json
```

### AlphaPose
```sh
python cli/detect.py --video input.mp4 --backend alphapose --net-resolution 384x288 --model-pose COCO --output outputs/alphapose.json
```

## Output Files

- **JSON**: Standardized pose data for all frames/images.
- **CSV**: Per-joint pose data (auto-generated alongside JSON).
- **Overlay video (MP4)**: Video with pose skeleton overlays (if --overlay-video is specified).
- **Toronto gait JSON**: Gait analysis for research (if --toronto-gait-format is specified).
- **Comprehensive frames JSON**: Per-frame statistics and pose quality (if --extract-comprehensive-frames is specified).

## Troubleshooting
- See CLI error messages for guidance on missing dependencies or invalid options.
- For backend-specific issues, see the relevant docs in the `docs/` folder.
- For OpenPose/AlphaPose, ensure all environment variables and model files are correctly set up.

## See Also
- [README.md](README.md) for quick start and feature overview
- [INSTALL.md](INSTALL.md) for installation and setup
- [OPENPOSE_SETUP.md](OPENPOSE_SETUP.md) for OpenPose backend setup
- [BACKEND_SPECIFIC.md](BACKEND_SPECIFIC.md) for backend feature matrix 