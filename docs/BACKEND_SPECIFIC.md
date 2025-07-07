# Backend Feature Matrix

This document summarizes the capabilities, supported options, and limitations of each pose detection backend in AmbientPose.

| Feature/Option                | MediaPipe      | Ultralytics YOLO      | OpenPose           | AlphaPose           |
|-------------------------------|----------------|-----------------------|--------------------|---------------------|
| Platform                      | All            | All                   | Win/Linux (GPU)    | Win/Linux (GPU)     |
| Python API                    | Yes            | Yes                   | Yes (if built)     | Yes                 |
| CLI Integration               | Yes            | Yes                   | Yes                | Yes                 |
| --net-resolution              | (auto, hint)   | Yes                   | Yes                | Yes (limited)       |
| --model-pose                  | (fixed)        | Yes                   | Yes                | Yes                 |
| --overlay-video               | Yes            | Yes                   | Yes                | Yes                 |
| --toronto-gait-format         | Yes            | Yes                   | Yes                | Yes                 |
| --extract-comprehensive-frames| Yes            | Yes                   | Yes                | Yes                 |
| Multi-person                  | Yes            | Yes                   | Yes                | Yes                 |
| Real-time                     | Yes (CPU/GPU)  | Yes (GPU)             | Yes (GPU)          | Yes (GPU)           |
| Keypoints                     | 33             | 17 (COCO)             | 18/25/COCO/MPI     | 17/COCO/HALPE       |
| Custom models                 | No             | Yes                   | Yes                | Yes                 |
| Tracking                      | Yes (simple)   | Yes (simple)          | No (per frame)     | Yes (advanced)      |
| Output formats                | JSON, CSV, MP4 | JSON, CSV, MP4        | JSON, CSV, MP4     | JSON, CSV, MP4      |
| Toronto gait output           | Yes            | Yes                   | Yes                | Yes                 |
| Comprehensive frame output    | Yes            | Yes                   | Yes                | Yes                 |

## Backend Notes

### MediaPipe
- Fastest, most portable, works on CPU and GPU.
- Fixed model (33 keypoints, COCO-like).
- `--net-resolution` adjusts model complexity, not true input size.
- Best for real-time, low-latency applications.

### Ultralytics YOLO
- Supports custom YOLOv8 pose models.
- `--net-resolution` and `--model-pose` fully supported.
- Good balance of speed and accuracy on GPU.

### OpenPose
- Most accurate, supports BODY_25, COCO, MPI models.
- Requires GPU for real-time performance.
- Python API preferred, binary fallback supported.
- `--net-resolution` and `--model-pose` fully supported.
- See [OPENPOSE_SETUP.md](OPENPOSE_SETUP.md) for details.

### AlphaPose
- Research-grade, supports COCO, HALPE, MPII models.
- Requires GPU and complex setup.
- `--net-resolution` supports only certain values (e.g., 256x192, 384x288).
- Advanced tracking and multi-person support.

## Recommendations
- For speed and portability: **MediaPipe**
- For accuracy and research: **OpenPose** or **AlphaPose**
- For custom models: **Ultralytics YOLO** or **AlphaPose**
- For gait analysis: Any backend, but OpenPose/AlphaPose recommended for BODY_25/COCO keypoints

See [ADVANCED_CLI.md](ADVANCED_CLI.md) for full CLI reference and [INSTALL.md](INSTALL.md) for setup instructions. 