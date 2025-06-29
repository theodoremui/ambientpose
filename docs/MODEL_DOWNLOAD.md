# Pretrained Model Download Guide for AlphaDetect

This guide covers how to download and set up pretrained models for all pose detection backends supported by AlphaDetect.

## 📋 Overview

AlphaDetect supports three backends, each with different model requirements:

- **MediaPipe**: Automatic download (no setup required)
- **Ultralytics YOLO**: Automatic download (no setup required)
- **AlphaPose**: Manual download required

## 🤖 MediaPipe Models (Automatic)

MediaPipe automatically downloads its pose detection models when first used.

**Features:**
- ✅ No manual setup required
- ✅ Fast and lightweight
- ✅ Works on CPU and GPU
- ✅ Real-time performance

**Models downloaded automatically:**
- `pose_landmarker.task` (~3MB)

**Storage location (Windows):**
- `C:\Users\{username}\AppData\Local\mediapipe\`

## 🎯 Ultralytics YOLO Models (Automatic)

Ultralytics YOLO also downloads models automatically when first used.

**Features:**
- ✅ No manual setup required  
- ✅ Multiple model sizes available
- ✅ High accuracy
- ✅ GPU acceleration

**Models downloaded automatically:**
- `yolov8n-pose.pt` (~6MB) - Nano (fastest)
- `yolov8s-pose.pt` (~22MB) - Small
- `yolov8m-pose.pt` (~51MB) - Medium
- `yolov8l-pose.pt` (~83MB) - Large
- `yolov8x-pose.pt` (~139MB) - Extra Large (most accurate)

**Storage location (Windows):**
- `C:\Users\{username}\.ultralytics\`

## 🚀 AlphaPose Models (Manual Download Required)

AlphaPose requires manual model download. We provide an automated script and manual instructions.

### Method 1: Automated Download (Recommended)

```bash
# List available models
python scripts/download_models.py --list

# Download recommended model only (fast_res50_256x192)
python scripts/download_models.py --fast-only

# Download all models
python scripts/download_models.py --all

# Download specific models
python scripts/download_models.py --models fast_res50_256x192 hrnet_w32_256x192
```

### Method 2: Manual Download

If the automated script fails, you can download models manually:

#### **Fast ResNet50 256x192 (Recommended - 94MB)**
```bash
# Create directory
mkdir -p AlphaPose/pretrained_models

# Option 1: Google Drive (recommended)
# Visit: https://drive.google.com/file/d/1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn/view
# Download and save as: AlphaPose/pretrained_models/fast_res50_256x192.pth

# Option 2: Alternative sources
# Check AlphaPose GitHub releases: https://github.com/MVIG-SJTU/AlphaPose/releases
```

#### **Other Available Models:**

1. **Fast ResNet152 256x192** (220MB) - Higher accuracy
   - Google Drive: `1kfyedqyn8exjbbNmYq8XGd2EooQjPtF9`

2. **Fast ResNet50 384x288** (94MB) - Higher resolution  
   - Google Drive: `18jFI_rQZSzHMzzkruv_k6S0-VGpz8hnR`

3. **HRNet-W32 256x192** (112MB) - Best accuracy
   - Google Drive: `1_wn2ifmoQprBrFoUCTGTse0SKwB9kGgF`

### AlphaPose Model Directory Structure

After downloading, your directory should look like:
```
AlphaPose/
├── pretrained_models/
│   ├── fast_res50_256x192.pth      # Recommended
│   ├── fast_res152_256x192.pth     # Optional
│   ├── fast_res50_384x288.pth      # Optional
│   └── hrnet_w32_256x192.pth       # Optional
├── configs/
└── ...
```

## 🎮 Model Performance Comparison

### Speed vs Accuracy Trade-offs

| Backend | Model | Speed | Accuracy | CPU Support | GPU Support |
|---------|-------|-------|----------|-------------|-------------|
| **MediaPipe** | Default | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ✅ |
| **Ultralytics** | YOLOv8n-pose | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ✅ |
| **Ultralytics** | YOLOv8x-pose | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ✅ |
| **AlphaPose** | Fast ResNet50 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ✅ |
| **AlphaPose** | HRNet-W32 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ✅ |

## 🔧 Recommended Confidence Thresholds

Based on testing, here are suggested confidence thresholds for each backend:

### For Video Processing:
```bash
# MediaPipe (very reliable)
python cli/detect.py --video video.mp4 --backend mediapipe --min-confidence 0.5

# Ultralytics YOLO (balanced)
python cli/detect.py --video video.mp4 --backend ultralytics --min-confidence 0.3

# AlphaPose (with pretrained models)
python cli/detect.py --video video.mp4 --backend alphapose --min-confidence 0.1
```

### For Image Processing:
```bash
# MediaPipe
--min-confidence 0.6

# Ultralytics  
--min-confidence 0.4

# AlphaPose
--min-confidence 0.2
```

## 🧪 Testing Your Setup

After downloading models, test each backend:

```bash
# Test MediaPipe
python cli/detect.py --video data/video/video.avi --backend mediapipe --min-confidence 0.5

# Test Ultralytics
python cli/detect.py --video data/video/video.avi --backend ultralytics --min-confidence 0.3

# Test AlphaPose (after downloading models)
python cli/detect.py --video data/video/video.avi --backend alphapose --min-confidence 0.1
```

## 🔍 Verification Commands

Verify your setup:

```bash
# Check if models exist
python scripts/download_models.py --list

# Check backend availability
python cli/detect.py --help

# Environment verification
python scripts/setup_alphapose_env.py
```

## 🛠️ Troubleshooting

### AlphaPose Model Issues:

1. **Model not found error:**
   ```
   WARNING: No checkpoint files found, using random initialization
   ```
   **Solution:** Download pretrained models using the script above.

2. **Download failures:**
   - Try the manual download method
   - Check your internet connection
   - Use VPN if Google Drive is blocked

3. **File corruption:**
   ```bash
   # Re-download the model
   python scripts/download_models.py --models fast_res50_256x192
   ```

### General Issues:

1. **Out of memory:**
   - Use smaller models (MediaPipe or YOLOv8n-pose)
   - Reduce video resolution
   - Process on CPU: `--device cpu`

2. **Slow performance:**
   - Use GPU acceleration
   - Choose faster models (MediaPipe or Fast ResNet50)
   - Increase confidence threshold

## 📚 Additional Resources

- [AlphaPose Model Zoo](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md)
- [Ultralytics Models](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes)
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

---

**Need help?** Check the issues section or create a new issue with your specific problem. 