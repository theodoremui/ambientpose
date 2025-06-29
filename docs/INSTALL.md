# AlphaDetect – Installation Guide

*Last updated: 2025-06-21*

This document walks you through installing **AlphaDetect** on macOS / Linux / Windows, from prerequisites to production-ready deployments. Follow the steps sequentially or jump to the section that matches your needs.

---

## Table of Contents
- [AlphaDetect – Installation Guide](#alphadetect--installation-guide)
  - [Table of Contents](#table-of-contents)
  - [1. System Requirements](#1-system-requirements)
  - [2. Prerequisites Installation](#2-prerequisites-installation)
    - [Linux (Ubuntu)](#linux-ubuntu)
    - [macOS](#macos)
    - [Windows](#windows)
  - [3. AlphaPose Setup and Configuration](#3-alphapose-setup-and-configuration)
    - [GPU vs CPU Build](#gpu-vs-cpu-build)
  - [4. Python Environment Setup with uv](#4-python-environment-setup-with-uv)
    - [4.1 Install **uv** (one-time)](#41-install-uv-one-time)
    - [4.2 Create virtual environment \& install deps](#42-create-virtual-environment--install-deps)
    - [4.3 Verify installation](#43-verify-installation)
    - [4.4 Install **AlphaPose** (one-time)](#44-install-alphapose-one-time)
  - [5. Frontend Setup](#5-frontend-setup)
  - [6. Server Setup](#6-server-setup)
  - [7. Database Configuration](#7-database-configuration)
    - [Postgres (prod)](#postgres-prod)
  - [8. Model Downloads](#8-model-downloads)
  - [9. Configuration Options](#9-configuration-options)
  - [10. Troubleshooting Common Issues](#10-troubleshooting-common-issues)
  - [11. Docker / Container Setup](#11-docker--container-setup)
  - [12. Development vs Production Setup](#12-development-vs-production-setup)
    - [Happy Detecting! 🎉](#happy-detecting-)
  - [Troubleshooting Installation Issues](#troubleshooting-installation-issues)
    - [AlphaPose Installation Errors](#alphapose-installation-errors)
      - [Error: `ModuleNotFoundError: No module named 'numpy'`](#error-modulenotfounderror-no-module-named-numpy)
      - [Error: `CUDA_HOME environment variable is not set`](#error-cuda_home-environment-variable-is-not-set)
      - [Error: Compilation failures with halpecocotools or other extensions](#error-compilation-failures-with-halpecocotools-or-other-extensions)
    - [Docker Installation](#docker-installation)
  - [System-Specific Instructions](#system-specific-instructions)
    - [macOS](#macos-1)
    - [Linux](#linux)
    - [Windows](#windows-1)
  - [Verification](#verification)
  - [Alternative Pose Detection Libraries](#alternative-pose-detection-libraries)
  - [Getting Help](#getting-help)
  - [Development Setup](#development-setup)
  - [Server Issues](#server-issues)
    - [SQLModel JSON Type Error](#sqlmodel-json-type-error)
    - [Testing the Server](#testing-the-server)
  - [CLI Multi-Backend Support](#cli-multi-backend-support)
    - [Overview](#overview)
    - [Quick Start](#quick-start)
    - [Backend Selection Strategy](#backend-selection-strategy)
    - [CLI Options](#cli-options)
    - [Backend Comparison](#backend-comparison)
    - [Output Format](#output-format)
    - [Error Handling](#error-handling)
    - [Performance Tips](#performance-tips)
    - [Integration with Server](#integration-with-server)

---

## 1. System Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Ubuntu 20.04 / macOS 12 / Windows 11 | Ubuntu 22.04 LTS |
| **CPU** | 4-core | 8-core+ |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | Optional | NVIDIA GPU (Turing or newer) w/ ≥6 GB VRAM |
| **CUDA** | — | CUDA 11.8 + cuDNN 8 |
| **Disk** | 5 GB | 30 GB (videos + outputs) |
| **Python** | 3.10 | 3.11 |
| **Node.js** | 18 | 20 |
| **Git** | 2.34 | latest |

> AlphaDetect _runs on CPU_, but real-time performance requires a CUDA-capable GPU.

---

## 2. Prerequisites Installation

### Linux (Ubuntu)

```bash
sudo apt update && sudo apt install -y \
  git ffmpeg build-essential python3.11 python3.11-venv python3.11-dev
# Optional GPU drivers
# sudo apt install nvidia-driver-535
```

### macOS

```bash
brew install git ffmpeg pyenv
pyenv install 3.12
pyenv local 3.12
```

> **ℹ️ Apple Silicon (M1 / M2 / M3) note**  
> AlphaPose contains CUDA-specific extensions that *do not* compile on ARM64
> Macs.  A dedicated, step-by-step guide for building AlphaPose with
> Metal Performance Shaders (MPS) **or pure-CPU mode** is available here:  
> **[docs/ALPHAPOSE_MAC_M2_INSTALL.md](ALPHAPOSE_MAC_M2_INSTALL.md)**.  
> Please follow that guide if you are installing on an Apple-Silicon Mac.

### Windows

1. Install **Chocolatey**, then run:  
   ```powershell
   choco install git ffmpeg python --version 3.11
   ```
2. Install **Visual C++ Build Tools 2022** (for Python wheels).  
3. Add Python & Git to your PATH.

---

## 3. AlphaPose Setup and Configuration

AlphaDetect vendors AlphaPose as a Git submodule for reproducibility.

```bash
git clone --recurse-submodules https://github.com/philmui/alphadetect.git
cd alphadetect
# OR, if you cloned previously without submodules:
git submodule update --init --recursive
```

### GPU vs CPU Build

| Mode | Command |
|------|---------|
| **GPU (CUDA)** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| **CPU-only**   | `pip install torch torchvision torchaudio` |

AlphaPose itself is installed as an editable dependency later via `pip -e`.

---

## 4. Python Environment Setup with uv

`uv` is a super-fast, all-in-one Python packaging tool written in Rust.  
It replaces the traditional *virtualenv + pip* workflow with a single binary.

### 4.1 Install **uv** (one-time)

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
iwr https://astral.sh/uv/install.ps1 | iex
```

Verify:

```bash
uv --version
```

### 4.2 Create virtual environment & install deps

From the repo root:

```bash
# ① create .venv (Python version read from .python-version if present)
uv venv

# ② install AlphaDetect core + dev extras
uv pip install -e ".[dev]"

# (optional) GPU wheels
# uv pip install -e ".[dev,gpu]"
```

### 4.3 Verify installation

You can either activate the environment **or** use `uv run` to execute commands
without activation:

```bash
# activate (bash / zsh)
source .venv/bin/activate

# OR just:
uv run python -m alphapose --help
uv run python cli/detect.py --help
```

`pyproject.toml` is the **single source-of-truth** for dependencies.  
If you need a fully-pinned lock file (for CI / production), generate it with

```bash
uv pip compile pyproject.toml -o requirements.txt --all-extras
```

> Tip: use `uv pip sync requirements.txt` in CI/CD for lightning-fast,
> reproducible installs directly from the generated lock-file.

### 4.4 Install **AlphaPose** (one-time)

AlphaPose is **not** listed inside `pyproject.toml` because its build step
requires NumPy to be present **before** the package is built, which breaks
most dependency-resolution workflows (including `uv sync`).  
Once the base environment from step&nbsp;4.2 is ready, install AlphaPose
manually:

```bash
# still inside the activated .venv  – or with `uv run`
uv pip install "alphapose @ git+https://github.com/MVIG-SJTU/AlphaPose"
```

If AlphaPose is missing, running `python cli/detect.py` will print a helpful
message reminding you to run the command above.

---

## 5. Frontend Setup

```bash
# still inside repo root
npm install --prefix frontend
npm run dev --prefix frontend     # http://localhost:3000
```

**Tailwind** & **Bootstrap Icons** are pre-configured. Hot-reload is enabled by default.

---

## 6. Server Setup

```bash
uvicorn server.app:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

Important env vars (defaults are `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `ALPHADETECT_STORAGE_DIR` | Where outputs are written | `./outputs` |
| `ALPHADETECT_DB_URL` | SQLModel connection string | `sqlite:///./alphadetect.db` |
| `ALPHADETECT_MAX_UPLOAD_MB` | File size cap | `4096` |

---

## 7. Database Configuration

SQLite is used for local dev; no manual steps needed.

### Postgres (prod)

```bash
docker run -d --name pg -e POSTGRES_PASSWORD=secret -p 5432:5432 postgres:16
export ALPHADETECT_DB_URL=postgresql+psycopg://postgres:secret@localhost:5432/alphadetect
python -m server.migrations upgrade head
```

---

## 8. Model Downloads

AlphaPose models (~200 MB) are fetched automatically on first run, but you can pre-download:

```bash
python scripts/download_models.py          # provided helper
# OR manual (example)
wget -P model_files https://.../yolox_x.pth
```

Ensure the **SMPL** file (`basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`) is placed in `model_files/` for 3D output.

---

## 9. Configuration Options

Edit `configs/default.yaml` (copied to `~/.alphadetect.yaml` on first run):

```yaml
detector: yolox-x
pose_model_cfg: configs/coco/resnet/256x192_res50.yaml
gpu_id: 0           # -1 for CPU
save_raw_frames: true
save_overlay: true
json_precision: 4
```

Override via CLI flags, env vars, or API request body (`"params": {...}`).

---

## 10. Troubleshooting Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No module named 'pycocotools'` | Missing build deps | `pip install --no-binary :all: pycocotools` |
| `CUDA not available` | Wrong Torch build | Reinstall Torch w/ correct CUDA version |
| Web UI 404 on refresh | Next.js SPA routing | Ensure **/app** router build; deploy with `npm run build` |
| `Permission denied ./outputs` | File system perms | `chmod -R u+rw outputs` |

Check `outputs/<taskId>/.stderr.log` for detailed CLI tracebacks.

---

## 11. Docker / Container Setup

```bash
docker compose up               # builds api, worker, frontend
```

`docker-compose.yml` provides:

| Service | Purpose | Ports |
|---------|---------|-------|
| `frontend` | Next.js | 3000 |
| `api`      | FastAPI | 8000 |
| `worker`   | CLI jobs | — |
| `db`       | Postgres | 5432 |
| `minio`    | S3-compatible storage | 9000 / 9001 |

GPU: `docker compose --profile gpu up` (requires NVIDIA Container Runtime).

---

## 12. Development vs Production Setup

| Aspect | Development (`--reload`, `npm run dev`) | Production (`--workers 4`, `npm run build && npm start`) |
|--------|-----------------------------------------|-----------------------------------------------------------|
| Hot reload | ✅ | ❌ |
| Logging | DEBUG | INFO/WARN |
| Database | SQLite | Postgres |
| CORS | Wide-open | Restricted |
| Static files | Next.js dev server | Optimised assets via `next start` |
| Workers | Single | Gunicorn/Uvicorn workers, HPA |

To produce a lean production image:

```bash
docker build -t alphadetect-api -f server/Dockerfile.prod .
docker run -d -p 8000:8000 --gpus all alphadetect-api
```

---

### Happy Detecting! 🎉

If you get stuck, open an issue or chat on our Slack. Contributions are welcome!

## Troubleshooting Installation Issues

### AlphaPose Installation Errors

#### Error: `ModuleNotFoundError: No module named 'numpy'`
**Solution:**
```bash
# Install build dependencies first
uv pip install numpy cython setuptools wheel
make fix-alphapose-build
```

#### Error: `CUDA_HOME environment variable is not set`
**Solution (macOS/CPU-only systems):**
```bash
# Use CPU-only installation
FORCE_CPU=1 make fix-alphapose-build
```

#### Error: Compilation failures with halpecocotools or other extensions
**Solution:**
```bash
# Switch to MediaPipe alternative
make install-pose-alternatives
```

### Docker Installation

If you encounter issues with local installation, use Docker:

```bash
# Build and run with Docker
make docker-build
make docker-up
```

The updated Dockerfile handles most compilation issues automatically.

## System-Specific Instructions

### macOS
- AlphaPose may have compilation issues due to lack of CUDA support
- **Recommended:** Use MediaPipe alternative
- If you must use AlphaPose, ensure you have Xcode command line tools installed

### Linux
- Most installations should work out of the box
- For GPU support, ensure CUDA is properly installed

### Windows
- AlphaPose compilation is challenging on Windows
- **Recommended:** Use Docker or MediaPipe alternative

## Verification

Test your installation:

```bash
# Test CLI
make run-cli

# Test server
make run-server
```

## Alternative Pose Detection Libraries

If AlphaPose continues to cause issues, the project supports these alternatives:

1. **MediaPipe** - Google's pose estimation framework
2. **Ultralytics YOLO** - Modern object detection with pose estimation
3. **OpenPose** - Carnegie Mellon's pose estimation

All alternatives are installed with:
```bash
make install-pose-alternatives
```

## Getting Help

If you continue to experience issues:

1. Check the [GitHub Issues](https://github.com/your-repo/alphadetect/issues)
2. Run `make install-pose-alternatives` for a more reliable setup
3. Use Docker installation as a fallback option

## Development Setup

For development work:

```bash
make install-dev  # Install development dependencies
make test         # Run tests
make format       # Format code
make lint         # Check code quality
```

## Server Issues

### SQLModel JSON Type Error

**Error:** `ValueError: <class 'dict'> has no matching SQLAlchemy type`

**Solution:**
The server uses SQLModel for database operations. When defining model fields with `Dict[str, Any]` type, you need to specify the SQLAlchemy type explicitly:

```python
# Before (causes error)
params: Dict[str, Any] = SQLField(default={})

# After (works correctly)  
from sqlalchemy import JSON
params: Dict[str, Any] = SQLField(default={}, sa_type=JSON)
```

This tells SQLAlchemy to use the JSON column type for storing dictionary data.

### Testing the Server

```bash
# Test server imports
python -c "from server.app import app; print('✅ Server imports successfully!')"

# Start the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

## CLI Multi-Backend Support

### Overview

The AlphaDetect CLI now supports multiple pose detection backends for maximum compatibility and performance:

- **MediaPipe**: Google's cross-platform pose detection (primary, most reliable)
- **Ultralytics YOLO**: Modern YOLOv8-based pose detection (alternative)
- **AlphaPose**: Original backend (if properly installed)

### Quick Start

```bash
# Auto-detect and use best available backend
python cli/detect.py --image-dir path/to/images --backend auto

# Use specific backend
python cli/detect.py --video video.mp4 --backend mediapipe
python cli/detect.py --image-dir images/ --backend ultralytics

# Test with Makefile
make run-cli
```

### Backend Selection Strategy

#### Auto Mode (Recommended)
```bash
python cli/detect.py --image-dir images/ --backend auto
```

The CLI automatically selects backends in this priority order:
1. **MediaPipe** (fastest, most reliable, works on all platforms)
2. **Ultralytics YOLO** (good alternative with modern YOLO models)
3. **AlphaPose** (if available and properly configured)

#### Manual Backend Selection
```bash
# Force MediaPipe (best for most use cases)
python cli/detect.py --video video.mp4 --backend mediapipe

# Force Ultralytics (good for object detection + pose)
python cli/detect.py --image-dir images/ --backend ultralytics

# Force AlphaPose (if you need specific AlphaPose features)
python cli/detect.py --video video.mp4 --backend alphapose
```

### CLI Options

```bash
usage: detect.py [-h] (--video VIDEO | --image-dir IMAGE_DIR) 
                 [--output OUTPUT] [--output-dir OUTPUT_DIR]
                 [--backend {auto,mediapipe,ultralytics,alphapose}] 
                 [--min-confidence MIN_CONFIDENCE] [--debug]

options:
  --video VIDEO         Path to input video file
  --image-dir IMAGE_DIR Path to directory containing image files
  --output OUTPUT       Path to output JSON file
  --output-dir OUTPUT_DIR Directory for output files (default: outputs)
  --backend BACKEND     Backend to use: auto, mediapipe, ultralytics, alphapose
  --min-confidence CONF Minimum confidence threshold (default: 0.5)
  --debug              Enable debug mode
```

### Backend Comparison

| Backend | Speed | Accuracy | Platform Support | Installation |
|---------|--------|----------|------------------|--------------|
| **MediaPipe** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Ultralytics** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **AlphaPose** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

### Output Format

All backends produce consistent JSON output:

```json
{
  "timestamp": "20250622_161230",
  "backend": "mediapipe",
  "input_path": "test_images",
  "frames_dir": "outputs/frames_test_images_20250622_161230",
  "overlay_dir": "outputs/overlay_test_images_20250622_161230",
  "total_poses": 1,
  "poses": [
    {
      "frame_idx": 0,
      "bbox": [x1, y1, x2, y2, confidence],
      "score": 1.0,
      "keypoints": [[x, y, confidence], ...],
      "backend": "mediapipe"
    }
  ]
}
```

### Error Handling

If no backends are available:
```bash
No pose detection backends available!
Please install at least one of the following:
  - MediaPipe: pip install mediapipe
  - Ultralytics: pip install ultralytics
  - AlphaPose: Follow instructions in docs/INSTALL.md
```

If specific backend is unavailable:
```bash
MediaPipe backend not available. Please install MediaPipe.
```

### Performance Tips

1. **For real-time applications**: Use MediaPipe
2. **For highest accuracy**: Use Ultralytics or AlphaPose
3. **For batch processing**: Any backend works well
4. **For production deployment**: Use MediaPipe or Docker

### Integration with Server

The server automatically uses the same CLI with proper backend selection:

```python
# Server calls CLI with auto backend selection
cli_args = ["python", "cli/detect.py", "--video", video_path, "--backend", "auto"]
```
