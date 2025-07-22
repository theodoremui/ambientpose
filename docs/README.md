# AmbientPose – Whole-Body Pose Detection System

> A project by Theodore Mui in collaboration with Shardul Sapkota
> Ambient Intelligence Project in Stanford Institute for Human-centric AI
> under the guidance of Professor James Landay.

## 1. Project Overview
AmbientPose is an end-to-end, production-ready platform for extracting human joint positions from videos or image sequences.  
Built on top of [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), it combines:

* A **modern Python CLI** for batch processing (`detect.py`)
* A **FastAPI backend** for scalable, asynchronous inference services
* A **Next.js + Tailwind/Bootstrap frontend** for intuitive project management and visualisation

AmbientPose is designed with SOLID principles, strong typing, automated tests, and rich documentation to serve research teams, computer-vision engineers, and hobbyists alike.

---

## 2. Key Features & Capabilities
| Area | Highlights |
|------|------------|
| Pose Estimation | Multi-backend support: MediaPipe, Ultralytics YOLO, OpenPose, and AlphaPose |
| Input Sources  | Local videos, image folders, or remote URLs |
| Outputs | • Per-video JSON with all poses<br>• Frame dumps (`frames/…`) <br>• Overlay images with keypoints (`overlay/…`) |
| Automation | Async FastAPI server shells out to CLI via robust `asyncio` subprocesses |
| UI | Drag-and-drop uploads, project workspace, result browser, tool-tips, dark/light mode |
| Extensibility | Modular packages, typed interfaces, plug-in detector architecture |
| Dev-Ops | Loguru logging, `pyproject.toml`, pre-commit hooks, CI-ready pytest suite |

---

## 3. System Architecture

```text
┌──────────┐     REST/WS       ┌────────────┐   asyncio/pipe    ┌───────────────┐
│ Frontend │  ◀─────────────▶ │  FastAPI   │◀────────────────▶│    detect.py  │
│ (Next.js)│                   │   Server   │    stdio/json     │               │
└──────────┘                   └────────────┘                   └───────────────┘
         ▲                                                               │
         │                              Outputs written to /outputs/<run>│
         └───────────────────────────────────────────────────────────────┘
```

* **Frontend** uploads media, polls status, and renders results.  
* **Server** validates tasks, spawns the CLI as an async subprocess, streams logs, and stores artefacts.  
* **CLI** wraps AlphaPose to generate JSON, raw frames, and overlay frames.  

All components are decoupled and communicate via well-defined HTTP/JSON contracts.

---

## 4. Quick Start

```bash
# 1. Clone & enter repo
git clone https://github.com/theodoremui/ambientpost.git
cd ambientpost

# 2. Create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # installs CLI+server deps
npm install --prefix frontend     # installs UI deps

# 3. Download AlphaPose models
python scripts/download_models.py  # helper script

# 4. Run everything
uvicorn server.app:app --reload          # backend
npm run dev --prefix frontend            # frontend
# or CLI only
python detect.py --video sample.mp4
```

---

## 5. Component Descriptions

### 5.1 CLI – `detect.py`
* Parses inputs (`--video`, `--image-dir`, `--output` …)
* Validates AlphaPose installation, downloads models on first run
* Streams progress to stdout for server capture
* Generates:
  * `outputs/pose_<timestamp>.json`
  * `outputs/frames_<video>_<timestamp>/frame_000123.jpg`
  * `outputs/overlay_<video>_<timestamp>/frame_000123.jpg`

### 5.2 Server – `server/`
* **FastAPI** with async endpoints:
  * `POST /tasks` – submit job
  * `GET /tasks/{id}` – status & metadata
  * `GET /files/...` – secure artefact access
* Spawns CLI via `asyncio.create_subprocess_exec`
* Stores task metadata in lightweight SQLite (pluggable)
* Provides Swagger docs at `/docs`
* 90 %+ pytest coverage

### 5.3 Frontend – `frontend/`
* **Next.js 14** (App Router) with Tailwind CSS + Bootstrap utilities
* Pages:
  * Dashboard, Project detail, Task monitor, Results gallery
* Reusable React hooks for polling task status & fetching artefacts
* Fully typed with TypeScript and ESLint/Prettier setup

---

## 6. Installation Requirements

| Layer        | Version / Notes                          |
|--------------|------------------------------------------|
| Python       | 3.10+ (see `.python-version`)            |
| Node.js      | 20+                                     |
| AlphaPose    | Installed via submodule or `pip install alphapose @ git+https://github.com/MVIG-SJTU/AlphaPose` and CUDA-compatible PyTorch |
| FFmpeg       | For robust video decoding                |
| Optional GPU | NVIDIA GPU w/ CUDA 11.x for real-time performance |

---

## 7. Usage Examples

### 7.1 CLI

```bash
# Process a single video with auto backend selection
python cli/detect.py --video dance.mp4 --backend auto

# Use specific backends
python cli/detect.py --video dance.mp4 --backend mediapipe    # Fast & reliable
python cli/detect.py --video dance.mp4 --backend ultralytics # Modern YOLO
python cli/detect.py --video dance.mp4 --backend openpose    # High accuracy
python cli/detect.py --video dance.mp4 --backend alphapose   # Original backend

# Images folder with explicit output filename
python cli/detect.py --image-dir ./images --output outputs/pose_images_run1.json --backend auto
```

### 7.2 API

```bash
curl -F file=@dance.mp4 http://localhost:8000/tasks
# → returns { "id": "abc123", "status": "PENDING" }

# Poll status
curl http://localhost:8000/tasks/abc123
```

### 7.3 Frontend
1. Open `http://localhost:3000`
2. Click **New Project → Upload Media**
3. Watch progress bar & live logs
4. Explore frames/overlays in the result gallery

---

## 8. Directory Structure

```text
ambientpost/
├── cli/
│   └── detect.py
├── server/
│   ├── app.py
│   ├── services/
│   └── tests/
├── frontend/
│   ├── app/
│   └── docs/DESIGN.md
├── outputs/                # auto-created
├── docs/
│   ├── README.md           # (this file)
│   └── INSTALL.md
└── pyproject.toml
```

---

## 9. Contributing

We welcome issues & pull requests!

1. Fork the repo & create a feature branch (`git checkout -b feat/my-idea`)
2. Run `pre-commit install` for linting hooks
3. Add tests (`pytest -q`)
4. Submit PR with a clear description  
   *Follow the Conventional Commits spec for messages.*

Please read `CODE_OF_CONDUCT.md` before contributing.

---

## 10. License

AmbientPost is released under the **MIT License**.  
See [`LICENSE`](../LICENSE) for full text.

Commercial use of AlphaPose models may require additional licensing – refer to the original AlphaPose repository.

# AmbientPose

AmbientPose is a multi-backend human pose detection toolkit supporting MediaPipe, Ultralytics YOLO, OpenPose, and AlphaPose. It provides a unified CLI for pose extraction from videos and images, with advanced options for research and production workflows.

## Features
- Multi-backend support: MediaPipe, Ultralytics YOLO, OpenPose, AlphaPose
- Automatic backend selection or manual override
- Advanced CLI options:
  - `--confidence-threshold` / `--min-confidence`: Set detection confidence
  - `--net-resolution`: Set network input resolution (backend-specific)
  - `--model-pose`: Select pose model (backend-specific)
  - `--overlay-video`: Generate overlay video with pose skeletons
  - `--toronto-gait-format`: Output Toronto gait analysis JSON
  - `--extract-comprehensive-frames`: Save detailed per-frame analysis
  - `--verbose`: Enable detailed logging
- Output formats: JSON, CSV, overlay video (MP4), Toronto gait JSON, comprehensive frame JSON
- Robust error handling and clear installation guidance
- Extensible, OOP, and DRY codebase

## Quick Start

```sh
python cli/detect.py \
    --video path/to/video.mp4 \
    --output outputs/pose_results.json \
    --backend openpose \
    --confidence-threshold 0.3 \
    --net-resolution 656x368 \
    --model-pose COCO \
    --overlay-video outputs/overlay.mp4 \
    --toronto-gait-format \
    --extract-comprehensive-frames \
    --verbose
```

## Backend-Specific Options

| Option                   | MediaPipe | Ultralytics | OpenPose         | AlphaPose        |
|-------------------------|-----------|-------------|------------------|------------------|
| --net-resolution        | (auto)    | yes         | yes              | yes (limited)    |
| --model-pose            | (fixed)   | yes         | yes              | yes              |
| --overlay-video         | yes       | yes         | yes              | yes              |
| --toronto-gait-format   | yes       | yes         | yes              | yes              |
| --extract-comprehensive-frames | yes | yes         | yes              | yes              |

See `docs/ADVANCED_CLI.md` for full details and backend-specific notes.

## Outputs
- **JSON**: Standardized pose data
- **CSV**: Per-joint pose data
- **Overlay video**: MP4 with pose skeletons
- **Toronto gait JSON**: Gait analysis for research
- **Comprehensive frames JSON**: Per-frame statistics and pose quality

## Documentation
- [INSTALL.md](INSTALL.md): Installation and setup
- [OPENPOSE_SETUP.md](OPENPOSE_SETUP.md): OpenPose backend setup
- [ADVANCED_CLI.md](ADVANCED_CLI.md): Full CLI reference and examples
- [BACKEND_SPECIFIC.md](BACKEND_SPECIFIC.md): Backend feature matrix

## Troubleshooting
- See CLI error messages for guidance on missing dependencies or invalid options.
- For backend-specific issues, see the relevant docs in the `docs/` folder.
