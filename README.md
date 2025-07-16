# AmbientPose

> A project by Theodore Mui in collaboration with Shardul Sapkota
> Ambient Intelligence Project in Stanford Institute for Human-centric AI
> under the guidance of Professor James Landay.

**Modern end-to-end human-pose detection platform powered by [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), OpenPose, MediaPipe, and Ultralytics YOLO.**  
AmbientPose offers a production-ready CLI, an asynchronous FastAPI backend, and a sleek frontend that work together to transform videos or image sequences into rich pose-estimation data and visualisations.

---

## 1.  Project Overview
AmbientPose is designed for researchers, engineers, and creators to extract accurate multi-person, whole-body joint positions from media with minimal effort. It wraps state-of-the-art pose engines in a highly-extensible micro-service architecture, delivering:

* **Batch & real-time inference** from the command line or REST/WebSocket API  
* **Interactive web UI** for project management, uploads, and result exploration  
* **Scalable deployment** via Docker/Kubernetes with optional GPU workers  
* **Rich artefacts** — structured JSON, raw frames, overlays, and CSV exports
* **Multi-backend support** — AlphaPose, MediaPipe, and Ultralytics YOLO

---

## 2.  Key Features & Capabilities
| Area | Highlights |
|------|------------|
| **Detection Backends** | AlphaPose (full-body, 136 kp, tracking, GPU/CPU), MediaPipe (fast, CPU), Ultralytics YOLO (YOLOv8-pose) |
| **Inputs** | Local videos, image folders, or remote URLs |
| **Outputs** | Pose JSON · `frames_*` raw dumps · `overlay_*` annotated frames · CSV export |
| **API** | Async FastAPI, Swagger/OpenAPI, WebSocket log/progress streaming, task queue, cancellation |
| **UI** | Next.js 14 (App Router), Tailwind CSS, dark/light, drag-and-drop uploads, dashboard, gallery, stats |
| **Extensibility** | Modular Python, typed interfaces, plug-in detector architecture |
| **DevOps** | Loguru logging, SQLModel ORM, pre-commit, 90 %+ test coverage, Docker/K8s profiles |
| **Advanced** | Person tracking (when supported), optional 3D pose, progress streaming, error handling |

---

## 3.  Supported Backends
AmbientPose supports multiple pose estimation engines, automatically selecting the best available or letting you choose:

- **AlphaPose**: State-of-the-art, 136-keypoint, multi-person, tracking, best accuracy, GPU/CPU, optional 3D pose.
- **MediaPipe**: Fast, reliable, CPU-friendly, ideal for quick or resource-limited jobs.
- **Ultralytics YOLO**: YOLOv8-pose, fast, alternative backend for pose estimation.

> The CLI and API will use the best available backend by default, or you can specify one with `--backend`.

---

## 4.  System Architecture

```
┌────────────┐  HTTPS/WS   ┌──────────────┐  asyncio/pipe   ┌─────────────┐
│  Frontend  │◀───────────▶│   FastAPI    │◀───────────────▶│ detect.py   │
│  Next.js   │             │    API       │   stdio/json    │     CLI     │
│            │             │              │                 │             │
│            │             │              │                 │             │
└────────────┘             └──────────────┘                 └─────────────┘
        ▲                                                          │
        └────────────────────  outputs/  ──────────────────────────┘
```

All artefacts live under `outputs/<taskId>/` for easy browsing and download.

---

## 5.  Quick-Start (Local)

```bash
git clone https://github.com/your-org/ambientpose.git
cd ambientpose

# ①  Python env & deps  (powered by ultra-fast "uv")
uv venv                                # creates .venv using the Python in .python-version
uv pip install -e ".[dev]"            # CLI + server (CPU-only)
#   └─  add ,gpu  extra for CUDA builds:  uv pip install -e ".[dev,gpu]"

# ② Front-end deps
npm install --prefix frontend

# ②.5 Install AlphaPose core (requires NumPy already installed)
uv pip install "alphapose @ git+https://github.com/MVIG-SJTU/AlphaPose"

# ④ Download AlphaPose weights (≈200 MB)
make download-models                  # or see docs/INSTALL.md

# ⑤ Run services
uvicorn server.app:app --reload       # http://localhost:8000
npm run dev --prefix frontend         # http://localhost:3000
```

---

## 6.  Usage Examples

### CLI  

```bash
# Video
python cli/detect.py --video demo.mp4 --backend alphapose

# Image folder with explicit output file
python cli/detect.py --image-dir samples/frames --output outputs/pose_run1.json --backend mediapipe

# Use YOLO backend
python cli/detect.py --video demo.mp4 --backend ultralytics
```

### API  

```bash
# Create task
curl -F file=@demo.mp4 http://localhost:8000/tasks

# Poll status
curl http://localhost:8000/tasks/<id>
```

Swagger / Redoc automatically available at `http://localhost:8000/docs`.

### Frontend  

1. Open `http://localhost:3000`  
2. **Upload** media via drag-and-drop  
3. Watch live progress and explore results in the gallery  

---

## 7.  API Documentation

* **Swagger / OpenAPI**: `GET /docs`  
* **WebSocket**: `ws://<host>/ws/tasks/{taskId}` for live logs & status  
* Python client examples in [`docs/API.md`](docs/API.md) *(WIP)*

---

## 8.  Frontend Highlights

* Responsive App-Router layout with breadcrumbs & dark mode  
* Project dashboard, task monitor, overlay gallery & stats widgets  
* Tool-tips, onboarding quick-start and progressive-disclosure settings  
* Built with Tailwind, Bootstrap Icons, Headless UI & SWR hooks  

See full design rationale in [`frontend/docs/DESIGN.md`](frontend/docs/DESIGN.md).

---

## 9.  Extensibility & DevOps

- **Modular, typed Python code** (SOLID, plug-in detector architecture)
- **Pre-commit hooks, CI-ready pytest suite, 90%+ test coverage**
- **Docker/Kubernetes support**:
  - Compose profiles for CPU, GPU, and production
  - Optional GPU worker pool for high-throughput
- **Easy model downloads and environment setup** (Makefile/scripts)
- **Rich logging and error handling** (Loguru)
- **Progress streaming for UI/API clients**

---

## 10.  Deployment & Integration

- **Local, containerized, or cloud deployment**
- **REST/WS API for integration with other systems**
- **Secure artefact serving and CORS configuration**
- **Task metadata stored in SQLite (pluggable)**

---

## 11.  Contributing

We 💙 contributions!  Please:

1. Fork & create feature branch (`feat/my-feature`)  
2. Ensure `make test` passes and add tests for new behaviour  
3. Follow [Conventional Commits](https://www.conventionalcommits.org/)  
4. Open a PR – the CI will run linting, typing and coverage gates

Read `docs/CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` for details.

---

## 12.  License

AmbientPose is licensed under the **MIT License** – see [`LICENSE`](LICENSE).  
Note: AlphaPose models are released for **non-commercial research**; commercial usage may require separate permission from the original authors.

---

## 13. Running the Streamlit Frontend

To launch the interactive pose detection frontend, use the following command from the project root:

```sh
streamlit run frontend/app.py
```

- On Windows, you can also use:
  ```sh
  streamlit run .\frontend\app.py
  ```
- Make sure your virtual environment is activated and all dependencies are installed (see below).

**Troubleshooting:**
- If you see an error like `stream : The term 'stream' is not recognized...`, you may have mistyped the command. The correct command is `streamlit run`, not `stream run`.
- To verify Streamlit is installed, run:
  ```sh
  streamlit --version
  ```

---
## 14.  Acknowledgments

* Professor James Landay -- for his wise guidance and support
* Institute of Human Centered AI

> Made with passion by the AmbientPose engineering team.  
> To contact the team, please send email to: Theodore Mui <theodoremui@gmail.com>
