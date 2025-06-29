# AlphaDetect – Design Document

*Document version: 1.0 – 2025-06-21*

---

## 1. System Architecture Overview

AlphaDetect is a three–tier system:

| Layer     | Technology                                           | Responsibility                                      |
|-----------|-------------------------------------------------------|-----------------------------------------------------|
| **UI**    | Next.js 14, React 18, Tailwind CSS, Bootstrap Icons  | Upload media, monitor jobs, visualise output        |
| **API**   | FastAPI 0.111, Uvicorn/Hypercorn, Python 3.10 asyncio | Task orchestration, authentication, artefact serve  |
| **Worker**| Python CLI (`detect.py`) wrapping AlphaPose + FFmpeg  | Heavy-weight pose estimation and artefact creation  |

Deployment topologies:

* **Local monolith** – All layers on the same host (dev/CI).
* **Micro-services** – API & CLI in container(s), UI on CDN (prod).
* **GPU pool** – Multiple worker pods behind a queue for high load.

### High-Level Diagram

```
Browser ──► Next.js ► REST/WS ► FastAPI API ──► asyncio subprocess ▶ detect.py ▶ AlphaPose ▶ GPU/CPU
   ▲                                                   │
   └──── S3/GCS/Local downloads ◄──── File Server ◄────┘
```

---

## 2. Component Design Decisions

### 2.1 CLI (`cli/detect.py`)
* **Interface**: argparse; supports `--video`, `--image-dir`, `--output`, detector & tracking flags.
* **AlphaPose Integration**: Uses the high-level API (`scripts/demo_api.py`) to avoid forked code.
* **Outputs**: JSON summary + two deterministic frame folders (`frames_*`, `overlay_*`) under `/outputs`.
* **Logging**: `loguru`; progress bars converted to newline logs for API streaming.

### 2.2 API (`server/`)
* **Async subprocess**: `asyncio.create_subprocess_exec` with stdout/stderr piping; back-pressure via bounded queues.
* **Task tracking**: SQLite via SQLModel; status enum (PENDING, RUNNING, SUCCESS, FAILED, CANCELED).
* **File storage**: Local disk by default; pluggable adapter (e.g. S3). Artefact path is stored in DB not returned directly.
* **Background execution**: Optional Celery/RQ adapter kept abstracted behind `TaskExecutor` interface.

### 2.3 Frontend (`frontend/`)
* **App Router**: `/projects`, `/tasks/[id]`, `/results/[taskId]`.
* **Data access**: SWR hooks + Axios; incremental static regeneration for public artefacts.
* **Design system**: Tailwind for layout; Bootstrap Icons for familiarity.
* **Responsiveness**: Mobile-first; media uploads use `tus-js-client` for resumable uploads >2 GB.

---

## 3. Data Flow Diagrams

### 3.1 Upload & Processing

```
User ─▶ POST /tasks (multipart/form-data) 
     └─┬─> API stores file (disk/S3)
       ├─> DB row status=PENDING
       └─> spawn detect.py
               detect.py reads video
               AlphaPose inference
               detect.py writes artefacts → /outputs/<taskId>/*
               detect.py stdout JSON logs
 API streams logs via /tasks/{id}/stream (SSE/WS)
 On exit ➜ DB status=SUCCESS | FAILED
```

### 3.2 Download Results

```
Browser ─▶ GET /files/{taskId}/pose.json → JSON
Browser ─▶ GET /files/{taskId}/overlay/frame_000123.jpg → image/jpeg
```

---

## 4. API Design

| Method | Path | Body / Params | Response | Notes |
|--------|------|---------------|----------|-------|
| POST | `/tasks` | multipart `file`, opt `config` json | `{id,status}` | Accept URL or local upload |
| GET | `/tasks/{id}` | – | Task DTO | Poll status |
| GET | `/tasks/{id}/stream` | – | SSE/WS stream | Live logs/events |
| GET | `/files/{id}/{path}` | – | Binary | Signed URL optional |
| DELETE | `/tasks/{id}` | – | 204 | Cancels running task & deletes artefacts |

DTO (simplified):

```ts
interface Task {
  id: string;
  filename: str;
  status: 'PENDING'|'RUNNING'|'SUCCESS'|'FAILED';
  created_at: datetime;
  finished_at?: datetime;
  metrics?: PoseMetrics;
  error?: str;
}
```

---

## 5. Database Schema (SQLModel)

```python
class Task(SQLModel, table=True):
    id: str = Field(primary_key=True)
    filename: str
    status: TaskStatus
    params: dict | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    metrics: dict | None = None
    error: str | None = None
```

Indices: `status`, `created_at`.

Future-proof for Postgres by keeping SQL-portable types.

---

## 6. Security Considerations

| Concern | Mitigation |
|---------|------------|
| Arbitrary file execution | Whitelist file extensions; scan MIME; store outside static path. |
| Path traversal | `pathlib.Path.resolve().is_relative_to(BASE_DIR)` check. |
| DoS (large uploads) | Limit size (env var), resumable uploads, background queue. |
| API auth | JWT (access/refresh) with [fastapi-users]; optional API keys for headless. |
| CSRF/XSS | Next.js uses same-site cookies; all user input escaped. |
| Secrets | `.env` loaded via `python-dotenv`; never commit secrets. |

---

## 7. Performance Considerations

* **GPU utilisation**: Batching disabled (AlphaPose per-frame); use TensorRT in roadmap.
* **I/O**: `mmap` video decoding via PyAV; frames streamed to GPU without disk round-trip.
* **Concurrency**: Each worker process CPU-bound on decoding + GPU-bound inference – run 1 per-GPU.
* **HTTP**: GZip + HTTP/2 on Uvicorn/H2O; chunked responses for overlay galleries.

---

## 8. Scalability & Extensibility

* Stateless API → horizontally scalable behind a load balancer.
* `TaskExecutor` interface allows swap-in of Kubernetes Jobs, SQS/Lambda, or on-prem SLURM.
* Storage adapters (LocalDisk, S3, GCS) implement a common `FileBackend`.
* Plugin registry for alternative detectors (e.g. OpenPifPaf).

---

## 9. Error Handling Strategy

| Layer | Strategy |
|-------|----------|
| CLI | Catch all exceptions; log traceback; exit code !=0; write `.error` file. |
| API | Pydantic/Validation errors → 422; Internal → 500 with UUID error code and sanitized message. |
| Frontend | SWR `onErrorRetry` exponential backoff; toast notifications; graceful downgrades. |
| Propagation | Errors bubble up to Task status=FAILED with message for UI display. |

---

## 10. Testing Strategy

* **Unit tests** (`pytest`, `pytest-asyncio`):  
  * CLI argument parsing, output writer, log parser  
  * API route validation, auth, RBAC
* **Integration tests**:  
  * Spawn `detect.py` with stub ― dry-run mode generating fake artefacts.  
  * Use `TestClient` to submit job and assert status flow.
* **e2e**: Playwright script uploads sample video and checks overlay image exists.
* **Coverage target**: 90 % lines, 100 % critical paths.
* **CI**: GitHub Actions matrix – Python 3.10/3.11 on Ubuntu + GPU job (self-hosted runner).

---

## 11. Deployment Considerations

### 11.1 Local Dev

```bash
docker compose up              # api + db + worker
npm run dev --prefix frontend  # ui
```

### 11.2 Production (Kubernetes)

```text
alphadetect-frontend   Deployment (replicas=3)  —> Cloud CDN
alphadetect-api        Deployment (replicas=3)  —> HPA CPU=60%
alphadetect-worker     Job per task / GPU node  —> NodeSelector nvidia.com/gpu.present
postgres               StatefulSet
minio                  StatefulSet (optional)
ingress-nginx → TLS via cert-manager
```

* Zero-downtime via rolling updates.  
* Observability: Prometheus exporter, Loki for logs, Jaeger for traces.  
* Feature flags with Unleash.

---

### Appendix A – Future Roadmap

| Feature | Rationale |
|---------|-----------|
| WebSocket live overlay | Stream inference frames to UI in real-time |
| TRT/Torch-TensorRT | 4× speedup on T4 GPUs |
| Multi-camera sync | Sports analytics use-case |
