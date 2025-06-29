"""
AlphaDetect Server - FastAPI Backend

This module provides a FastAPI application that serves as the backend for the AlphaDetect
pose detection system. It handles file uploads, task execution, status tracking, and
file serving.

Author: AlphaDetect Team
Date: 2025-06-21
"""

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO

import aiofiles
import aiofiles.os
from fastapi import (
    BackgroundTasks, 
    Depends, 
    FastAPI, 
    File, 
    Form, 
    HTTPException, 
    Query, 
    Request, 
    Response, 
    UploadFile, 
    WebSocket, 
    WebSocketDisconnect,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import AnyHttpUrl, BaseModel, Field, validator
from sqlmodel import Field as SQLField, Session, SQLModel, create_engine, select
from sqlalchemy import JSON
from starlette.websockets import WebSocketState

# Configure logger
import sys
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
logger.add(
    "server_{time}.log",
    rotation="10 MB",
    retention="1 week",
    level="INFO"
)

# Configuration
class Settings:
    """Application settings loaded from environment variables."""
    
    # Base settings
    APP_NAME: str = "AlphaDetect API"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "API for human pose detection using AlphaPose"
    
    # File storage
    STORAGE_DIR: Path = Path(os.getenv("ALPHADETECT_STORAGE_DIR", "outputs"))
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("ALPHADETECT_MAX_UPLOAD_MB", "4096"))
    
    # Database
    DB_URL: str = os.getenv("ALPHADETECT_DB_URL", "sqlite:///./alphadetect.db")
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Frontend dev server
        "http://localhost:8000",  # Backend dev server
        "https://alphadetect.example.com",  # Production
    ]
    
    # CLI
    CLI_PATH: Path = Path(os.getenv("ALPHADETECT_CLI_PATH", "cli/detect.py"))
    
    # Create necessary directories
    @classmethod
    def setup(cls):
        """Create necessary directories if they don't exist."""
        cls.STORAGE_DIR.mkdir(exist_ok=True, parents=True)
        cls.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)


# Initialize settings
settings = Settings()
settings.setup()

# Initialize database
engine = create_engine(settings.DB_URL, echo=False)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving outputs
app.mount("/files", StaticFiles(directory=settings.STORAGE_DIR), name="files")

# Models
class TaskStatus(str, Enum):
    """Enum for task status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class Task(SQLModel, table=True):
    """SQLModel for task data."""
    
    id: str = SQLField(primary_key=True)
    filename: str
    status: TaskStatus = SQLField(default=TaskStatus.PENDING, index=True)
    params: Dict[str, Any] = SQLField(default={}, sa_type=JSON)
    created_at: datetime = SQLField(default_factory=datetime.utcnow, index=True)
    started_at: Optional[datetime] = SQLField(default=None)
    finished_at: Optional[datetime] = SQLField(default=None)
    error: Optional[str] = SQLField(default=None)
    output_json: Optional[str] = SQLField(default=None)
    frames_dir: Optional[str] = SQLField(default=None)
    overlay_dir: Optional[str] = SQLField(default=None)


class TaskCreate(BaseModel):
    """Pydantic model for task creation request."""
    
    filename: str = Field(..., description="Original filename")
    params: Dict[str, Any] = Field(default={}, description="Additional parameters for the task")


class TaskResponse(BaseModel):
    """Pydantic model for task response."""
    
    id: str
    filename: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    output_json: Optional[str] = None
    frames_dir: Optional[str] = None
    overlay_dir: Optional[str] = None
    params: Dict[str, Any] = {}


class TaskList(BaseModel):
    """Pydantic model for task list response."""
    
    tasks: List[TaskResponse]
    total: int
    page: int
    page_size: int


class ErrorResponse(BaseModel):
    """Pydantic model for error response."""
    
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Pydantic model for health check response."""
    
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WebSocketManager:
    """Manager for WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        """Connect a WebSocket to a task."""
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)
        logger.info(f"WebSocket connected to task {task_id}")
    
    def disconnect(self, websocket: WebSocket, task_id: str):
        """Disconnect a WebSocket from a task."""
        if task_id in self.active_connections:
            if websocket in self.active_connections[task_id]:
                self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
        logger.info(f"WebSocket disconnected from task {task_id}")
    
    async def broadcast(self, task_id: str, message: Dict[str, Any]):
        """Broadcast a message to all WebSockets connected to a task."""
        if task_id in self.active_connections:
            disconnected_websockets = []
            for websocket in self.active_connections[task_id]:
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_json(message)
                    except WebSocketDisconnect:
                        disconnected_websockets.append(websocket)
                else:
                    disconnected_websockets.append(websocket)
            
            # Clean up disconnected WebSockets
            for websocket in disconnected_websockets:
                self.disconnect(websocket, task_id)


# Initialize WebSocket manager
websocket_manager = WebSocketManager()


class TaskExecutor:
    """Executor for running CLI tasks."""
    
    @staticmethod
    async def execute_task(task_id: str, file_path: Path, is_video: bool, params: Dict[str, Any]) -> None:
        """Execute a CLI task asynchronously."""
        logger.info(f"Executing task {task_id} with file {file_path}")
        
        # Update task status to RUNNING
        async with AsyncSessionManager() as session:
            task = await get_task(task_id, session)
            if not task:
                logger.error(f"Task {task_id} not found")
                return
            
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            session.add(task)
            await session.commit()
            await session.refresh(task)
        
        # Broadcast task status update
        await websocket_manager.broadcast(
            task_id,
            {"event": "status_update", "task_id": task_id, "status": TaskStatus.RUNNING}
        )
        
        # Prepare CLI command
        cli_args = [
            "python",
            str(settings.CLI_PATH),
        ]
        
        # Add input file argument
        if is_video:
            cli_args.extend(["--video", str(file_path)])
        else:
            cli_args.extend(["--image-dir", str(file_path)])
        
        # Add output directory
        output_dir = settings.STORAGE_DIR / task_id
        output_dir.mkdir(exist_ok=True, parents=True)
        cli_args.extend(["--output-dir", str(output_dir)])
        
        # Add output JSON path
        output_json = output_dir / f"pose_{task_id}.json"
        cli_args.extend(["--output", str(output_json)])
        
        # Add additional parameters
        for key, value in params.items():
            if isinstance(value, bool) and value:
                cli_args.append(f"--{key}")
            elif not isinstance(value, bool):
                cli_args.extend([f"--{key}", str(value)])
        
        # Create subprocess
        process = None
        try:
            logger.info(f"Running command: {' '.join(cli_args)}")
            process = await asyncio.create_subprocess_exec(
                *cli_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024  # 1MB buffer
            )
            
            # Process stdout in real-time
            async def read_stream(stream, is_error=False):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    
                    line_str = line.decode("utf-8").strip()
                    if is_error:
                        logger.error(f"Task {task_id} stderr: {line_str}")
                    else:
                        logger.info(f"Task {task_id} stdout: {line_str}")
                    
                    # Broadcast log line
                    await websocket_manager.broadcast(
                        task_id,
                        {
                            "event": "log",
                            "task_id": task_id,
                            "line": line_str,
                            "is_error": is_error
                        }
                    )
            
            # Start reading streams
            stdout_task = asyncio.create_task(read_stream(process.stdout))
            stderr_task = asyncio.create_task(read_stream(process.stderr, is_error=True))
            
            # Wait for process to complete
            exit_code = await process.wait()
            
            # Wait for stream reading to complete
            await stdout_task
            await stderr_task
            
            # Update task status based on exit code
            async with AsyncSessionManager() as session:
                task = await get_task(task_id, session)
                if not task:
                    logger.error(f"Task {task_id} not found")
                    return
                
                task.finished_at = datetime.utcnow()
                
                if exit_code == 0:
                    task.status = TaskStatus.SUCCESS
                    
                    # Find output directories
                    for path in output_dir.iterdir():
                        if path.is_dir() and path.name.startswith("frames_"):
                            task.frames_dir = path.name
                        elif path.is_dir() and path.name.startswith("overlay_"):
                            task.overlay_dir = path.name
                    
                    # Set output JSON path
                    if output_json.exists():
                        task.output_json = output_json.name
                else:
                    task.status = TaskStatus.FAILED
                    task.error = f"Process exited with code {exit_code}"
                
                session.add(task)
                await session.commit()
                await session.refresh(task)
            
            # Broadcast final status update
            await websocket_manager.broadcast(
                task_id,
                {
                    "event": "status_update",
                    "task_id": task_id,
                    "status": task.status,
                    "output_json": task.output_json,
                    "frames_dir": task.frames_dir,
                    "overlay_dir": task.overlay_dir,
                    "error": task.error
                }
            )
            
            logger.info(f"Task {task_id} completed with status {task.status}")
            
        except asyncio.CancelledError:
            logger.warning(f"Task {task_id} was cancelled")
            
            # Kill process if it's still running
            if process and process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.sleep(1)
                    if process.returncode is None:
                        process.kill()
                except Exception as e:
                    logger.error(f"Error terminating process: {e}")
            
            # Update task status to CANCELED
            async with AsyncSessionManager() as session:
                task = await get_task(task_id, session)
                if task:
                    task.status = TaskStatus.CANCELED
                    task.finished_at = datetime.utcnow()
                    task.error = "Task was cancelled"
                    session.add(task)
                    await session.commit()
            
            # Broadcast cancellation
            await websocket_manager.broadcast(
                task_id,
                {"event": "status_update", "task_id": task_id, "status": TaskStatus.CANCELED}
            )
            
            raise
            
        except Exception as e:
            logger.exception(f"Error executing task {task_id}: {e}")
            
            # Update task status to FAILED
            async with AsyncSessionManager() as session:
                task = await get_task(task_id, session)
                if task:
                    task.status = TaskStatus.FAILED
                    task.finished_at = datetime.utcnow()
                    task.error = str(e)
                    session.add(task)
                    await session.commit()
            
            # Broadcast error
            await websocket_manager.broadcast(
                task_id,
                {"event": "status_update", "task_id": task_id, "status": TaskStatus.FAILED, "error": str(e)}
            )


class AsyncSessionManager:
    """Async context manager for database sessions."""
    
    async def __aenter__(self):
        """Enter the context manager and create a session."""
        self.session = Session(engine)
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close the session."""
        self.session.close()


# Database functions
async def get_task(task_id: str, session: Session) -> Optional[Task]:
    """Get a task by ID."""
    statement = select(Task).where(Task.id == task_id)
    results = session.exec(statement)
    return results.first()


async def get_tasks(
    session: Session,
    status: Optional[TaskStatus] = None,
    skip: int = 0,
    limit: int = 100,
    sort_by: str = "created_at",
    sort_desc: bool = True
) -> List[Task]:
    """Get tasks with optional filtering and sorting."""
    statement = select(Task)
    
    if status:
        statement = statement.where(Task.status == status)
    
    # Add sorting
    if sort_desc:
        statement = statement.order_by(getattr(Task, sort_by).desc())
    else:
        statement = statement.order_by(getattr(Task, sort_by))
    
    statement = statement.offset(skip).limit(limit)
    results = session.exec(statement)
    return results.all()


async def count_tasks(session: Session, status: Optional[TaskStatus] = None) -> int:
    """Count tasks with optional filtering."""
    statement = select(Task)
    
    if status:
        statement = statement.where(Task.status == status)
    
    results = session.exec(statement)
    return len(results.all())


# Dependency for getting database session
async def get_session():
    """Get a database session."""
    async with AsyncSessionManager() as session:
        yield session


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_code=getattr(exc, "error_code", None)
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="An unexpected error occurred",
            error_code="internal_server_error"
        ).dict(),
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    logger.info("Starting up AlphaDetect API")
    
    # Create database tables
    SQLModel.metadata.create_all(engine)
    
    # Create necessary directories
    settings.setup()
    
    logger.info(f"Database URL: {settings.DB_URL}")
    logger.info(f"Storage directory: {settings.STORAGE_DIR}")
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"CLI path: {settings.CLI_PATH}")
    logger.info("AlphaDetect API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Run shutdown tasks."""
    logger.info("Shutting down AlphaDetect API")
    
    # Close database connection
    if engine:
        engine.dispose()
    
    logger.info("AlphaDetect API shut down successfully")


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API is healthy."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION
    )


# Task endpoints
@app.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED, tags=["Tasks"])
async def create_task(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    params: str = Form("{}"),
    session: Session = Depends(get_session)
):
    """Create a new task by uploading a file."""
    # Parse parameters
    try:
        params_dict = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in params field"
        )
    
    # Check file size
    file_size_mb = 0
    chunk_size = 1024 * 1024  # 1MB
    
    # Read the first chunk to get the size
    chunk = await file.read(chunk_size)
    file_size_mb = len(chunk) / (1024 * 1024)
    
    # Reset file position
    await file.seek(0)
    
    if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB}MB"
        )
    
    # Check file extension
    filename = file.filename or "unknown_file"
    file_extension = Path(filename).suffix.lower()
    
    # Check if it's a video or image
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    
    is_video = file_extension in video_extensions
    is_image = file_extension in image_extensions
    
    if not (is_video or is_image):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(video_extensions + image_extensions)}"
        )
    
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # Create upload directory
    upload_dir = settings.UPLOAD_DIR / task_id
    upload_dir.mkdir(exist_ok=True, parents=True)
    
    # Save file
    file_path = upload_dir / filename
    
    async with aiofiles.open(file_path, "wb") as out_file:
        # Write the chunk we already read
        await out_file.write(chunk)
        
        # Continue reading and writing chunks
        while chunk := await file.read(chunk_size):
            await out_file.write(chunk)
    
    # Create task in database
    task = Task(
        id=task_id,
        filename=filename,
        status=TaskStatus.PENDING,
        params=params_dict
    )
    
    session.add(task)
    session.commit()
    session.refresh(task)
    
    # Execute task in background
    background_tasks.add_task(
        TaskExecutor.execute_task,
        task_id=task_id,
        file_path=file_path,
        is_video=is_video,
        params=params_dict
    )
    
    return task


@app.get("/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
async def get_task_status(task_id: str, session: Session = Depends(get_session)):
    """Get the status of a task."""
    task = await get_task(task_id, session)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    return task


@app.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Tasks"])
async def delete_task(task_id: str, session: Session = Depends(get_session)):
    """Delete a task and its associated files."""
    task = await get_task(task_id, session)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    # Delete task from database
    session.delete(task)
    session.commit()
    
    # Delete task files
    task_dir = settings.STORAGE_DIR / task_id
    if task_dir.exists():
        shutil.rmtree(task_dir)
    
    # Delete upload files
    upload_dir = settings.UPLOAD_DIR / task_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/tasks", response_model=TaskList, tags=["Tasks"])
async def list_tasks(
    status: Optional[TaskStatus] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    sort_by: str = Query("created_at", regex="^(created_at|status|filename)$"),
    sort_desc: bool = Query(True),
    session: Session = Depends(get_session)
):
    """List tasks with optional filtering and pagination."""
    skip = (page - 1) * page_size
    
    tasks = await get_tasks(
        session=session,
        status=status,
        skip=skip,
        limit=page_size,
        sort_by=sort_by,
        sort_desc=sort_desc
    )
    
    total = await count_tasks(session=session, status=status)
    
    return TaskList(
        tasks=tasks,
        total=total,
        page=page,
        page_size=page_size
    )


@app.post("/tasks/{task_id}/cancel", response_model=TaskResponse, tags=["Tasks"])
async def cancel_task(task_id: str, session: Session = Depends(get_session)):
    """Cancel a running task."""
    task = await get_task(task_id, session)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    if task.status != TaskStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task {task_id} is not running"
        )
    
    # Update task status
    task.status = TaskStatus.CANCELED
    task.finished_at = datetime.utcnow()
    task.error = "Task was cancelled by user"
    
    session.add(task)
    session.commit()
    session.refresh(task)
    
    # Broadcast cancellation
    await websocket_manager.broadcast(
        task_id,
        {"event": "status_update", "task_id": task_id, "status": TaskStatus.CANCELED}
    )
    
    return task


@app.get("/files/{task_id}/{file_path:path}", tags=["Files"])
async def get_file(task_id: str, file_path: str, session: Session = Depends(get_session)):
    """Get a file from a task."""
    task = await get_task(task_id, session)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    # Build file path
    full_path = settings.STORAGE_DIR / task_id / file_path
    
    # Check if file exists
    if not full_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {file_path} not found for task {task_id}"
        )
    
    # Check if file is within storage directory
    if not full_path.resolve().is_relative_to(settings.STORAGE_DIR.resolve()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to this file is forbidden"
        )
    
    # Return file
    return FileResponse(full_path)


@app.websocket("/ws/tasks/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str, session: Session = Depends(get_session)):
    """WebSocket endpoint for real-time task updates."""
    # Check if task exists
    task = await get_task(task_id, session)
    
    if not task:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    # Accept connection
    await websocket_manager.connect(websocket, task_id)
    
    try:
        # Send initial task status
        await websocket.send_json({
            "event": "status_update",
            "task_id": task_id,
            "status": task.status,
            "output_json": task.output_json,
            "frames_dir": task.frames_dir,
            "overlay_dir": task.overlay_dir,
            "error": task.error
        })
        
        # Keep connection open and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                # Currently, we don't process any incoming messages
            except WebSocketDisconnect:
                websocket_manager.disconnect(websocket, task_id)
                break
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket, task_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
