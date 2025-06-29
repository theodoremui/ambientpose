"""
AlphaDetect Server Package

This package provides a FastAPI backend for the AlphaDetect pose detection system.
It handles file uploads, task execution via the CLI, status tracking, and file serving.
"""

__version__ = "0.1.0"
__author__ = "AlphaDetect Team"
__description__ = "FastAPI backend for human pose detection using AlphaPose"

# Import key components to make them available at the package level
from .app import (
    app,
    Task,
    TaskStatus,
    TaskResponse,
    TaskExecutor,
    WebSocketManager,
    websocket_manager,
    Settings,
    settings,
)

# Define what's available when using `from server import *`
__all__ = [
    "app",
    "Task",
    "TaskStatus",
    "TaskResponse",
    "TaskExecutor",
    "WebSocketManager",
    "websocket_manager",
    "Settings",
    "settings",
]
