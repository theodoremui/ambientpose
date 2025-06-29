"""
AlphaDetect CLI Package

This package provides command-line tools for human pose detection using AlphaPose.
"""

__version__ = "0.1.0"
__author__ = "AlphaDetect Team"
__description__ = "Command-line tools for human pose detection using AlphaPose"

# Import key components to make them available at the package level
from .detect import (
    PoseDetector,
    AlphaDetectConfig,
    parse_args,
)

# Define what's available when using `from cli import *`
__all__ = [
    "PoseDetector",
    "AlphaDetectConfig",
    "parse_args",
]
