#!/usr/bin/env python3
"""
AmbientPose CLI - Human Pose Detection Tool

This tool provides pose detection capabilities using multiple backends:
- MediaPipe (primary, most reliable)
- Ultralytics YOLO (alternative)
- AlphaPose (if available)
- OpenPose (if available: check env var OPENPOSE_HOME)

Author: Theodore Mui
Email: theodoremui@gmail.com
Date: 2025-06-22
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

# Add AlphaPose to Python path if it exists
alphapose_path = Path(__file__).parent.parent / "AlphaPose"
if alphapose_path.exists():
    sys.path.insert(0, str(alphapose_path))
    logger.info(f"Added AlphaPose path: {alphapose_path}")

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:MM-DD HH:mm:ss}</green>|<level>{level: <5}</level>|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)


# Check available backends
MEDIAPIPE_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False
ALPHAPOSE_AVAILABLE = False
OPENPOSE_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe backend available")
except ImportError:
    logger.warning("MediaPipe not available")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    logger.info("Ultralytics YOLO backend available")
except ImportError:
    logger.warning("Ultralytics YOLO not available")

try:
    import torch
    from alphapose.models import builder
    from alphapose.utils.config import update_config
    from alphapose.utils.detector import DetectionLoader
    from alphapose.utils.pPose_nms import pose_nms
    
    # Try to import YOLOX detector, fall back to alternatives if not available
    try:
        from detector.yolox.detector import YoloxDetector
    except ImportError:
        try:
            # Alternative import path
            from detector.yolox_api import YOLOXDetector as YoloxDetector
        except ImportError:
            try:
                # Another alternative - use the API directly
                from detector.apis import get_detector
                YoloxDetector = None
                logger.debug("YOLOX detector not available, will use fallback")
            except ImportError:
                YoloxDetector = None
                logger.debug("No YOLOX detector available")
    
    ALPHAPOSE_AVAILABLE = True
    logger.info("AlphaPose backend available")
except ImportError as e:
    ALPHAPOSE_AVAILABLE = False
    logger.warning(f"AlphaPose not available: {e}")

# Check OpenPose availability
try:
    # Check if OPENPOSE_HOME environment variable is set
    openpose_home = os.environ.get('OPENPOSE_HOME')
    if openpose_home:
        openpose_home_path = Path(openpose_home)
        openpose_bin_path = openpose_home_path / "bin"
        
        # Check if bin directory exists
        if openpose_bin_path.exists():
            # Look for OpenPose executable or Python bindings
            # Try different possible binary names and locations
            possible_binaries = [
                openpose_bin_path / "OpenPoseDemo.exe",  # Windows
                openpose_bin_path / "examples" / "openpose" / "openpose.bin",  # Linux
                openpose_bin_path / "openpose",  # Alternative Linux
            ]
            
            # Check for Python bindings (preferred)
            python_binding_paths = [
                openpose_home_path / "build" / "python" / "openpose",
                openpose_home_path / "python" / "openpose",
                openpose_bin_path / "python",
            ]
            
            openpose_found = False
            
            # First try to import Python bindings
            for python_path in python_binding_paths:
                if python_path.exists():
                    try:
                        # Add to Python path temporarily to test import
                        temp_path = str(python_path.parent)
                        if temp_path not in sys.path:
                            sys.path.insert(0, temp_path)
                        
                        import openpose as op
                        OPENPOSE_AVAILABLE = True
                        openpose_found = True
                        logger.info(f"OpenPose Python bindings available at: {python_path}")
                        break
                    except ImportError:
                        continue
            
            # If Python bindings not available, check for binary executable
            if not openpose_found:
                for binary_path in possible_binaries:
                    if binary_path.exists():
                        OPENPOSE_AVAILABLE = True
                        openpose_found = True
                        logger.info(f"OpenPose binary available at: {binary_path}")
                        break
            
            if not openpose_found:
                logger.warning(f"OpenPose not available: No valid binaries or Python bindings found in {openpose_home}")
        else:
            logger.warning(f"OpenPose not available: bin directory not found at {openpose_bin_path}")
    else:
        logger.warning("OpenPose not available: OPENPOSE_HOME environment variable not set")
        
except Exception as e:
    OPENPOSE_AVAILABLE = False
    logger.warning(f"OpenPose availability check failed: {e}")


class AmbientPoseConfig:
    """
    Configuration for AmbientPose CLI.

    Handles all CLI options, advanced parameters, validation, and logging.
    Supports backend-specific defaults and overrides for:
      - net_resolution
      - model_pose
      - overlay_video_path
      - toronto_gait_format
      - extract_comprehensive_frames
      - verbose/debug
    """
    
    # Default paths
    DEFAULT_OUTPUT_DIR = Path("outputs")
    
    # Backend-specific default configurations
    BACKEND_DEFAULTS = {
        'mediapipe': {
            'net_resolution': '256x256',
            'model_pose': 'POSE_LANDMARKS',
        },
        'ultralytics': {
            'net_resolution': '640x640',
            'model_pose': 'yolov8n-pose.pt',
        },
        'openpose': {
            'net_resolution': '656x368',
            'model_pose': 'BODY_25',
        },
        'alphapose': {
            'net_resolution': '256x192',
            'model_pose': 'COCO',
        }
    }
    
    def __init__(self, args: argparse.Namespace):
        """Initialize configuration from parsed arguments."""
        self.input_path: Path = Path(args.video) if args.video else Path(args.image_dir)
        self.is_video: bool = args.video is not None
        
        # Create output directory if it doesn't exist
        self.output_dir: Path = Path(args.output_dir) if args.output_dir else self.DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate timestamp for output files
        self.timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set output JSON path
        if args.output:
            self.output_json: Path = Path(args.output)
        else:
            filename = f"pose_{self.timestamp}.json"
            self.output_json = self.output_dir / filename
        
        # Create frame and overlay directories
        input_name = self.input_path.stem
        self.frames_dir: Path = self.output_dir / f"frames_{input_name}_{self.timestamp}"
        self.overlay_dir: Path = self.output_dir / f"overlay_{input_name}_{self.timestamp}"
        
        self.frames_dir.mkdir(exist_ok=True)
        self.overlay_dir.mkdir(exist_ok=True)
        
        # Basic configuration
        self.backend: str = args.backend
        self.debug: bool = args.debug
        self.min_confidence: float = args.min_confidence
        
        # Advanced configuration options
        self.net_resolution: Optional[str] = args.net_resolution
        self.model_pose: Optional[str] = args.model_pose
        self.overlay_video_path: Optional[str] = args.overlay_video
        self.toronto_gait_format: bool = args.toronto_gait_format
        self.extract_comprehensive_frames: bool = args.extract_comprehensive_frames
        self.verbose: bool = args.verbose
        
        # Set up logging based on debug and verbose flags
        self._setup_logging()
        
        # Validate configuration
        self._validate()
        
        logger.info(f"Configuration initialized: backend={self.backend}, input={self.input_path}")
        if self.verbose:
            self._log_configuration_details()
    
    def _setup_logging(self) -> None:
        """Configure logging based on debug and verbose flags."""
        if self.debug:
            logger.remove()
            logger.add(
                sys.stderr,
                format="<green>{time:MM-DD HH:mm:ss}</green>|<level>{level: <5}</level>|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level="DEBUG"
            )
        elif self.verbose:
            logger.remove()
            logger.add(
                sys.stderr,
                format="<green>{time:MM-DD HH:mm:ss}</green>|<level>{level: <5}</level>|<cyan>{name}</cyan> - <level>{message}</level>",
                level="INFO"
            )
    
    def _log_configuration_details(self) -> None:
        """Log detailed configuration information in verbose mode."""
        logger.info("=== AmbientPose Configuration Details ===")
        logger.info(f"Input: {self.input_path} ({'video' if self.is_video else 'images'})")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Output JSON: {self.output_json}")
        logger.info(f"Backend: {self.backend}")
        logger.info(f"Confidence threshold: {self.min_confidence}")
        
        if self.net_resolution:
            logger.info(f"Network resolution: {self.net_resolution}")
        if self.model_pose:
            logger.info(f"Pose model: {self.model_pose}")
        if self.overlay_video_path:
            logger.info(f"Overlay video: {self.overlay_video_path}")
        if self.toronto_gait_format:
            logger.info("Toronto gait format: ENABLED")
        if self.extract_comprehensive_frames:
            logger.info("Comprehensive frame extraction: ENABLED")
        
        logger.info("=" * 50)
    
    def get_backend_config(self, backend_name: str) -> Dict[str, Any]:
        """Get configuration parameters for a specific backend."""
        # Start with defaults for the backend
        backend_config = self.BACKEND_DEFAULTS.get(backend_name, {}).copy()
        
        # Override with user-specified values
        if self.net_resolution:
            backend_config['net_resolution'] = self.net_resolution
        if self.model_pose:
            backend_config['model_pose'] = self.model_pose
            
        # Add common configuration
        backend_config.update({
            'min_confidence': self.min_confidence,
            'overlay_video_path': self.overlay_video_path,
            'toronto_gait_format': self.toronto_gait_format,
            'extract_comprehensive_frames': self.extract_comprehensive_frames,
            'verbose': self.verbose,
            'debug': self.debug,
        })
        
        return backend_config
    
    def _validate_net_resolution(self, resolution: str) -> bool:
        """Validate network resolution format (WIDTHxHEIGHT)."""
        if not resolution:
            return True
        
        import re
        pattern = r'^\d+x\d+$'
        if not re.match(pattern, resolution):
            return False
        
        try:
            width, height = map(int, resolution.split('x'))
            return width > 0 and height > 0 and width <= 4096 and height <= 4096
        except (ValueError, AttributeError):
            return False
    
    def _validate(self) -> None:
        """Validate the configuration."""
        # Check if input exists
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
        
        # Check if output directory is writable
        if not os.access(self.output_dir, os.W_OK):
            raise PermissionError(f"Output directory is not writable: {self.output_dir}")
        
        # Validate network resolution format
        if self.net_resolution and not self._validate_net_resolution(self.net_resolution):
            raise ValueError(f"Invalid network resolution format: {self.net_resolution}. Expected format: WIDTHxHEIGHT (e.g., 656x368)")
        
        # Validate confidence threshold
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got: {self.min_confidence}")
        
        # Validate overlay video path if specified
        if self.overlay_video_path:
            overlay_path = Path(self.overlay_video_path)
            # Create parent directory if it doesn't exist
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if we can write to the directory
            if not os.access(overlay_path.parent, os.W_OK):
                raise PermissionError(f"Cannot write overlay video to: {overlay_path.parent}")
        
        # Log warnings for advanced options that may not be supported by all backends
        if self.verbose:
            if self.net_resolution:
                logger.info(f"Network resolution override: {self.net_resolution}")
            if self.model_pose:
                logger.info(f"Model pose override: {self.model_pose}")
            if self.toronto_gait_format:
                logger.info("Toronto gait format enabled")
            if self.extract_comprehensive_frames:
                logger.info("Comprehensive frame extraction enabled")


class MediaPipeDetector:
    """MediaPipe-based pose detector."""
    
    def __init__(self, config: AmbientPoseConfig):
        """Initialize MediaPipe pose detector."""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is not available")
        
        self.config = config
        
        # Get backend-specific configuration
        self.backend_config = config.get_backend_config('mediapipe')
        
        # MediaPipe doesn't have configurable network resolution or model selection
        # But we can adjust model complexity based on resolution hint
        model_complexity = 1  # Default
        if config.net_resolution:
            try:
                width, height = map(int, config.net_resolution.split('x'))
                if width <= 256 and height <= 256:
                    model_complexity = 0  # Lightweight
                elif width >= 512 and height >= 512:
                    model_complexity = 2  # Full model
            except (ValueError, AttributeError):
                logger.warning(f"Invalid net-resolution for MediaPipe: {config.net_resolution}, using default")
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=config.min_confidence,
            min_tracking_confidence=config.min_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        if config.verbose:
            logger.info(f"MediaPipe pose detector initialized (model_complexity={model_complexity})")
            if config.net_resolution:
                logger.info(f"Note: MediaPipe doesn't support custom resolutions, using model_complexity adjustment")
            if config.model_pose and config.model_pose != 'POSE_LANDMARKS':
                logger.warning(f"MediaPipe only supports POSE_LANDMARKS model, ignoring: {config.model_pose}")
        else:
            logger.info("MediaPipe pose detector initialized")
    
    def detect_poses(self, image: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """Detect poses in a single image."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(rgb_image)
        
        poses = []
        if results.pose_landmarks:
            # Convert landmarks to our format
            landmarks = results.pose_landmarks.landmark
            height, width = image.shape[:2]
            
            # Calculate bounding box
            x_coords = [lm.x * width for lm in landmarks]
            y_coords = [lm.y * height for lm in landmarks]
            
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            
            # Add padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            # Convert landmarks to keypoints
            keypoints = []
            for lm in landmarks:
                x, y = lm.x * width, lm.y * height
                confidence = lm.visibility if hasattr(lm, 'visibility') else 1.0
                keypoints.append([x, y, confidence])
            
            pose = {
                'frame_idx': frame_idx,
                'bbox': [x1, y1, x2, y2, 1.0],  # confidence = 1.0 for MediaPipe
                'score': 1.0,
                'keypoints': keypoints,
                'backend': 'mediapipe'
            }
            poses.append(pose)
        
        return poses
    
    def draw_poses(self, image: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
        """Draw poses on the image."""
        if not poses:
            return image
        
        # Convert to RGB for MediaPipe drawing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for pose in poses:
            # Draw bounding box
            bbox = pose['bbox']
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Convert keypoints back to MediaPipe format for drawing
            if len(pose['keypoints']) == 33:  # MediaPipe has 33 landmarks
                # Create a mock landmark list
                height, width = image.shape[:2]
                landmark_list = []
                
                for x, y, conf in pose['keypoints']:
                    # Create a simple object with x, y attributes
                    class MockLandmark:
                        def __init__(self, x, y):
                            self.x = x / width
                            self.y = y / height
                    
                    landmark_list.append(MockLandmark(x, y))
                
                # Draw pose connections
                self._draw_landmarks(image, landmark_list)
        
        return image
    
    def _draw_landmarks(self, image: np.ndarray, landmarks: List) -> None:
        """Draw pose landmarks and connections."""
        # MediaPipe pose connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
            (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
        ]
        
        height, width = image.shape[:2]
        
        # Draw landmarks
        for landmark in landmarks:
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
        
        # Draw connections
        for connection in connections:
            if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                start_landmark = landmarks[connection[0]]
                end_landmark = landmarks[connection[1]]
                
                start_x, start_y = int(start_landmark.x * width), int(start_landmark.y * height)
                end_x, end_y = int(end_landmark.x * width), int(end_landmark.y * height)
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)


class UltralyticsDetector:
    """Ultralytics YOLO-based pose detector."""
    
    def __init__(self, config: AmbientPoseConfig):
        """Initialize Ultralytics YOLO pose detector."""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics YOLO is not available")
        
        self.config = config
        
        # Get backend-specific configuration
        self.backend_config = config.get_backend_config('ultralytics')
        
        # Select model based on user preference or default
        model_name = config.model_pose if config.model_pose else 'yolov8n-pose.pt'
        
        # Validate and adjust model name for Ultralytics
        valid_models = ['yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt']
        if model_name not in valid_models:
            if config.verbose:
                logger.warning(f"Model '{model_name}' not in standard Ultralytics models: {valid_models}")
                logger.info(f"Attempting to use '{model_name}' as custom model")
        
        # Store network resolution for inference
        self.net_resolution = None
        if config.net_resolution:
            try:
                self.net_resolution = tuple(map(int, config.net_resolution.split('x')))
                if config.verbose:
                    logger.info(f"Will use custom inference resolution: {self.net_resolution}")
            except (ValueError, AttributeError):
                logger.warning(f"Invalid net-resolution for Ultralytics: {config.net_resolution}, using default")
                self.net_resolution = None
        
        try:
            self.model = YOLO(model_name)
            if config.verbose:
                logger.info(f"Ultralytics YOLO pose detector initialized with model: {model_name}")
                if self.net_resolution:
                    logger.info(f"Custom inference resolution: {self.net_resolution[0]}x{self.net_resolution[1]}")
            else:
                logger.info("Ultralytics YOLO pose detector initialized")
        except Exception as e:
            logger.warning(f"Failed to load model '{model_name}': {e}")
            logger.info("Falling back to default yolov8n-pose.pt")
            self.model = YOLO('yolov8n-pose.pt')
            logger.info("Ultralytics YOLO pose detector initialized with fallback model")
    
    def detect_poses(self, image: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """Detect poses in a single image."""
        # Run YOLO inference with custom resolution if specified
        inference_args = {'verbose': False}
        if self.net_resolution:
            inference_args['imgsz'] = self.net_resolution
        
        results = self.model(image, **inference_args)
        
        poses = []
        for result in results:
            if result.keypoints is not None:
                boxes = result.boxes
                keypoints = result.keypoints
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    kpts = keypoints[i]
                    
                    # Extract box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    box_conf = box.conf[0].cpu().numpy()
                    
                    # Skip low-confidence detections
                    if box_conf < self.config.min_confidence:
                        continue
                    
                    # Extract keypoints
                    kpts_data = kpts.data[0].cpu().numpy()  # Shape: (17, 3) for COCO format
                    keypoints_list = []
                    
                    for j in range(len(kpts_data)):
                        x, y, conf = kpts_data[j]
                        keypoints_list.append([float(x), float(y), float(conf)])
                    
                    pose = {
                        'frame_idx': frame_idx,
                        'bbox': [float(x1), float(y1), float(x2), float(y2), float(box_conf)],
                        'score': float(box_conf),
                        'keypoints': keypoints_list,
                        'backend': 'ultralytics'
                    }
                    poses.append(pose)
        
        return poses
    
    def draw_poses(self, image: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
        """Draw poses on the image."""
        if not poses:
            return image
        
        # COCO skeleton connections (17 keypoints)
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms  
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Colors for different keypoints
        colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
            (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
            (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
            (255, 0, 255), (255, 0, 170)
        ]
        
        # Improved drawing thresholds
        drawing_threshold = 0.3  # Higher threshold for drawing
        edge_margin = 10  # Minimum distance from image edges
        height, width = image.shape[:2]
        
        for pose in poses:
            # Draw bounding box
            bbox = pose['bbox']
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            score = bbox[4] if len(bbox) > 4 else pose['score']
            cv2.putText(image, f'{score:.4f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            keypoints = pose['keypoints']
            
            # Function to validate keypoint
            def is_valid_keypoint(x, y, conf):
                """Check if keypoint is valid for drawing."""
                return (conf > drawing_threshold and 
                        x > edge_margin and y > edge_margin and 
                        x < width - edge_margin and y < height - edge_margin)
            
            # Draw keypoints (only valid ones)
            drawn_keypoints = 0
            valid_keypoints = []
            
            for i, (x, y, conf) in enumerate(keypoints):
                if is_valid_keypoint(x, y, conf):
                    x, y = int(x), int(y)
                    color = colors[i % len(colors)]
                    cv2.circle(image, (x, y), 4, color, -1)
                    
                    # Optional: Draw keypoint index for debugging
                    # cv2.putText(image, f'{i}', (x+5, y-5), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    
                    valid_keypoints.append(i)
                    drawn_keypoints += 1
            
            # Debug: Add drawn keypoint count
            cv2.putText(image, f'KP:{drawn_keypoints}', (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw skeleton (only between valid keypoints)
            for connection in skeleton:
                pt1_idx, pt2_idx = connection
                
                # Check if both keypoints exist and are valid
                if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                    pt1_idx in valid_keypoints and pt2_idx in valid_keypoints):
                    
                    x1_kp, y1_kp, conf1 = keypoints[pt1_idx]
                    x2_kp, y2_kp, conf2 = keypoints[pt2_idx]
                    
                    # Double-check both keypoints are valid
                    if (is_valid_keypoint(x1_kp, y1_kp, conf1) and 
                        is_valid_keypoint(x2_kp, y2_kp, conf2)):
                        
                        x1_kp, y1_kp, x2_kp, y2_kp = int(x1_kp), int(y1_kp), int(x2_kp), int(y2_kp)
                        color = colors[pt1_idx % len(colors)]
                        cv2.line(image, (x1_kp, y1_kp), (x2_kp, y2_kp), color, 2)
        
        return image


class AlphaPoseDetector:
    """AlphaPose-based pose detector using official DetectionLoader pipeline."""
    
    def __init__(self, config: AmbientPoseConfig):
        """Initialize AlphaPose detector."""
        if not ALPHAPOSE_AVAILABLE:
            raise ImportError("AlphaPose is not available")
        
        self.config = config
        
        # Get backend-specific configuration
        self.backend_config = config.get_backend_config('alphapose')
        
        # Parse network resolution (AlphaPose uses this for pose model resolution)
        self.net_resolution = "256x192"  # Default for AlphaPose
        if config.net_resolution:
            try:
                width, height = map(int, config.net_resolution.split('x'))
                # AlphaPose typically uses specific resolutions like 256x192, 384x288
                if f"{width}x{height}" in ["256x192", "384x288", "320x256"]:
                    self.net_resolution = f"{width}x{height}"
                    if config.verbose:
                        logger.info(f"AlphaPose network resolution: {self.net_resolution}")
                else:
                    if config.verbose:
                        logger.warning(f"AlphaPose resolution {width}x{height} not standard, using 256x192")
            except (ValueError, AttributeError):
                logger.warning(f"Invalid net-resolution for AlphaPose: {config.net_resolution}, using default")
        
        # Parse pose model
        self.pose_model = config.model_pose if config.model_pose else 'COCO'
        valid_models = ['COCO', 'HALPE_26', 'HALPE_136', 'MPII']
        if self.pose_model not in valid_models:
            if config.verbose:
                logger.warning(f"Model '{self.pose_model}' not in standard AlphaPose models: {valid_models}")
                logger.info(f"Using '{self.pose_model}' as custom model")
        
        # Set up device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load AlphaPose configuration
        alphapose_config_path = alphapose_path / "configs" / "coco" / "resnet" / "256x192_res50_lr1e-3_1x.yaml"
        if not alphapose_config_path.exists():
            # Fallback to any available config
            config_files = list((alphapose_path / "configs").rglob("*.yaml"))
            if not config_files:
                raise FileNotFoundError("No AlphaPose config files found")
            alphapose_config_path = config_files[0]
            logger.warning(f"Using fallback config: {alphapose_config_path}")
        
        self.cfg = update_config(str(alphapose_config_path))
        logger.info(f"Loaded AlphaPose config: {alphapose_config_path}")
        
        # Create AlphaPose options object for DetectionLoader
        from easydict import EasyDict as edict
        
        # Set up paths for YOLOX
        model_weights_path = alphapose_path / "detector" / "yolox" / "data" / "yolox_x.pth"
        
        self.opt = edict({
            'device': self.device,
            'sp': True,  # Single process mode
            'tracking': False,  # No tracking for now
            'detector': 'yolox-x',
            'inputpath': None,  # Will be set per video
            'outputpath': None,
            'gpus': [0] if 'cuda' in str(self.device) else [-1],  # Required by get_detector
            'ckpt': str(model_weights_path.resolve()),  # YOLOX checkpoint path
        })
        
        # Initialize YOLOX detector (using the working approach from before)
        try:
            # YOLOX expects to be initialized from the AlphaPose directory
            original_cwd = os.getcwd()
            try:
                os.chdir(str(alphapose_path))
                logger.debug(f"Changed working directory to: {alphapose_path}")
                
                from detector.apis import get_detector
                self.detector = get_detector(self.opt)
                logger.info("Initialized YOLOX detector using official API")
            finally:
                os.chdir(original_cwd)
                logger.debug(f"Restored working directory to: {original_cwd}")
                
        except Exception as e:
            logger.warning(f"Failed to initialize via get_detector: {e}")
            # Fallback to the working YOLOX approach
            try:
                original_cwd = os.getcwd()
                try:
                    os.chdir(str(alphapose_path))
                    
                    from detector.yolox_cfg import cfg as yolox_cfg
                    from detector.yolox_api import YOLOXDetector
                    
                    # Set up YOLOX configuration (the working approach)
                    yolox_cfg.MODEL_NAME = 'yolox-x'
                    model_weights_path = alphapose_path / "detector" / "yolox" / "data" / "yolox_x.pth"
                    yolox_cfg.MODEL_WEIGHTS = str(model_weights_path.resolve())  # Use absolute path
                    
                    # Also try setting the ckpt attribute that some YOLOX versions use
                    if hasattr(yolox_cfg, 'ckpt'):
                        yolox_cfg.ckpt = str(model_weights_path.resolve())
                    
                    if not model_weights_path.exists():
                        logger.warning(f"YOLOX model weights not found: {model_weights_path}")
                        logger.warning("YOLOX detector will not work properly without model weights")
                        logger.info("Detection will fall back to Ultralytics YOLO")
                        # Create a dummy detector that will fail gracefully
                        self.detector = None
                    else:
                        self.detector = YOLOXDetector(yolox_cfg, self.opt)
                        logger.info("Initialized YOLOX detector using direct approach")
                finally:
                    os.chdir(original_cwd)
                    
            except Exception as e2:
                logger.warning(f"YOLOX initialization failed: {e2}")
                logger.info("Will use Ultralytics YOLO fallback for detection")
                self.detector = None
        
        # Build pose estimation model
        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        
        # Load pretrained weights
        preferred_models = [
            "fast_res50_256x192.pth",
            "fast_dcn_res50_256x192.pth", 
            "fast_421_res50-shuffle_256x192.pth",
            "simple_res50_256x192.pth"
        ]
        
        checkpoint_path = None
        for model_name in preferred_models:
            candidate_path = alphapose_path / "pretrained_models" / model_name
            if candidate_path.exists():
                checkpoint_path = candidate_path
                logger.info(f"Using pose estimation model: {model_name}")
                break
        
        if checkpoint_path and checkpoint_path.exists():
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        self.pose_model.load_state_dict(checkpoint['state_dict'])
                    elif 'model' in checkpoint:
                        self.pose_model.load_state_dict(checkpoint['model'])
                    else:
                        self.pose_model.load_state_dict(checkpoint)
                else:
                    self.pose_model.load_state_dict(checkpoint)
                    
                logger.info(f"Successfully loaded pose model: {checkpoint_path.name}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
                logger.warning("Checkpoint file may be corrupted. Using random weights instead.")
                # Delete corrupted file
                try:
                    checkpoint_path.unlink()
                    logger.info(f"Deleted corrupted checkpoint: {checkpoint_path}")
                except:
                    pass
        else:
            logger.warning("No checkpoint loaded - using random weights!")
        
        self.pose_model.to(self.device)
        self.pose_model.eval()
        
        logger.info("AlphaPose detector initialized with official pipeline")
    
    def detect_poses(self, image: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """Detect poses using hybrid approach: working YOLOX detection + improved pose processing."""
        height, width = image.shape[:2]
        
        try:
            # Step 1: Human detection using the working YOLOX approach
            human_boxes = []
            
            # Skip YOLOX if detector is None (failed to initialize)
            if self.detector is None:
                logger.debug("YOLOX detector not available, skipping to Ultralytics fallback")
            else:
                # Try the newer approach first, fall back to working file-based approach
                try:
                    # Method 1: In-memory detection (faster)
                    logger.debug("Attempting in-memory YOLOX detection...")
                    
                    img_tensor = self.detector.image_preprocess(image)
                    logger.debug(f"Image preprocessed, tensor type: {type(img_tensor)}")
                    
                    if isinstance(img_tensor, np.ndarray):
                        img_tensor = torch.from_numpy(img_tensor)
                    if img_tensor.dim() == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                    
                    logger.debug(f"Image tensor shape: {img_tensor.shape}")
                    
                    with torch.no_grad():
                        im_dim_list = torch.FloatTensor([[width, height]]).repeat(1, 2)
                        logger.debug(f"Running images_detection with dimensions: {im_dim_list}")
                        
                        dets = self.detector.images_detection(img_tensor, im_dim_list)
                        logger.debug(f"Detection result type: {type(dets)}, shape: {getattr(dets, 'shape', 'no shape')}")
                        
                        if not isinstance(dets, int) and hasattr(dets, 'shape') and dets.shape[0] > 0:
                            if isinstance(dets, np.ndarray):
                                dets = torch.from_numpy(dets)
                            dets = dets.cpu()
                            
                            # Extract human boxes from detections
                            for det in dets:
                                x1, y1, x2, y2 = det[1:5].numpy()
                                det_conf = det[5].item()
                                if det_conf >= self.config.min_confidence:
                                    human_boxes.append([x1, y1, x2, y2, det_conf])
                            
                            logger.debug(f"In-memory YOLOX detected {len(human_boxes)} human boxes")
                        else:
                            logger.debug(f"No valid detections from in-memory method")
                    
                except Exception as e:
                    import traceback
                    logger.warning(f"In-memory detection failed: {str(e)}")
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    logger.debug("Trying file-based approach...")
                
                # Method 2: File-based detection (the working approach from before)
                if len(human_boxes) == 0 and self.detector is not None:
                    try:
                        import tempfile
                        import time
                        
                        logger.debug("Attempting file-based YOLOX detection...")
                        
                        # Create temp file with numeric name (this was working before)
                        temp_dir = tempfile.gettempdir()
                        numeric_name = f"{int(time.time() * 1000000) % 999999999:09d}"
                        temp_path = os.path.join(temp_dir, f"{numeric_name}.jpg")
                        
                        logger.debug(f"Writing temp image to: {temp_path}")
                        success = cv2.imwrite(temp_path, image)
                        if not success:
                            raise Exception(f"Failed to write image to {temp_path}")
                        
                        logger.debug(f"Temp file created, size: {os.path.getsize(temp_path)} bytes")
                        
                        # Check if detector has detect_one_img method
                        if not hasattr(self.detector, 'detect_one_img'):
                            raise AttributeError(f"Detector {type(self.detector)} does not have detect_one_img method")
                        
                        # Use the working detect_one_img approach
                        logger.debug("Calling detect_one_img...")
                        detection_results = self.detector.detect_one_img(temp_path)
                        logger.debug(f"Detection results type: {type(detection_results)}, length: {len(detection_results) if detection_results else 0}")
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            logger.debug("Temp file cleaned up")
                        
                        # Convert to our format (this was working before)
                        if detection_results:
                            logger.debug(f"Processing {len(detection_results)} detection results")
                            for i, det in enumerate(detection_results):
                                logger.debug(f"Detection {i}: {det}")
                                if det.get('category_id') == 1:  # person category
                                    x, y, w, h = det['bbox']
                                    x1, y1, x2, y2 = x, y, x + w, y + h
                                    score = det['score']
                                    if score >= self.config.min_confidence:
                                        human_boxes.append([x1, y1, x2, y2, score])
                                        logger.debug(f"Added human box: [{x1}, {y1}, {x2}, {y2}] score={score}")
                        else:
                            logger.debug("No detection results returned")
                        
                        logger.debug(f"File-based YOLOX detected {len(human_boxes)} human boxes")
                        
                    except Exception as e:
                        import traceback
                        logger.warning(f"File-based detection also failed: {str(e)}")
                        logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            # Method 3: Fallback to Ultralytics YOLO if YOLOX completely fails
            if len(human_boxes) == 0:
                logger.warning("YOLOX detection completely failed, trying Ultralytics YOLO fallback...")
                try:
                    if ULTRALYTICS_AVAILABLE:
                        from ultralytics import YOLO
                        if not hasattr(self, '_fallback_detector'):
                            self._fallback_detector = YOLO('yolov8n.pt')
                            logger.debug("Initialized fallback Ultralytics detector")
                        
                        results = self._fallback_detector(image, verbose=False)
                        logger.debug(f"Ultralytics YOLO returned {len(results)} results")
                        
                        for result in results:
                            if result.boxes is not None:
                                boxes = result.boxes
                                logger.debug(f"Found {len(boxes)} boxes")
                                for i in range(len(boxes)):
                                    box = boxes[i]
                                    cls = int(box.cls[0])
                                    # Only keep person class (class 0 in COCO)
                                    if cls == 0:
                                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                        conf = box.conf[0].cpu().numpy()
                                        if conf >= self.config.min_confidence:
                                            human_boxes.append([x1, y1, x2, y2, conf])
                                            logger.debug(f"Added Ultralytics detection: [{x1}, {y1}, {x2}, {y2}] score={conf}")
                        
                        logger.info(f"Ultralytics fallback detected {len(human_boxes)} human boxes")
                    else:
                        logger.warning("Ultralytics not available for fallback")
                        
                except Exception as e:
                    import traceback
                    logger.error(f"Ultralytics fallback also failed: {str(e)}")
                    logger.debug(f"Ultralytics traceback: {traceback.format_exc()}")
            
            if len(human_boxes) == 0:
                logger.warning("All detection methods failed - no human detections found")
                return []
            
            # Step 2: Pose estimation using official AlphaPose coordinate processing
            from alphapose.utils.presets import SimpleTransform
            pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
            transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self.cfg.DATA_PRESET.IMAGE_SIZE,
                output_size=self.cfg.DATA_PRESET.HEATMAP_SIZE,
                rot=0, sigma=self.cfg.DATA_PRESET.SIGMA,
                train=False, add_dpg=False, gpu_device=self.device)
            
            poses = []
            
            for box in human_boxes:
                x1, y1, x2, y2, det_conf = box
                
                try:
                    # Use official AlphaPose transformation
                    pose_input, cropped_box = transformation.test_transform(image, [x1, y1, x2, y2])
                    
                    # Run pose estimation
                    with torch.no_grad():
                        pose_input = pose_input.unsqueeze(0).to(self.device)
                        heatmaps = self.pose_model(pose_input)
                        if isinstance(heatmaps, list):
                            heatmaps = heatmaps[-1]
                        
                        # Use official coordinate conversion
                        from alphapose.utils.transforms import heatmap_to_coord_simple
                        heatmaps_np = heatmaps[0].cpu().numpy()
                        coords, maxvals = heatmap_to_coord_simple(heatmaps_np, cropped_box)
                        
                        keypoints = []
                        for j in range(len(coords)):
                            if j < len(maxvals):
                                x_coord = float(coords[j, 0])
                                y_coord = float(coords[j, 1])
                                confidence = float(maxvals[j, 0])
                                
                                # Validate coordinates
                                if 0 <= x_coord <= width and 0 <= y_coord <= height and confidence > 0:
                                    keypoints.append([x_coord, y_coord, confidence])
                        
                        if len(keypoints) > 0:
                            pose = {
                                'frame_idx': frame_idx,
                                'bbox': [float(x1), float(y1), float(x2), float(y2), float(det_conf)],
                                'score': float(det_conf),
                                'keypoints': keypoints,
                                'backend': 'alphapose'
                            }
                            poses.append(pose)
                            
                            if self.config.debug:
                                logger.debug(f"Pose extracted: {len(keypoints)} keypoints, score={det_conf:.3f}")
                
                except Exception as e:
                    logger.debug(f"Error processing detection: {e}")
                    continue
            
            # Apply pose NMS if needed
            if len(poses) > 1:
                try:
                    pose_coords = []
                    pose_scores = []
                    for pose in poses:
                        coords = np.array([[kp[0], kp[1], kp[2]] for kp in pose['keypoints']])
                        pose_coords.append(coords)
                        pose_scores.append(pose['score'])
                    
                    keep_indices = pose_nms(np.array(pose_coords), np.array(pose_scores), 0.5)
                    if keep_indices is not None and len(keep_indices) > 0:
                        poses = [poses[i] for i in keep_indices if i < len(poses)]
                except Exception as e:
                    logger.debug(f"Pose NMS failed: {e}, keeping all poses")
            
            return poses
            
        except Exception as e:
            logger.error(f"AlphaPose detection failed: {e}")
            return []
    

    
    def _simple_pose_filter(self, poses: List[Dict[str, Any]]) -> List[int]:
        """Simple pose filtering based on overlap and confidence when NMS is not available."""
        if len(poses) <= 1:
            return list(range(len(poses)))
        
        # Sort poses by confidence (highest first)
        sorted_indices = sorted(range(len(poses)), key=lambda i: poses[i]['score'], reverse=True)
        
        keep_indices = []
        for i in sorted_indices:
            keep = True
            pose_i = poses[i]
            
            # Check overlap with already selected poses
            for j in keep_indices:
                pose_j = poses[j]
                
                # Calculate bbox overlap
                bbox_i = pose_i['bbox'][:4]
                bbox_j = pose_j['bbox'][:4]
                
                # Simple overlap check - if bboxes overlap significantly, skip lower confidence pose
                overlap = self._calculate_bbox_overlap(bbox_i, bbox_j)
                if overlap > 0.5:  # 50% overlap threshold
                    keep = False
                    break
            
            if keep:
                keep_indices.append(i)
        
        return keep_indices
    
    def _calculate_bbox_overlap(self, box1: List[float], box2: List[float]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Return intersection over smaller area (more conservative)
        return intersection / min(area1, area2) if min(area1, area2) > 0 else 0.0
    
    def draw_poses(self, image: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
        """Draw poses on the image."""
        if not poses:
            return image
        
        # COCO skeleton connections (17 keypoints)
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms  
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Colors for different keypoints
        colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
            (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
            (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
            (255, 0, 255), (255, 0, 170)
        ]
        
        # Appropriate drawing thresholds for AlphaPose using official coordinate processing
        drawing_threshold = 0.1  # Lower threshold for AlphaPose as official processing gives better confidence scores
        edge_margin = 10  # Minimum distance from image edges
        height, width = image.shape[:2]
        
        for pose in poses:
            # Draw bounding box
            bbox = pose['bbox']
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            score = bbox[4] if len(bbox) > 4 else pose['score']
            cv2.putText(image, f'{score:.4f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            keypoints = pose['keypoints']
            
            # Function to validate keypoint
            def is_valid_keypoint(x, y, conf):
                """Check if keypoint is valid for drawing."""
                return (conf > drawing_threshold and 
                        x > edge_margin and y > edge_margin and 
                        x < width - edge_margin and y < height - edge_margin)
            
            # Draw keypoints (only valid ones)
            drawn_keypoints = 0
            valid_keypoints = []
            
            for i, (x, y, conf) in enumerate(keypoints):
                if is_valid_keypoint(x, y, conf):
                    x, y = int(x), int(y)
                    color = colors[i % len(colors)]
                    cv2.circle(image, (x, y), 4, color, -1)
                    
                    # Optional: Draw keypoint index for debugging
                    # cv2.putText(image, f'{i}', (x+5, y-5), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    
                    valid_keypoints.append(i)
                    drawn_keypoints += 1
            
            # Debug: Add drawn keypoint count
            cv2.putText(image, f'KP:{drawn_keypoints}', (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw skeleton (only between valid keypoints)
            for connection in skeleton:
                pt1_idx, pt2_idx = connection
                
                # Check if both keypoints exist and are valid
                if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                    pt1_idx in valid_keypoints and pt2_idx in valid_keypoints):
                    
                    x1_kp, y1_kp, conf1 = keypoints[pt1_idx]
                    x2_kp, y2_kp, conf2 = keypoints[pt2_idx]
                    
                    # Double-check both keypoints are valid
                    if (is_valid_keypoint(x1_kp, y1_kp, conf1) and 
                        is_valid_keypoint(x2_kp, y2_kp, conf2)):
                        
                        x1_kp, y1_kp, x2_kp, y2_kp = int(x1_kp), int(y1_kp), int(x2_kp), int(y2_kp)
                        color = colors[pt1_idx % len(colors)]
                        cv2.line(image, (x1_kp, y1_kp), (x2_kp, y2_kp), color, 2)
        
        return image


class OpenPoseDetector:
    """OpenPose-based pose detector using the official OpenPose installation."""
    
    def __init__(self, config: AmbientPoseConfig):
        """Initialize OpenPose detector."""
        if not OPENPOSE_AVAILABLE:
            raise ImportError("OpenPose is not available")
        
        self.config = config
        
        # Get backend-specific configuration
        self.backend_config = config.get_backend_config('openpose')
        
        # Get OpenPose installation path
        self.openpose_home = Path(os.environ.get('OPENPOSE_HOME', ''))
        self.openpose_bin_path = self.openpose_home / "bin"
        
        # Parse network resolution
        self.net_resolution = "656x368"  # Default
        if config.net_resolution:
            try:
                width, height = map(int, config.net_resolution.split('x'))
                self.net_resolution = f"{width}x{height}"
                if config.verbose:
                    logger.info(f"OpenPose network resolution: {self.net_resolution}")
            except (ValueError, AttributeError):
                logger.warning(f"Invalid net-resolution for OpenPose: {config.net_resolution}, using default")
        
        # Parse pose model
        self.pose_model = config.model_pose if config.model_pose else 'BODY_25'
        valid_models = ['COCO', 'BODY_25', 'MPI', 'MPI_4_layers']
        if self.pose_model not in valid_models:
            if config.verbose:
                logger.warning(f"Model '{self.pose_model}' not in valid OpenPose models: {valid_models}")
                logger.info(f"Falling back to BODY_25")
            self.pose_model = 'BODY_25'
        
        # Initialize OpenPose
        self._init_openpose()
        
        if config.verbose:
            logger.info(f"OpenPose detector initialized")
            logger.info(f"Pose model: {self.pose_model}")
            logger.info(f"Network resolution: {self.net_resolution}")
        else:
            logger.info("OpenPose detector initialized")
    
    def _init_openpose(self):
        """Initialize OpenPose with appropriate method (Python bindings or subprocess)."""
        self.use_python_bindings = False
        self.openpose_binary_path = None
        
        # First try Python bindings
        python_binding_paths = [
            self.openpose_home / "build" / "python" / "openpose",
            self.openpose_home / "python" / "openpose",
            self.openpose_bin_path / "python",
        ]
        
        for python_path in python_binding_paths:
            if python_path.exists():
                try:
                    # Add to Python path temporarily
                    temp_path = str(python_path.parent)
                    if temp_path not in sys.path:
                        sys.path.insert(0, temp_path)
                    
                    import openpose as op
                    self.op = op
                    self.use_python_bindings = True
                    
                    # Set up OpenPose parameters
                    self.params = dict()
                    self.params["model_folder"] = str(self.openpose_home / "models")
                    self.params["face"] = False
                    self.params["hand"] = False
                    self.params["net_resolution"] = self.net_resolution
                    self.params["model_pose"] = self.pose_model
                    
                    # Create OpenPose wrapper
                    self.opWrapper = op.WrapperPython()
                    self.opWrapper.configure(self.params)
                    self.opWrapper.start()
                    
                    logger.info(f"OpenPose initialized with Python bindings from: {python_path}")
                    return
                    
                except ImportError as e:
                    logger.debug(f"Failed to import OpenPose Python bindings from {python_path}: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Failed to initialize OpenPose Python bindings from {python_path}: {e}")
                    continue
        
        # Fall back to binary execution if Python bindings not available
        possible_binaries = [
            self.openpose_bin_path / "OpenPoseDemo.exe",  # Windows
            self.openpose_bin_path / "examples" / "openpose" / "openpose.bin",  # Linux
            self.openpose_bin_path / "openpose",  # Alternative Linux
        ]
        
        for binary_path in possible_binaries:
            if binary_path.exists():
                self.openpose_binary_path = binary_path
                self.models_path = self.openpose_home / "models"
                logger.info(f"OpenPose initialized with binary execution: {binary_path}")
                return
        
        raise RuntimeError("No valid OpenPose Python bindings or binary found")
    
    def detect_poses(self, image: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """Detect poses in a single image."""
        height, width = image.shape[:2]
        poses = []
        
        try:
            if self.use_python_bindings:
                # Use Python bindings
                datum = self.op.Datum()
                datum.cvInputData = image
                self.opWrapper.emplaceAndPop([datum])
                
                if datum.poseKeypoints is not None and datum.poseKeypoints.size > 0:
                    # datum.poseKeypoints shape: (num_people, num_keypoints, 3)
                    keypoints_array = datum.poseKeypoints
                    
                    for person_idx in range(keypoints_array.shape[0]):
                        person_keypoints = keypoints_array[person_idx]
                        
                        # Filter out keypoints with zero confidence
                        valid_keypoints = []
                        confidences = []
                        x_coords = []
                        y_coords = []
                        
                        for kp_idx in range(person_keypoints.shape[0]):
                            x, y, conf = person_keypoints[kp_idx]
                            if conf > 0:  # Only include detected keypoints
                                valid_keypoints.append([float(x), float(y), float(conf)])
                                confidences.append(conf)
                                x_coords.append(x)
                                y_coords.append(y)
                            else:
                                valid_keypoints.append([0.0, 0.0, 0.0])
                        
                        # Calculate bounding box from valid keypoints
                        if x_coords and y_coords:
                            x1, x2 = min(x_coords), max(x_coords)
                            y1, y2 = min(y_coords), max(y_coords)
                            
                            # Add padding
                            padding = 20
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding)
                            x2 = min(width, x2 + padding)
                            y2 = min(height, y2 + padding)
                            
                            # Calculate overall confidence
                            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                            
                            # Skip low-confidence detections
                            if avg_confidence < self.config.min_confidence:
                                continue
                            
                            pose = {
                                'frame_idx': frame_idx,
                                'bbox': [float(x1), float(y1), float(x2), float(y2), float(avg_confidence)],
                                'score': float(avg_confidence),
                                'keypoints': valid_keypoints,
                                'backend': 'openpose'
                            }
                            poses.append(pose)
            
            else:
                # Use binary execution (less efficient but more compatible)
                poses = self._detect_with_binary(image, frame_idx)
                
        except Exception as e:
            logger.error(f"OpenPose detection failed: {e}")
            return []
        
        return poses
    
    def _detect_with_binary(self, image: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """Detect poses using OpenPose binary execution."""
        import tempfile
        import subprocess
        import json as json_module
        
        poses = []
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # Save input image
                input_image_path = temp_dir_path / f"input_{frame_idx}.jpg"
                cv2.imwrite(str(input_image_path), image)
                
                # Set up output paths
                output_dir = temp_dir_path / "output"
                output_dir.mkdir()
                
                # Construct OpenPose command
                cmd = [
                    str(self.openpose_binary_path),
                    "--image_dir", str(temp_dir_path),
                    "--write_json", str(output_dir),
                    "--display", "0",
                    "--render_pose", "0",
                    "--model_folder", str(self.models_path),
                    "--net_resolution", self.net_resolution,
                    "--model_pose", self.pose_model
                ]
                
                # Run OpenPose
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Read JSON output
                    json_files = list(output_dir.glob("*.json"))
                    if json_files:
                        json_file = json_files[0]
                        with open(json_file, 'r') as f:
                            openpose_data = json_module.load(f)
                        
                        # Process OpenPose results
                        if 'people' in openpose_data:
                            for person_data in openpose_data['people']:
                                if 'pose_keypoints_2d' in person_data:
                                    keypoints_flat = person_data['pose_keypoints_2d']
                                    
                                    # Convert flat list to (x, y, confidence) triplets
                                    keypoints = []
                                    confidences = []
                                    x_coords = []
                                    y_coords = []
                                    
                                    for i in range(0, len(keypoints_flat), 3):
                                        x = keypoints_flat[i]
                                        y = keypoints_flat[i + 1]
                                        conf = keypoints_flat[i + 2]
                                        
                                        keypoints.append([float(x), float(y), float(conf)])
                                        
                                        if conf > 0:
                                            confidences.append(conf)
                                            x_coords.append(x)
                                            y_coords.append(y)
                                    
                                    # Calculate bounding box and overall confidence
                                    if x_coords and y_coords:
                                        height, width = image.shape[:2]
                                        x1, x2 = min(x_coords), max(x_coords)
                                        y1, y2 = min(y_coords), max(y_coords)
                                        
                                        # Add padding
                                        padding = 20
                                        x1 = max(0, x1 - padding)
                                        y1 = max(0, y1 - padding)
                                        x2 = min(width, x2 + padding)
                                        y2 = min(height, y2 + padding)
                                        
                                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                                        
                                        # Skip low-confidence detections
                                        if avg_confidence < self.config.min_confidence:
                                            continue
                                        
                                        pose = {
                                            'frame_idx': frame_idx,
                                            'bbox': [float(x1), float(y1), float(x2), float(y2), float(avg_confidence)],
                                            'score': float(avg_confidence),
                                            'keypoints': keypoints,
                                            'backend': 'openpose'
                                        }
                                        poses.append(pose)
                else:
                    logger.warning(f"OpenPose binary execution failed: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            logger.warning("OpenPose binary execution timed out")
        except Exception as e:
            logger.error(f"OpenPose binary execution error: {e}")
        
        return poses
    
    def draw_poses(self, image: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
        """Draw poses on the image."""
        if not poses:
            return image
        
        # OpenPose BODY_25 skeleton connections
        skeleton = [
            (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),  # Head and arms
            (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),  # Torso and legs
            (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),  # Neck to face
            (2, 16), (5, 17)  # Arms to ears
        ]
        
        # Colors for different keypoints
        colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
            (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
            (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
            (255, 0, 255), (255, 0, 170), (255, 85, 85), (255, 170, 170), (255, 255, 85),
            (170, 255, 85), (85, 255, 170), (85, 170, 255), (170, 85, 255), (255, 85, 170)
        ]
        
        # Drawing thresholds
        drawing_threshold = 0.1
        edge_margin = 10
        height, width = image.shape[:2]
        
        for pose in poses:
            # Draw bounding box
            bbox = pose['bbox']
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            score = bbox[4] if len(bbox) > 4 else pose['score']
            cv2.putText(image, f'{score:.3f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            keypoints = pose['keypoints']
            
            # Function to validate keypoint
            def is_valid_keypoint(x, y, conf):
                """Check if keypoint is valid for drawing."""
                return (conf > drawing_threshold and 
                        x > edge_margin and y > edge_margin and 
                        x < width - edge_margin and y < height - edge_margin)
            
            # Draw keypoints (only valid ones)
            drawn_keypoints = 0
            valid_keypoints = []
            
            for i, (x, y, conf) in enumerate(keypoints):
                if is_valid_keypoint(x, y, conf):
                    x, y = int(x), int(y)
                    color = colors[i % len(colors)]
                    cv2.circle(image, (x, y), 4, color, -1)
                    
                    valid_keypoints.append(i)
                    drawn_keypoints += 1
            
            # Debug: Add drawn keypoint count
            cv2.putText(image, f'KP:{drawn_keypoints}', (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw skeleton (only between valid keypoints)
            for connection in skeleton:
                pt1_idx, pt2_idx = connection
                
                # Check if both keypoints exist and are valid
                if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                    pt1_idx in valid_keypoints and pt2_idx in valid_keypoints):
                    
                    x1_kp, y1_kp, conf1 = keypoints[pt1_idx]
                    x2_kp, y2_kp, conf2 = keypoints[pt2_idx]
                    
                    # Double-check both keypoints are valid
                    if (is_valid_keypoint(x1_kp, y1_kp, conf1) and 
                        is_valid_keypoint(x2_kp, y2_kp, conf2)):
                        
                        x1_kp, y1_kp, x2_kp, y2_kp = int(x1_kp), int(y1_kp), int(x2_kp), int(y2_kp)
                        color = colors[pt1_idx % len(colors)]
                        cv2.line(image, (x1_kp, y1_kp), (x2_kp, y2_kp), color, 2)
        
        return image


class PersonTracker:
    """Simple person tracker to maintain consistent person_id across frames."""
    
    def __init__(self, iou_threshold: float = 0.3):
        """Initialize person tracker."""
        self.iou_threshold = iou_threshold
        self.tracks = []  # List of tracked persons
        self.next_person_id = 0
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, current_poses: List[Dict[str, Any]], frame_number: int) -> List[Dict[str, Any]]:
        """Update tracks with current frame poses and assign person_ids."""
        if not current_poses:
            return []
        
        # Extract bounding boxes from current poses
        current_boxes = [pose['bbox'] for pose in current_poses]
        
        # If this is the first frame, initialize all tracks
        if not self.tracks:
            for i, pose in enumerate(current_poses):
                track = {
                    'person_id': self.next_person_id,
                    'bbox': pose['bbox'],
                    'last_seen_frame': frame_number,
                    'pose': pose
                }
                self.tracks.append(track)
                pose['person_id'] = self.next_person_id
                self.next_person_id += 1
            return current_poses
        
        # Match current detections with existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for i, current_box in enumerate(current_boxes):
            best_iou = 0
            best_track_idx = -1
            
            for j, track in enumerate(self.tracks):
                if j in matched_tracks:
                    continue
                    
                iou = self.calculate_iou(current_box, track['bbox'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track_idx = j
            
            if best_track_idx >= 0:
                # Match found
                track = self.tracks[best_track_idx]
                current_poses[i]['person_id'] = track['person_id']
                track['bbox'] = current_box
                track['last_seen_frame'] = frame_number
                track['pose'] = current_poses[i]
                matched_tracks.add(best_track_idx)
                matched_detections.add(i)
        
        # Create new tracks for unmatched detections
        for i, pose in enumerate(current_poses):
            if i not in matched_detections:
                track = {
                    'person_id': self.next_person_id,
                    'bbox': pose['bbox'],
                    'last_seen_frame': frame_number,
                    'pose': pose
                }
                self.tracks.append(track)
                pose['person_id'] = self.next_person_id
                self.next_person_id += 1
        
        # Remove old tracks (not seen for 30 frames)
        self.tracks = [track for track in self.tracks if frame_number - track['last_seen_frame'] <= 30]
        
        return current_poses


def get_coco_joint_names() -> List[str]:
    """Get COCO joint names in order."""
    return [
        "nose",           # 0
        "left_eye",       # 1  
        "right_eye",      # 2
        "left_ear",       # 3
        "right_ear",      # 4
        "left_shoulder",  # 5
        "right_shoulder", # 6
        "left_elbow",     # 7
        "right_elbow",    # 8
        "left_wrist",     # 9
        "right_wrist",    # 10
        "left_hip",       # 11
        "right_hip",      # 12
        "left_knee",      # 13
        "right_knee",     # 14
        "left_ankle",     # 15
        "right_ankle",    # 16
        "joint_17"        # 17 (additional joint from example)
    ]


def convert_pose_to_joints_format(pose: Dict[str, Any], frame_number: int, timestamp: float) -> Dict[str, Any]:
    """Convert pose data to the target joints format."""
    joint_names = get_coco_joint_names()
    
    joints = []
    keypoints = pose['keypoints']
    
    # Ensure we have enough keypoints, pad with zeros if needed
    while len(keypoints) < len(joint_names):
        keypoints.append([0.0, 0.0, 0.0])
    
    for i, joint_name in enumerate(joint_names):
        if i < len(keypoints):
            x, y, confidence = keypoints[i]
        else:
            x, y, confidence = 0.0, 0.0, 0.0
        
        # Skip keypoints at (0,0) as they are spurious/undetected
        if x == 0.0 and y == 0.0:
            continue
            
        joint = {
            "name": joint_name,
            "joint_id": i,
            "keypoint": {
                "x": round(float(x), 4),
                "y": round(float(y), 4), 
                "confidence": round(float(confidence), 4)
            }
        }
        joints.append(joint)
    
    # Calculate overall pose confidence (average of valid keypoint confidences)
    valid_confidences = [kp[2] for kp in keypoints if kp[2] > 0 and not (kp[0] == 0.0 and kp[1] == 0.0)]
    overall_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
    
    return {
        "person_id": pose.get('person_id', 0),
        "frame_number": frame_number,
        "timestamp": timestamp,
        "confidence": round(float(overall_confidence), 4),
        "joints": joints
    }


class PoseDetector:
    """
    Main pose detector that manages different backends and advanced output options.

    Supports:
      - Video and image directory input
      - Backend selection and fallback
      - Overlay video generation
      - Toronto gait format output
      - Comprehensive frame extraction
      - Verbose and debug logging
    """
    
    def __init__(self, config: AmbientPoseConfig):
        """Initialize the pose detector with the best available backend."""
        self.config = config
        self.detector = None
        self.person_tracker = PersonTracker()
        self.video_metadata = {}
        self.all_converted_poses = []  # Store poses in the new format
        self.comprehensive_frame_data = []  # Store comprehensive frame analysis
        
        # Select backend
        if config.backend == "auto":
            # Auto-select the best available backend
            if MEDIAPIPE_AVAILABLE:
                self.backend_name = "mediapipe"
                self.detector = MediaPipeDetector(config)
            elif ULTRALYTICS_AVAILABLE:
                self.backend_name = "ultralytics"
                self.detector = UltralyticsDetector(config)
            elif OPENPOSE_AVAILABLE:
                self.backend_name = "openpose"
                self.detector = OpenPoseDetector(config)
            elif ALPHAPOSE_AVAILABLE:
                self.backend_name = "alphapose"
                self.detector = AlphaPoseDetector(config)
            else:
                raise RuntimeError("No pose detection backends available. Please install MediaPipe, Ultralytics, OpenPose, or AlphaPose.")
        elif config.backend == "mediapipe":
            if not MEDIAPIPE_AVAILABLE:
                raise RuntimeError("MediaPipe backend not available. Please install MediaPipe.")
            self.backend_name = "mediapipe"
            self.detector = MediaPipeDetector(config)
        elif config.backend == "ultralytics":
            if not ULTRALYTICS_AVAILABLE:
                raise RuntimeError("Ultralytics backend not available. Please install Ultralytics.")
            self.backend_name = "ultralytics"
            self.detector = UltralyticsDetector(config)
        elif config.backend == "openpose":
            if not OPENPOSE_AVAILABLE:
                logger.error("❌ OpenPose backend requested but not available!")
                logger.error("")
                logger.error("🔧 To use OpenPose:")
                logger.error("   1. Set OPENPOSE_HOME environment variable to your OpenPose installation directory")
                logger.error("   2. Ensure OpenPose binaries are in the bin/ subdirectory")
                logger.error("   3. For optimal performance, compile OpenPose with Python bindings")
                logger.error("")
                logger.error("💡 Falling back to best available backend...")
                
                # Auto-select the best available backend as fallback
                if MEDIAPIPE_AVAILABLE:
                    logger.info("🔄 Using MediaPipe backend as fallback")
                    self.backend_name = "mediapipe"
                    self.detector = MediaPipeDetector(config)
                elif ULTRALYTICS_AVAILABLE:
                    logger.info("🔄 Using Ultralytics backend as fallback")
                    self.backend_name = "ultralytics"
                    self.detector = UltralyticsDetector(config)
                elif ALPHAPOSE_AVAILABLE:
                    logger.info("🔄 Using AlphaPose backend as fallback")
                    self.backend_name = "alphapose"
                    self.detector = AlphaPoseDetector(config)
                else:
                    raise RuntimeError("OpenPose backend not available and no fallback backends found. Please install MediaPipe, Ultralytics, or AlphaPose.")
            else:
                self.backend_name = "openpose"
                self.detector = OpenPoseDetector(config)
        elif config.backend == "alphapose":
            if not ALPHAPOSE_AVAILABLE:
                logger.error("❌ AlphaPose backend requested but not available!")
                logger.error("")
                logger.error("🔧 To install AlphaPose on Windows:")
                logger.error("   1. Install Visual Studio Build Tools with C++ workload")
                logger.error("   2. Run: uv run python scripts/install_alphapose.py")
                logger.error("")
                logger.error("💡 Falling back to best available backend...")
                
                # Auto-select the best available backend as fallback
                if MEDIAPIPE_AVAILABLE:
                    logger.info("🔄 Using MediaPipe backend as fallback")
                    self.backend_name = "mediapipe"
                    self.detector = MediaPipeDetector(config)
                elif ULTRALYTICS_AVAILABLE:
                    logger.info("🔄 Using Ultralytics backend as fallback")
                    self.backend_name = "ultralytics"
                    self.detector = UltralyticsDetector(config)
                elif OPENPOSE_AVAILABLE:
                    logger.info("🔄 Using OpenPose backend as fallback")
                    self.backend_name = "openpose"
                    self.detector = OpenPoseDetector(config)
                else:
                    raise RuntimeError("AlphaPose backend not available and no fallback backends found. Please install MediaPipe, Ultralytics, or OpenPose.")
            else:
                self.backend_name = "alphapose"
                self.detector = AlphaPoseDetector(config)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")
        
        logger.info(f"Using backend: {self.backend_name}")
    
    def process_video(self) -> List[Dict[str, Any]]:
        """Process a video file and return pose data."""
        video_path = str(self.config.input_path)
        logger.info(f"Processing video: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Store video metadata
        self.video_metadata = {
            "width": width,
            "height": height,
            "fps": float(fps),
            "frame_count": total_frames,
            "duration": float(duration)
        }
        
        logger.info(f"Video info: {total_frames} frames, {fps:.4f} FPS, {width}x{height}")
        
        # Process frames
        all_poses = []
        frame_idx = 0
        overlay_frames = []  # Store overlay frames for video generation
        
        # Set up overlay video writer if requested
        overlay_video_writer = None
        if self.config.overlay_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            overlay_video_writer = cv2.VideoWriter(
                self.config.overlay_video_path, 
                fourcc, 
                fps, 
                (width, height)
            )
            if self.config.verbose:
                logger.info(f"Overlay video writer initialized: {self.config.overlay_video_path}")
        
        with logger.contextualize(video=video_path):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Log progress
                if frame_idx % 10 == 0 or frame_idx == 0:
                    progress = frame_idx / total_frames * 100 if total_frames > 0 else 0
                    logger.info(f"Processing frame {frame_idx}/{total_frames} ({progress:.1f}%)")
                
                # Save raw frame
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(self.config.frames_dir / frame_filename), frame)
                
                # Detect poses
                poses = self.detector.detect_poses(frame, frame_idx)
                
                # Apply person tracking
                tracked_poses = self.person_tracker.update(poses, frame_idx)
                all_poses.extend(tracked_poses)
                
                # Convert to new format and store
                timestamp = frame_idx / fps if fps > 0 else 0.0
                for pose in tracked_poses:
                    converted_pose = convert_pose_to_joints_format(pose, frame_idx, timestamp)
                    self.all_converted_poses.append(converted_pose)
                
                # Comprehensive frame extraction if requested
                if self.config.extract_comprehensive_frames:
                    self._extract_comprehensive_frame_data(frame, frame_idx, timestamp, tracked_poses)
                
                # Draw and save overlay frame
                overlay_frame = self.detector.draw_poses(frame.copy(), tracked_poses)
                cv2.imwrite(str(self.config.overlay_dir / frame_filename), overlay_frame)
                
                # Add frame to overlay video if requested
                if overlay_video_writer is not None:
                    overlay_video_writer.write(overlay_frame)
                
                frame_idx += 1
        
        cap.release()
        
        # Finalize overlay video
        if overlay_video_writer is not None:
            overlay_video_writer.release()
            if self.config.verbose:
                logger.success(f"Overlay video saved: {self.config.overlay_video_path}")
            else:
                logger.info(f"Overlay video saved: {self.config.overlay_video_path}")
        logger.info(f"Video processing complete: {frame_idx} frames processed, {len(all_poses)} poses detected")
        
        return all_poses
    
    def process_images(self) -> List[Dict[str, Any]]:
        """Process a directory of images and return pose data."""
        image_dir = self.config.input_path
        logger.info(f"Processing images in directory: {image_dir}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = sorted([
            f for f in image_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            raise ValueError(f"No image files found in directory: {image_dir}")
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Set metadata for image processing
        if image_files:
            first_image = cv2.imread(str(image_files[0]))
            if first_image is not None:
                height, width = first_image.shape[:2]
                self.video_metadata = {
                    "width": width,
                    "height": height,
                    "fps": 1.0,  # 1 FPS for images
                    "frame_count": len(image_files),
                    "duration": float(len(image_files))
                }
        
        # Process images
        all_poses = []
        
        for idx, image_path in enumerate(image_files):
            # Log progress
            if idx % 10 == 0 or idx == 0:
                progress = (idx + 1) / len(image_files) * 100
                logger.info(f"Processing image {idx+1}/{len(image_files)} ({progress:.1f}%)")
            
            # Read image
            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.warning(f"Failed to read image: {image_path}")
                continue
            
            # Save raw frame
            frame_filename = f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(self.config.frames_dir / frame_filename), frame)
            
            # Detect poses
            poses = self.detector.detect_poses(frame, idx)
            
            # Apply person tracking
            tracked_poses = self.person_tracker.update(poses, idx)
            all_poses.extend(tracked_poses)
            
            # Convert to new format and store
            timestamp = float(idx)  # Use frame index as timestamp for images
            for pose in tracked_poses:
                converted_pose = convert_pose_to_joints_format(pose, idx, timestamp)
                self.all_converted_poses.append(converted_pose)
            
            # Draw and save overlay frame
            overlay_frame = self.detector.draw_poses(frame.copy(), tracked_poses)
            cv2.imwrite(str(self.config.overlay_dir / frame_filename), overlay_frame)
        
        logger.info(f"Image processing complete: {len(image_files)} images processed, {len(all_poses)} poses detected")
        
        return all_poses
    
    def save_results(self, poses: List[Dict[str, Any]]) -> None:
        """Save pose detection results to JSON file in the new format."""
        logger.info(f"Saving results to {self.config.output_json}")
        
        # Create output directory if it doesn't exist
        self.config.output_json.parent.mkdir(exist_ok=True, parents=True)
        
        # Generate timestamp in the required format
        from datetime import datetime
        output_generated_at = datetime.now().isoformat()
        
        # Calculate summary statistics
        total_poses = len(self.all_converted_poses)
        unique_people = len(set(pose['person_id'] for pose in self.all_converted_poses))
        frames_with_poses = len(set(pose['frame_number'] for pose in self.all_converted_poses))
        total_frames = self.video_metadata.get('frame_count', frames_with_poses)
        avg_poses_per_frame = round(total_poses / total_frames, 4) if total_frames > 0 else 0
        
        # Determine model info based on backend
        if self.backend_name == "alphapose":
            model_pose = "COCO"
            net_resolution = "256x192"  # Based on AlphaPose config
        elif self.backend_name == "ultralytics":
            model_pose = "COCO"
            net_resolution = "640x640"  # YOLOv8 default
        elif self.backend_name == "openpose":
            model_pose = "BODY_25"  # OpenPose uses BODY_25 format
            net_resolution = "656x368"  # OpenPose default resolution
        else:  # mediapipe
            model_pose = "COCO"
            net_resolution = "256x256"  # MediaPipe pose
        
        # Build the complete JSON structure
        result = {
            "metadata": {
                "input_file": str(self.config.input_path).replace('/', '\\'),
                "output_generated_at": output_generated_at,
                "total_poses_detected": total_poses,
                "processing_info": {
                    "input_type": "video" if self.config.is_video else "images",
                    "video_metadata": self.video_metadata,
                    "model_pose": model_pose,
                    "net_resolution": net_resolution,
                    "confidence_threshold": self.config.min_confidence
                }
            },
            "summary": {
                "total_poses": total_poses,
                "total_frames": total_frames,
                "people_detected": unique_people,
                "frames_with_poses": frames_with_poses,
                "average_poses_per_frame": avg_poses_per_frame
            },
            "poses": self.all_converted_poses
        }
        
        # Save JSON
        with open(self.config.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.success(f"Results saved: {total_poses} poses detected using {self.backend_name} backend")
        logger.success(f"Tracked {unique_people} unique people across {total_frames} frames")
    
    def save_csv_results(self, poses: List[Dict[str, Any]]) -> None:
        """Save pose detection results to CSV file with detailed joint information."""
        # Generate CSV filename
        csv_filename = self.config.output_json.stem + ".csv"
        csv_path = self.config.output_json.parent / csv_filename
        
        logger.info(f"Saving CSV results to {csv_path}")
        
        # Helper function to format floating point numbers
        def format_float(value: float) -> str:
            """Format float to max 2 decimal places or 4 significant digits."""
            if abs(value) >= 100:
                # For large numbers, use 2 decimal places
                return f"{value:.4f}"
            elif abs(value) >= 1:
                # For numbers >= 1, use up to 2 decimal places
                formatted = f"{value:.4f}"
                # Remove trailing zeros
                return formatted.rstrip('0').rstrip('.')
            else:
                # For small numbers, use up to 4 significant digits
                if value == 0:
                    return "0"
                # Format with 4 significant digits, then limit decimal places
                formatted = f"{value:.4g}"
                # If it has more than 2 decimal places, limit to 2
                if '.' in formatted and len(formatted.split('.')[1]) > 2:
                    formatted = f"{value:.4f}"
                return formatted
        
        # CSV headers
        headers = [
            'frame_number', 'timestamp', 'person_id', 'joint_name', 'joint_id',
            'x', 'y', 'confidence', 'pose_confidence', 'total_joints'
        ]
        
        # Write CSV file
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(headers)
            
            # Write data rows
            for pose in self.all_converted_poses:
                frame_number = pose['frame_number']
                timestamp = pose['timestamp']
                person_id = pose['person_id']
                pose_confidence = pose['confidence']
                total_joints = len(pose['joints'])
                
                # Write one row per joint
                for joint in pose['joints']:
                    joint_name = joint['name']
                    joint_id = joint['joint_id']
                    x = joint['keypoint']['x']
                    y = joint['keypoint']['y']
                    confidence = joint['keypoint']['confidence']
                    
                    # Format the row with proper number formatting
                    row = [
                        frame_number,                    # int
                        format_float(timestamp),         # float
                        person_id,                       # int
                        joint_name,                      # string
                        joint_id,                        # int
                        format_float(x),                 # float
                        format_float(y),                 # float
                        format_float(confidence),        # float
                        format_float(pose_confidence),   # float
                        total_joints                     # int
                    ]
                    
                    writer.writerow(row)
        
        # Calculate statistics
        total_rows = sum(len(pose['joints']) for pose in self.all_converted_poses)
        logger.success(f"CSV saved: {total_rows} joint records across {len(self.all_converted_poses)} poses")
        logger.success(f"CSV output: {csv_path}")
    
    def save_toronto_gait_format(self, poses: List[Dict[str, Any]]) -> None:
        """Save pose detection results in Toronto gait analysis format."""
        # Generate Toronto gait filename
        toronto_filename = self.config.output_json.stem + "_toronto_gait.json"
        toronto_path = self.config.output_json.parent / toronto_filename
        
        logger.info(f"Saving Toronto gait format to {toronto_path}")
        
        from datetime import datetime
        
        # Organize data by person_id and frame
        gait_data = {}
        for pose in self.all_converted_poses:
            person_id = pose['person_id']
            frame_number = pose['frame_number']
            
            if person_id not in gait_data:
                gait_data[person_id] = {}
            
            if frame_number not in gait_data[person_id]:
                gait_data[person_id][frame_number] = {
                    'timestamp': pose['timestamp'],
                    'joints': {}
                }
            
            # Store joints with focus on gait-relevant keypoints
            for joint in pose['joints']:
                joint_name = joint['name']
                gait_data[person_id][frame_number]['joints'][joint_name] = {
                    'x': joint['keypoint']['x'],
                    'y': joint['keypoint']['y'],
                    'confidence': joint['keypoint']['confidence']
                }
        
        # Calculate gait metrics for each person
        gait_analysis = []
        for person_id, person_data in gait_data.items():
            frames = sorted(person_data.keys())
            if len(frames) < 10:  # Need minimum frames for gait analysis
                continue
            
            # Extract key gait landmarks over time
            left_hip_positions = []
            right_hip_positions = []
            left_knee_positions = []
            right_knee_positions = []
            left_ankle_positions = []
            right_ankle_positions = []
            timestamps = []
            
            for frame in frames:
                frame_data = person_data[frame]
                joints = frame_data['joints']
                timestamps.append(frame_data['timestamp'])
                
                # Extract positions for gait analysis
                if 'left_hip' in joints:
                    left_hip_positions.append([joints['left_hip']['x'], joints['left_hip']['y']])
                else:
                    left_hip_positions.append([0, 0])
                
                if 'right_hip' in joints:
                    right_hip_positions.append([joints['right_hip']['x'], joints['right_hip']['y']])
                else:
                    right_hip_positions.append([0, 0])
                
                if 'left_knee' in joints:
                    left_knee_positions.append([joints['left_knee']['x'], joints['left_knee']['y']])
                else:
                    left_knee_positions.append([0, 0])
                
                if 'right_knee' in joints:
                    right_knee_positions.append([joints['right_knee']['x'], joints['right_knee']['y']])
                else:
                    right_knee_positions.append([0, 0])
                
                if 'left_ankle' in joints:
                    left_ankle_positions.append([joints['left_ankle']['x'], joints['left_ankle']['y']])
                else:
                    left_ankle_positions.append([0, 0])
                
                if 'right_ankle' in joints:
                    right_ankle_positions.append([joints['right_ankle']['x'], joints['right_ankle']['y']])
                else:
                    right_ankle_positions.append([0, 0])
            
            # Calculate basic gait metrics
            def calculate_stride_length(positions):
                """Calculate approximate stride length from ankle positions."""
                if len(positions) < 2:
                    return 0
                distances = []
                for i in range(1, len(positions)):
                    if positions[i][0] != 0 and positions[i][1] != 0 and positions[i-1][0] != 0 and positions[i-1][1] != 0:
                        dx = positions[i][0] - positions[i-1][0]
                        dy = positions[i][1] - positions[i-1][1]
                        distances.append((dx**2 + dy**2)**0.5)
                return sum(distances) / len(distances) if distances else 0
            
            def calculate_step_frequency(positions, timestamps):
                """Calculate step frequency from position changes."""
                if len(positions) < 3 or len(timestamps) < 3:
                    return 0
                
                # Find peaks in vertical movement (y-axis)
                y_positions = [pos[1] for pos in positions if pos[1] != 0]
                if len(y_positions) < 3:
                    return 0
                
                peaks = 0
                for i in range(1, len(y_positions) - 1):
                    if y_positions[i] > y_positions[i-1] and y_positions[i] > y_positions[i+1]:
                        peaks += 1
                
                duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1
                return peaks / duration if duration > 0 else 0
            
            left_stride_length = calculate_stride_length(left_ankle_positions)
            right_stride_length = calculate_stride_length(right_ankle_positions)
            left_step_frequency = calculate_step_frequency(left_ankle_positions, timestamps)
            right_step_frequency = calculate_step_frequency(right_ankle_positions, timestamps)
            
            person_analysis = {
                'person_id': person_id,
                'total_frames': len(frames),
                'duration_seconds': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
                'gait_metrics': {
                    'stride_length': {
                        'left': round(left_stride_length, 4),
                        'right': round(right_stride_length, 4),
                        'asymmetry': round(abs(left_stride_length - right_stride_length), 4)
                    },
                    'step_frequency': {
                        'left': round(left_step_frequency, 4),
                        'right': round(right_step_frequency, 4),
                        'asymmetry': round(abs(left_step_frequency - right_step_frequency), 4)
                    }
                },
                'raw_data': {
                    'timestamps': timestamps,
                    'joint_trajectories': {
                        'left_hip': left_hip_positions,
                        'right_hip': right_hip_positions,
                        'left_knee': left_knee_positions,
                        'right_knee': right_knee_positions,
                        'left_ankle': left_ankle_positions,
                        'right_ankle': right_ankle_positions
                    }
                }
            }
            gait_analysis.append(person_analysis)
        
        # Build the complete Toronto gait format
        toronto_result = {
            'metadata': {
                'format': 'Toronto Gait Analysis v1.0',
                'input_file': str(self.config.input_path).replace('/', '\\'),
                'generated_at': datetime.now().isoformat(),
                'backend': self.backend_name,
                'processing_parameters': {
                    'confidence_threshold': self.config.min_confidence,
                    'net_resolution': getattr(self.config, 'net_resolution', None),
                    'model_pose': getattr(self.config, 'model_pose', None)
                },
                'video_metadata': self.video_metadata
            },
            'summary': {
                'total_people_analyzed': len(gait_analysis),
                'total_frames_processed': sum(analysis['total_frames'] for analysis in gait_analysis),
                'analysis_duration': sum(analysis['duration_seconds'] for analysis in gait_analysis)
            },
            'gait_analysis': gait_analysis
        }
        
        # Save Toronto gait format JSON
        with open(toronto_path, 'w') as f:
            json.dump(toronto_result, f, indent=2)
        
        logger.success(f"Toronto gait format saved: {len(gait_analysis)} people analyzed")
        logger.success(f"Toronto gait output: {toronto_path}")
    
    def _extract_comprehensive_frame_data(self, frame: np.ndarray, frame_idx: int, timestamp: float, poses: List[Dict[str, Any]]) -> None:
        """Extract comprehensive frame analysis data."""
        import numpy as np
        
        height, width = frame.shape[:2]
        
        # Basic frame statistics
        frame_stats = {
            'brightness': float(np.mean(frame)),
            'contrast': float(np.std(frame)),
            'sharpness': self._calculate_sharpness(frame),
            'motion_blur': self._calculate_motion_blur(frame, frame_idx)
        }
        
        # Pose analysis
        pose_analysis = {
            'total_poses': len(poses),
            'confidence_scores': [pose['score'] for pose in poses],
            'bounding_boxes': [pose['bbox'] for pose in poses],
            'pose_quality_metrics': []
        }
        
        # Calculate pose quality metrics
        for pose in poses:
            keypoints = pose['keypoints']
            valid_keypoints = sum(1 for kp in keypoints if kp[2] > 0.1)  # confidence > 0.1
            total_keypoints = len(keypoints)
            
            # Calculate pose completeness
            completeness = valid_keypoints / total_keypoints if total_keypoints > 0 else 0
            
            # Calculate keypoint spread (how well distributed the keypoints are)
            valid_kps = [kp for kp in keypoints if kp[2] > 0.1]
            if len(valid_kps) > 1:
                x_coords = [kp[0] for kp in valid_kps]
                y_coords = [kp[1] for kp in valid_kps]
                x_spread = max(x_coords) - min(x_coords)
                y_spread = max(y_coords) - min(y_coords)
                spread_ratio = (x_spread * y_spread) / (width * height)
            else:
                spread_ratio = 0
            
            # Calculate average confidence
            avg_confidence = sum(kp[2] for kp in valid_kps) / len(valid_kps) if valid_kps else 0
            
            pose_quality = {
                'completeness': round(completeness, 4),
                'spread_ratio': round(spread_ratio, 6),
                'average_confidence': round(avg_confidence, 4),
                'valid_keypoints': valid_keypoints,
                'total_keypoints': total_keypoints
            }
            pose_analysis['pose_quality_metrics'].append(pose_quality)
        
        # Motion analysis (if previous frame exists)
        motion_analysis = None
        if frame_idx > 0 and hasattr(self, '_previous_frame'):
            motion_analysis = self._calculate_frame_motion(self._previous_frame, frame)
        
        # Store current frame for next iteration
        self._previous_frame = frame.copy()
        
        # Comprehensive frame data
        comprehensive_data = {
            'frame_number': frame_idx,
            'timestamp': timestamp,
            'frame_statistics': frame_stats,
            'pose_analysis': pose_analysis,
            'motion_analysis': motion_analysis,
            'processing_metadata': {
                'backend': self.backend_name,
                'frame_dimensions': {'width': width, 'height': height},
                'extracted_at': timestamp
            }
        }
        
        self.comprehensive_frame_data.append(comprehensive_data)
        
        # Save individual frame analysis if verbose
        if self.config.verbose and frame_idx % 30 == 0:  # Log every 30 frames
            logger.debug(f"Frame {frame_idx}: {len(poses)} poses, avg confidence: {np.mean(pose_analysis['confidence_scores']) if pose_analysis['confidence_scores'] else 0:.3f}")
    
    def _calculate_sharpness(self, frame: np.ndarray) -> float:
        """Calculate frame sharpness using Laplacian variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    def _calculate_motion_blur(self, frame: np.ndarray, frame_idx: int) -> float:
        """Estimate motion blur in the frame."""
        if frame_idx == 0:
            return 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Use FFT to detect motion blur
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calculate the ratio of high-frequency to low-frequency components
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Define regions
        low_freq_region = magnitude_spectrum[center_h-10:center_h+10, center_w-10:center_w+10]
        high_freq_region = magnitude_spectrum[0:20, 0:20]  # Corner region
        
        low_freq_energy = np.mean(low_freq_region)
        high_freq_energy = np.mean(high_freq_region)
        
        # Motion blur indicator (higher value = more blur)
        blur_ratio = low_freq_energy / (high_freq_energy + 1e-6)
        return float(blur_ratio)
    
    def _calculate_frame_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Dict[str, Any]:
        """Calculate motion between consecutive frames."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray,
            np.array([[100, 100]], dtype=np.float32),  # Single point for overall motion
            None
        )
        
        # Calculate frame difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        motion_magnitude = float(np.mean(diff))
        
        # Calculate histogram correlation
        hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
        hist_curr = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
        correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
        
        return {
            'motion_magnitude': round(motion_magnitude, 4),
            'histogram_correlation': round(float(correlation), 4),
            'frame_similarity': round(1.0 - (motion_magnitude / 255.0), 4)
        }
    
    def save_comprehensive_frame_analysis(self) -> None:
        """Save comprehensive frame analysis to JSON file."""
        if not self.comprehensive_frame_data:
            return
        
        # Generate comprehensive analysis filename
        analysis_filename = self.config.output_json.stem + "_comprehensive_frames.json"
        analysis_path = self.config.output_json.parent / analysis_filename
        
        logger.info(f"Saving comprehensive frame analysis to {analysis_path}")
        
        from datetime import datetime
        
        # Calculate summary statistics
        total_frames = len(self.comprehensive_frame_data)
        total_poses = sum(data['pose_analysis']['total_poses'] for data in self.comprehensive_frame_data)
        avg_poses_per_frame = total_poses / total_frames if total_frames > 0 else 0
        
        all_confidences = []
        for data in self.comprehensive_frame_data:
            all_confidences.extend(data['pose_analysis']['confidence_scores'])
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        # Build comprehensive analysis result
        analysis_result = {
            'metadata': {
                'format': 'AmbientPose Comprehensive Frame Analysis v1.0',
                'input_file': str(self.config.input_path).replace('/', '\\'),
                'generated_at': datetime.now().isoformat(),
                'backend': self.backend_name,
                'processing_parameters': {
                    'confidence_threshold': self.config.min_confidence,
                    'net_resolution': getattr(self.config, 'net_resolution', None),
                    'model_pose': getattr(self.config, 'model_pose', None),
                    'comprehensive_extraction': True
                },
                'video_metadata': self.video_metadata
            },
            'summary': {
                'total_frames_analyzed': total_frames,
                'total_poses_detected': total_poses,
                'average_poses_per_frame': round(avg_poses_per_frame, 4),
                'average_pose_confidence': round(avg_confidence, 4),
                'analysis_completeness': 100.0  # Always 100% for processed frames
            },
            'frame_analysis': self.comprehensive_frame_data
        }
        
        # Save comprehensive analysis JSON
        with open(analysis_path, 'w') as f:
            json.dump(analysis_result, f, indent=2)
        
        logger.success(f"Comprehensive frame analysis saved: {total_frames} frames analyzed")
        logger.success(f"Comprehensive analysis output: {analysis_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for AmbientPose CLI.

    Options:
      --video <path>                   Path to input video file
      --image-dir <path>               Path to directory of input images
      --output <path>                  Path to output JSON file (default: outputs/pose_<timestamp>.json)
      --output-dir <path>              Directory for output files (default: outputs)
      --backend <name>                 Backend to use: auto, mediapipe, ultralytics, openpose, alphapose (default: auto)
      --min-confidence <float>         Minimum confidence threshold for detections (default: 0.5)
      --confidence-threshold <float>   Alias for --min-confidence (OpenPose compatibility)
      --net-resolution <WxH>           Network input resolution (e.g., 656x368, 832x512)
      --model-pose <name>              Pose model to use (backend-specific: COCO, BODY_25, MPI, etc.)
      --overlay-video <path>           Path to save overlay video file (MP4 format)
      --toronto-gait-format            Output results in Toronto gait analysis format
      --extract-comprehensive-frames   Extract comprehensive frame metadata and analysis
      --debug                          Enable debug mode
      --verbose                        Enable verbose logging with detailed information

    See docs/ADVANCED_CLI.md for full details and backend-specific notes.
    """
    parser = argparse.ArgumentParser(
        description='AmbientPose: Multi-backend pose detection tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video file')
    input_group.add_argument('--image-dir', type=str, help='Path to directory containing image files')
    
    # Output options
    parser.add_argument('--output', type=str, help='Path to output JSON file (default: outputs/pose_<timestamp>.json)')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory for output files')
    
    # Backend options
    parser.add_argument('--backend', type=str, default='auto', 
                        choices=['auto', 'mediapipe', 'ultralytics', 'openpose', 'alphapose'],
                        help='Pose detection backend to use')
    
    # Confidence options
    parser.add_argument('--min-confidence', type=float, default=0.5,
                        help='Minimum confidence threshold for detections')
    parser.add_argument('--confidence-threshold', type=float, dest='min_confidence',
                        help='Alias for --min-confidence (OpenPose compatibility)')
    
    # Advanced backend-specific options
    parser.add_argument('--net-resolution', type=str, default=None,
                        help='Network input resolution (format: WIDTHxHEIGHT, e.g., 656x368)')
    parser.add_argument('--model-pose', type=str, default=None,
                        help='Pose model to use (backend-specific: COCO, BODY_25, MPI, etc.)')
    
    # Video output options
    parser.add_argument('--overlay-video', type=str, default=None,
                        help='Path to save overlay video file (MP4 format)')
    
    # Output format options
    parser.add_argument('--toronto-gait-format', action='store_true', default=False,
                        help='Output results in Toronto gait analysis format')
    parser.add_argument('--extract-comprehensive-frames', action='store_true', default=False,
                        help='Extract comprehensive frame metadata and analysis')
    
    # Logging options
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging with detailed information')
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    start_time = time.time()
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Check if any backends are available
        if not (MEDIAPIPE_AVAILABLE or ULTRALYTICS_AVAILABLE or OPENPOSE_AVAILABLE or ALPHAPOSE_AVAILABLE):
            logger.error("No pose detection backends available!")
            logger.error("Please install at least one of the following:")
            logger.error("  - MediaPipe: pip install mediapipe")
            logger.error("  - Ultralytics: pip install ultralytics")
            logger.error("  - OpenPose: Set OPENPOSE_HOME environment variable")
            logger.error("  - AlphaPose: Follow instructions in docs/INSTALL.md")
            return 1
        
        # Initialize configuration
        config = AmbientPoseConfig(args)
        
        # Initialize pose detector
        detector = PoseDetector(config)
        
        # Process input
        if config.is_video:
            poses = detector.process_video()
        else:
            poses = detector.process_images()
        
        # Save results
        detector.save_results(poses)
        detector.save_csv_results(poses)
        
        # Save Toronto gait format if requested
        if config.toronto_gait_format:
            detector.save_toronto_gait_format(poses)
        
        # Save comprehensive frame analysis if requested
        if config.extract_comprehensive_frames:
            detector.save_comprehensive_frame_analysis()
        
        # Print summary
        elapsed_time = time.time() - start_time
        logger.success(f"Processing completed in {elapsed_time:.4f} seconds")
        logger.success(f"Output saved to: {config.output_json}")
        logger.success(f"Frames saved to: {config.frames_dir}")
        logger.success(f"Overlays saved to: {config.overlay_dir}")
        
        # Report additional outputs
        if config.overlay_video_path:
            logger.success(f"Overlay video saved to: {config.overlay_video_path}")
        if config.toronto_gait_format:
            toronto_path = config.output_json.parent / (config.output_json.stem + "_toronto_gait.json")
            logger.success(f"Toronto gait analysis saved to: {toronto_path}")
        if config.extract_comprehensive_frames:
            analysis_path = config.output_json.parent / (config.output_json.stem + "_comprehensive_frames.json")
            logger.success(f"Comprehensive frame analysis saved to: {analysis_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
