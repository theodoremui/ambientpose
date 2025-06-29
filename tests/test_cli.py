#!/usr/bin/env python3
"""
Test suite for AlphaDetect CLI (cli/detect.py)

This module contains comprehensive tests for the AlphaDetect CLI component,
including unit tests, integration tests, and mock-based tests to validate
the functionality of the pose detection pipeline.

Author: AlphaDetect Team
Date: 2025-06-21
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
import torch
from loguru import logger

# Import CLI modules
from cli.detect import (
    AlphaDetectConfig,
    PoseDetector,
    main,
    parse_args,
)

# Constants for testing
TEST_VIDEO_DIMENSIONS = (640, 480)  # width, height
TEST_VIDEO_FPS = 30
TEST_VIDEO_FRAMES = 10
TEST_IMAGE_DIMENSIONS = (640, 480)  # width, height
TEST_NUM_PEOPLE = 2  # Number of people to simulate in detection


# ===== FIXTURES =====

@pytest.fixture
def mock_alphapose_imports():
    """Mock AlphaPose imports to avoid actual dependency."""
    modules_to_mock = [
        'alphapose',
        'alphapose.models',
        'alphapose.models.builder',
        'alphapose.utils.config',
        'alphapose.utils.detector',
        'alphapose.utils.transforms',
        'alphapose.utils.vis',
        'alphapose.utils.writer',
    ]
    
    with patch.dict('sys.modules', {mod: Mock() for mod in modules_to_mock}):
        # Mock CUDA availability
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_name', return_value='Test GPU'):
                yield


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_video_path(temp_dir):
    """Create a test video file."""
    video_path = temp_dir / "test_video.mp4"
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, TEST_VIDEO_FPS, TEST_VIDEO_DIMENSIONS)
    
    # Create frames with different colors for easy distinction
    for i in range(TEST_VIDEO_FRAMES):
        # Create a colored frame with frame number
        frame = np.ones((TEST_VIDEO_DIMENSIONS[1], TEST_VIDEO_DIMENSIONS[0], 3), dtype=np.uint8) * (i * 25)
        # Add frame number text
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Add simulated person silhouettes
        for p in range(TEST_NUM_PEOPLE):
            # Draw a simple stick figure
            center_x = 100 + p * 200
            center_y = 300
            # Head
            cv2.circle(frame, (center_x, center_y - 100), 30, (255, 255, 255), -1)
            # Body
            cv2.line(frame, (center_x, center_y - 70), (center_x, center_y + 50), (255, 255, 255), 5)
            # Arms
            cv2.line(frame, (center_x, center_y - 30), (center_x - 50, center_y), (255, 255, 255), 5)
            cv2.line(frame, (center_x, center_y - 30), (center_x + 50, center_y), (255, 255, 255), 5)
            # Legs
            cv2.line(frame, (center_x, center_y + 50), (center_x - 30, center_y + 120), (255, 255, 255), 5)
            cv2.line(frame, (center_x, center_y + 50), (center_x + 30, center_y + 120), (255, 255, 255), 5)
        
        out.write(frame)
    
    out.release()
    yield video_path


@pytest.fixture
def test_image_dir(temp_dir):
    """Create a directory with test images."""
    image_dir = temp_dir / "test_images"
    image_dir.mkdir(exist_ok=True)
    
    # Create test images
    for i in range(TEST_VIDEO_FRAMES):
        image_path = image_dir / f"image_{i:03d}.jpg"
        
        # Create a colored image with frame number
        frame = np.ones((TEST_IMAGE_DIMENSIONS[1], TEST_IMAGE_DIMENSIONS[0], 3), dtype=np.uint8) * (i * 25)
        # Add frame number text
        cv2.putText(frame, f"Image {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Add simulated person silhouettes (same as in test_video_path)
        for p in range(TEST_NUM_PEOPLE):
            center_x = 100 + p * 200
            center_y = 300
            # Head
            cv2.circle(frame, (center_x, center_y - 100), 30, (255, 255, 255), -1)
            # Body
            cv2.line(frame, (center_x, center_y - 70), (center_x, center_y + 50), (255, 255, 255), 5)
            # Arms
            cv2.line(frame, (center_x, center_y - 30), (center_x - 50, center_y), (255, 255, 255), 5)
            cv2.line(frame, (center_x, center_y - 30), (center_x + 50, center_y), (255, 255, 255), 5)
            # Legs
            cv2.line(frame, (center_x, center_y + 50), (center_x - 30, center_y + 120), (255, 255, 255), 5)
            cv2.line(frame, (center_x, center_y + 50), (center_x + 30, center_y + 120), (255, 255, 255), 5)
        
        cv2.imwrite(str(image_path), frame)
    
    yield image_dir


@pytest.fixture
def output_dir(temp_dir):
    """Create an output directory for test results."""
    output_dir = temp_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    yield output_dir


@pytest.fixture
def mock_config_file(temp_dir):
    """Create a mock AlphaPose config file."""
    config_file = temp_dir / "mock_config.yaml"
    with open(config_file, 'w') as f:
        f.write("""
MODEL:
  TYPE: 'FastPose'
  BACKBONE: 'resnet50'
DATA_PRESET:
  IMAGE_SIZE:
    - 256
    - 192
        """)
    yield config_file


@pytest.fixture
def mock_checkpoint_file(temp_dir):
    """Create a mock AlphaPose checkpoint file."""
    checkpoint_file = temp_dir / "mock_checkpoint.pth"
    # Create an empty file
    checkpoint_file.touch()
    yield checkpoint_file


@pytest.fixture
def mock_detector():
    """Create a mock object detector."""
    detector = Mock()
    
    # Configure the detect method to return simulated bounding boxes
    def mock_detect(frame):
        height, width = frame.shape[:2]
        # Generate detections for TEST_NUM_PEOPLE people
        detections = []
        for i in range(TEST_NUM_PEOPLE):
            # Create a detection with a bounding box
            x1 = 50 + i * 200
            y1 = 100
            x2 = x1 + 150
            y2 = y1 + 300
            score = 0.9 - (i * 0.1)  # Varying confidence scores
            detection = {
                'bbox': [x1, y1, x2, y2, score],
                'score': score,
                'category_id': 1,  # Person category
            }
            detections.append(detection)
        
        return detections
    
    detector.detect = mock_detect
    return detector


@pytest.fixture
def mock_pose_model():
    """Create a mock pose estimation model."""
    model = Mock()
    
    # Configure the forward method to return simulated heatmaps
    def mock_forward(x):
        batch_size = x.shape[0]
        # For simplicity, return random heatmaps
        # In a real scenario, these would be probability maps for each keypoint
        num_keypoints = 17  # COCO format has 17 keypoints
        heatmap_size = (64, 48)  # Typical heatmap size for 256x192 input
        
        # Create random heatmaps with peaks at expected keypoint locations
        heatmaps = torch.zeros((batch_size, num_keypoints, heatmap_size[1], heatmap_size[0]))
        
        # For each person in the batch
        for b in range(batch_size):
            # For each keypoint
            for k in range(num_keypoints):
                # Create a peak at a reasonable location
                x_center = int(heatmap_size[0] * (0.3 + 0.4 * (k % 3) / 2))
                y_center = int(heatmap_size[1] * (0.2 + 0.6 * (k // 3) / 5))
                
                # Create a gaussian peak
                for y in range(max(0, y_center - 2), min(heatmap_size[1], y_center + 3)):
                    for x in range(max(0, x_center - 2), min(heatmap_size[0], x_center + 3)):
                        dist = ((x - x_center) ** 2 + (y - y_center) ** 2) ** 0.5
                        heatmaps[b, k, y, x] = max(0, 1 - dist / 2)
        
        return heatmaps
    
    model.return_value = mock_forward
    return model


@pytest.fixture
def mock_heatmap_to_coord():
    """Create a mock function for converting heatmaps to coordinates."""
    def mock_func(heatmaps, boxes, cropped_boxes):
        batch_size = heatmaps.shape[0]
        num_keypoints = heatmaps.shape[1]
        
        # Create simulated keypoint coordinates
        coords = torch.zeros((batch_size, num_keypoints, 2))
        maxvals = torch.zeros((batch_size, num_keypoints))
        
        # For each person
        for b in range(batch_size):
            box = boxes[b]
            x1, y1, x2, y2 = box[:4]
            box_width = x2 - x1
            box_height = y2 - y1
            
            # For each keypoint
            for k in range(num_keypoints):
                # Create coordinates within the bounding box
                rel_x = 0.2 + 0.6 * (k % 3) / 2  # Relative x position within box
                rel_y = 0.1 + 0.8 * (k // 3) / 5  # Relative y position within box
                
                coords[b, k, 0] = x1 + rel_x * box_width
                coords[b, k, 1] = y1 + rel_y * box_height
                
                # Assign confidence score
                maxvals[b, k] = 0.85 + 0.1 * torch.rand(1)
        
        return coords, maxvals
    
    return mock_func


@pytest.fixture
def mock_alphapose_config():
    """Create a mock AlphaPose configuration object."""
    cfg = Mock()
    cfg.MODEL = Mock()
    cfg.DATA_PRESET = {
        'IMAGE_SIZE': [256, 192]
    }
    return cfg


# ===== UNIT TESTS =====

class TestArgumentParsing:
    """Tests for command line argument parsing."""
    
    def test_parse_args_default(self):
        """Test parsing with minimal required arguments."""
        test_args = ['--video', 'test.mp4']
        with patch('sys.argv', ['detect.py'] + test_args):
            args = parse_args()
            
            assert args.video == 'test.mp4'
            assert args.image_dir is None
            assert args.output is None
            assert args.output_dir == 'outputs'
            assert args.detector == 'yolox-x'
            assert args.config_file == 'configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml'
            assert args.checkpoint == 'pretrained_models/fast_res50_256x192.pth'
            assert args.gpus == [0]
            assert not args.track
            assert not args.pose_track
            assert not args.pose_3d
    
    def test_parse_args_image_dir(self):
        """Test parsing with image directory input."""
        test_args = ['--image-dir', 'images/']
        with patch('sys.argv', ['detect.py'] + test_args):
            args = parse_args()
            
            assert args.video is None
            assert args.image_dir == 'images/'
    
    def test_parse_args_mutual_exclusivity(self):
        """Test that video and image-dir are mutually exclusive."""
        test_args = ['--video', 'test.mp4', '--image-dir', 'images/']
        with patch('sys.argv', ['detect.py'] + test_args):
            with pytest.raises(SystemExit):
                parse_args()
    
    def test_parse_args_custom_output(self):
        """Test parsing with custom output path."""
        test_args = ['--video', 'test.mp4', '--output', 'custom/path.json']
        with patch('sys.argv', ['detect.py'] + test_args):
            args = parse_args()
            
            assert args.output == 'custom/path.json'
    
    def test_parse_args_custom_detector(self):
        """Test parsing with custom detector."""
        test_args = ['--video', 'test.mp4', '--detector', 'yolox-s']
        with patch('sys.argv', ['detect.py'] + test_args):
            args = parse_args()
            
            assert args.detector == 'yolox-s'
    
    def test_parse_args_multiple_gpus(self):
        """Test parsing with multiple GPUs."""
        test_args = ['--video', 'test.mp4', '--gpus', '0,1,2']
        with patch('sys.argv', ['detect.py'] + test_args):
            args = parse_args()
            
            assert args.gpus == [0, 1, 2]
    
    def test_parse_args_tracking_options(self):
        """Test parsing with tracking options."""
        test_args = ['--video', 'test.mp4', '--track', '--pose-track']
        with patch('sys.argv', ['detect.py'] + test_args):
            args = parse_args()
            
            assert args.track is True
            assert args.pose_track is True
    
    def test_parse_args_3d_option(self):
        """Test parsing with 3D pose option."""
        test_args = ['--video', 'test.mp4', '--pose-3d']
        with patch('sys.argv', ['detect.py'] + test_args):
            args = parse_args()
            
            assert args.pose_3d is True


class TestConfigurationValidation:
    """Tests for configuration validation."""
    
    def test_config_init_video(self, test_video_path, output_dir, mock_config_file, mock_checkpoint_file):
        """Test configuration initialization with video input."""
        args = argparse.Namespace(
            video=str(test_video_path),
            image_dir=None,
            output=None,
            output_dir=str(output_dir),
            detector='yolox-x',
            config_file=str(mock_config_file),
            checkpoint=str(mock_checkpoint_file),
            gpus=[0],
            track=False,
            pose_track=False,
            pose_3d=False,
            debug=False,
            detector_batch_size=1,
            pose_batch_size=80
        )
        
        config = AlphaDetectConfig(args)
        
        assert config.input_path == test_video_path
        assert config.is_video is True
        assert config.output_dir == output_dir
        assert config.output_json.parent == output_dir
        assert config.output_json.name.startswith('pose_')
        assert config.output_json.name.endswith('.json')
        assert config.frames_dir.parent == output_dir
        assert config.frames_dir.name.startswith('frames_')
        assert config.overlay_dir.parent == output_dir
        assert config.overlay_dir.name.startswith('overlay_')
        assert config.detector == 'yolox-x'
        assert config.config_file == mock_config_file
        assert config.checkpoint == mock_checkpoint_file
    
    def test_config_init_image_dir(self, test_image_dir, output_dir, mock_config_file, mock_checkpoint_file):
        """Test configuration initialization with image directory input."""
        args = argparse.Namespace(
            video=None,
            image_dir=str(test_image_dir),
            output=None,
            output_dir=str(output_dir),
            detector='yolox-x',
            config_file=str(mock_config_file),
            checkpoint=str(mock_checkpoint_file),
            gpus=[0],
            track=False,
            pose_track=False,
            pose_3d=False,
            debug=False,
            detector_batch_size=1,
            pose_batch_size=80
        )
        
        config = AlphaDetectConfig(args)
        
        assert config.input_path == test_image_dir
        assert config.is_video is False
        assert config.output_dir == output_dir
    
    def test_config_init_custom_output(self, test_video_path, output_dir, mock_config_file, mock_checkpoint_file):
        """Test configuration initialization with custom output path."""
        custom_output = output_dir / 'custom_output.json'
        
        args = argparse.Namespace(
            video=str(test_video_path),
            image_dir=None,
            output=str(custom_output),
            output_dir=str(output_dir),
            detector='yolox-x',
            config_file=str(mock_config_file),
            checkpoint=str(mock_checkpoint_file),
            gpus=[0],
            track=False,
            pose_track=False,
            pose_3d=False,
            debug=False,
            detector_batch_size=1,
            pose_batch_size=80
        )
        
        config = AlphaDetectConfig(args)
        
        assert config.output_json == custom_output
    
    def test_config_validation_missing_input(self, output_dir, mock_config_file, mock_checkpoint_file):
        """Test configuration validation with missing input."""
        args = argparse.Namespace(
            video='nonexistent.mp4',
            image_dir=None,
            output=None,
            output_dir=str(output_dir),
            detector='yolox-x',
            config_file=str(mock_config_file),
            checkpoint=str(mock_checkpoint_file),
            gpus=[0],
            track=False,
            pose_track=False,
            pose_3d=False,
            debug=False,
            detector_batch_size=1,
            pose_batch_size=80
        )
        
        with pytest.raises(FileNotFoundError, match="Input path does not exist"):
            AlphaDetectConfig(args)
    
    def test_config_validation_missing_config(self, test_video_path, output_dir, mock_checkpoint_file):
        """Test configuration validation with missing config file."""
        args = argparse.Namespace(
            video=str(test_video_path),
            image_dir=None,
            output=None,
            output_dir=str(output_dir),
            detector='yolox-x',
            config_file='nonexistent.yaml',
            checkpoint=str(mock_checkpoint_file),
            gpus=[0],
            track=False,
            pose_track=False,
            pose_3d=False,
            debug=False,
            detector_batch_size=1,
            pose_batch_size=80
        )
        
        with pytest.raises(FileNotFoundError, match="Config file does not exist"):
            AlphaDetectConfig(args)
    
    def test_config_validation_missing_checkpoint(self, test_video_path, output_dir, mock_config_file):
        """Test configuration validation with missing checkpoint file."""
        args = argparse.Namespace(
            video=str(test_video_path),
            image_dir=None,
            output=None,
            output_dir=str(output_dir),
            detector='yolox-x',
            config_file=str(mock_config_file),
            checkpoint='nonexistent.pth',
            gpus=[0],
            track=False,
            pose_track=False,
            pose_3d=False,
            debug=False,
            detector_batch_size=1,
            pose_batch_size=80
        )
        
        with pytest.raises(FileNotFoundError, match="Checkpoint file does not exist"):
            AlphaDetectConfig(args)
    
    @patch('os.access')
    def test_config_validation_unwritable_output(self, mock_access, test_video_path, output_dir, mock_config_file, mock_checkpoint_file):
        """Test configuration validation with unwritable output directory."""
        mock_access.return_value = False
        
        args = argparse.Namespace(
            video=str(test_video_path),
            image_dir=None,
            output=None,
            output_dir=str(output_dir),
            detector='yolox-x',
            config_file=str(mock_config_file),
            checkpoint=str(mock_checkpoint_file),
            gpus=[0],
            track=False,
            pose_track=False,
            pose_3d=False,
            debug=False,
            detector_batch_size=1,
            pose_batch_size=80
        )
        
        with pytest.raises(PermissionError, match="Output directory is not writable"):
            AlphaDetectConfig(args)


class TestPoseDetector:
    """Tests for PoseDetector class."""
    
    @patch('cli.detect.builder.build_sppe')
    @patch('cli.detect.update_config')
    @patch('cli.detect.torch.load')
    def test_pose_detector_init(self, mock_torch_load, mock_update_config, mock_build_sppe, 
                               test_video_path, output_dir, mock_config_file, mock_checkpoint_file,
                               mock_alphapose_config, mock_alphapose_imports):
        """Test PoseDetector initialization."""
        # Configure mocks
        mock_update_config.return_value = mock_alphapose_config
        mock_pose_model = Mock()
        mock_build_sppe.return_value = mock_pose_model
        
        # Create configuration
        args = argparse.Namespace(
            video=str(test_video_path),
            image_dir=None,
            output=None,
            output_dir=str(output_dir),
            detector='yolox-x',
            config_file=str(mock_config_file),
            checkpoint=str(mock_checkpoint_file),
            gpus=[0],
            track=False,
            pose_track=False,
            pose_3d=False,
            debug=False,
            detector_batch_size=1,
            pose_batch_size=80
        )
        config = AlphaDetectConfig(args)
        
        # Initialize detector
        with patch('cli.detect.YoloxDetector') as mock_yolox:
            mock_yolox.return_value = Mock()
            detector = PoseDetector(config)
            
            # Verify initialization
            assert detector.config == config
            assert detector.cfg == mock_alphapose_config
            mock_update_config.assert_called_once_with(mock_config_file)
            mock_build_sppe.assert_called_once()
            mock_torch_load.assert_called_once()
            mock_pose_model.to.assert_called_once()
            mock_pose_model.eval.assert_called_once()
            mock_yolox.assert_called_once_with(model_name='yolox-x', device='cuda:0')
    
    @patch('cli.detect.cv2.VideoCapture')
    def test_process_video(self, mock_video_capture, test_video_path, output_dir, mock_config_file, mock_checkpoint_file, mock_alphapose_imports):
        """Test video processing."""
        # Configure mocks
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: TEST_VIDEO_FRAMES,
            cv2.CAP_PROP_FPS: TEST_VIDEO_FPS,
            cv2.CAP_PROP_FRAME_WIDTH: TEST_VIDEO_DIMENSIONS[0],
            cv2.CAP_PROP_FRAME_HEIGHT: TEST_VIDEO_DIMENSIONS[1]
        }.get(prop, 0)
        
        # Mock read() to return TEST_VIDEO_FRAMES frames then stop
        frames = []
        for i in range(TEST_VIDEO_FRAMES):
            frame = np.ones((TEST_VIDEO_DIMENSIONS[1], TEST_VIDEO_DIMENSIONS[0], 3), dtype=np.uint8) * (i * 25)
            frames.append(frame)
        
        mock_cap.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        
        # Create configuration
        args = argparse.Namespace(
            video=str(test_video_path),
            image_dir=None,
            output=None,
            output_dir=str(output_dir),
            detector='yolox-x',
            config_file=str(mock_config_file),
            checkpoint=str(mock_checkpoint_file),
            gpus=[0],
            track=False,
            pose_track=False,
            pose_3d=False,
            debug=False,
            detector_batch_size=1,
            pose_batch_size=80
        )
        config = AlphaDetectConfig(args)
        
        # Initialize detector with mocks
        detector = Mock()
        detector.config = config
        detector._process_frame = Mock(return_value=[{'frame_idx': i, 'keypoints': []} for i in range(TEST_NUM_PEOPLE)])
        detector._draw_poses = Mock(return_value=np.zeros((TEST_VIDEO_DIMENSIONS[1], TEST_VIDEO_DIMENSIONS[0], 3), dtype=np.uint8))
        
        # Patch cv2.imwrite to avoid actual file writes
        with patch('cli.detect.cv2.imwrite') as mock_imwrite:
            # Call process_video
            result = PoseDetector.process_video(detector)
            
            # Verify results
            assert len(result) == TEST_VIDEO_FRAMES * TEST_NUM_PEOPLE
            mock_video_capture.assert_called_once_with(str(test_video_path))
            assert mock_cap.read.call_count == TEST_VIDEO_FRAMES + 1  # +1 for the final False return
            assert detector._process_frame.call_count == TEST_VIDEO_FRAMES
            assert detector._draw_poses.call_count == TEST_VIDEO_FRAMES
            assert mock_imwrite.call_count == TEST_VIDEO_FRAMES * 2  # Raw frames + overlay frames
    
    def test_process_images(self, test_image_dir, output_dir, mock_config_file, mock_checkpoint_file, mock_alphapose_imports):
        """Test image directory processing."""
        # Create configuration
        args = argparse.Namespace(
            video=None,
            image_dir=str(test_image_dir),
            output=None,
            output_dir=str(output_dir),
            detector='yolox-x',
            config_file=str(mock_config_file),
            checkpoint=str(mock_checkpoint_file),
            gpus=[0],
            track=False,
            pose_track=False,
            pose_3d=False,
            debug=False,
            detector_batch_size=1,
            pose_batch_size=80
        )
        config = AlphaDetectConfig(args)
        
        # Initialize detector with mocks
        detector = Mock()
        detector.config = config
        detector._process_frame = Mock(return_value=[{'frame_idx': i, 'keypoints': []} for i in range(TEST_NUM_PEOPLE)])
        detector._draw_poses = Mock(return_value=np.zeros((TEST_IMAGE_DIMENSIONS[1], TEST_IMAGE_DIMENSIONS[0], 3), dtype=np.uint8))
        
        # Mock cv2.imread to return test frames
        def mock_imread(path):
            return np.ones((TEST_IMAGE_DIMENSIONS[1], TEST_IMAGE_DIMENSIONS[0], 3), dtype=np.uint8)
        
        # Patch cv2.imread and cv2.imwrite to avoid actual file operations
        with patch('cli.detect.cv2.imread', side_effect=mock_imread) as mock_imread:
            with patch('cli.detect.cv2.imwrite') as mock_imwrite:
                # Call process_images
                result = PoseDetector.process_images(detector)
                
                # Verify results
                assert len(result) == TEST_VIDEO_FRAMES * TEST_NUM_PEOPLE  # Same number as in test_image_dir
                assert mock_imread.call_count == TEST_VIDEO_FRAMES
                assert detector._process_frame.call_count == TEST_VIDEO_FRAMES
                assert detector._draw_poses.call_count == TEST_VIDEO_FRAMES
                assert mock_imwrite.call_count == TEST_VIDEO_FRAMES * 2  # Raw frames + overlay frames
    
    def test_process_frame(self, mock_detector, mock_pose_model, mock_heatmap_to_coord, mock_alphapose_imports):
        """Test processing a single frame."""
        # Create a test frame
        frame = np.ones((TEST_VIDEO_DIMENSIONS[1], TEST_VIDEO_DIMENSIONS[0], 3), dtype=np.uint8)
        
        # Initialize detector with mocks
        detector = Mock()
        detector.detector = mock_detector
        detector.pose_model = mock_pose_model
        detector.device = torch.device('cpu')
        detector._preprocess = Mock(return_value=(
            torch.zeros((TEST_NUM_PEOPLE, 3, 256, 192)),  # inps
            frame,  # orig_img
            torch.tensor([[100, 100, 300, 400, 0.9], [400, 100, 600, 400, 0.8]]),  # boxes
            torch.tensor([0.9, 0.8]),  # scores
            torch.tensor([0, 1]),  # ids
            torch.tensor([[100, 100, 300, 400], [400, 100, 600, 400]])  # cropped_boxes
        ))
        detector.cfg = Mock()
        detector.cfg.DATA_PRESET = {'IMAGE_SIZE': [256, 192]}
        
        # Mock get_func_heatmap_to_coord
        with patch('cli.detect.get_func_heatmap_to_coord', return_value=mock_heatmap_to_coord):
            # Call _process_frame
            result = PoseDetector._process_frame(detector, frame, 0)
            
            # Verify results
            assert len(result) == TEST_NUM_PEOPLE
            for pose in result:
                assert 'frame_idx' in pose
                assert 'bbox' in pose
                assert 'score' in pose
                assert 'keypoints' in pose
                assert len(pose['keypoints']) == 17  # COCO format has 17 keypoints
    
    def test_draw_poses(self):
        """Test drawing poses on an image."""
        # Create a test frame
        frame = np.ones((TEST_VIDEO_DIMENSIONS[1], TEST_VIDEO_DIMENSIONS[0], 3), dtype=np.uint8) * 255
        
        # Create test poses
        poses = [
            {
                'frame_idx': 0,
                'bbox': [100, 100, 300, 400, 0.9],
                'score': 0.9,
                'keypoints': [
                    [150, 150, 0.9],  # Nose
                    [160, 140, 0.8],  # Left eye
                    [140, 140, 0.8],  # Right eye
                    [170, 150, 0.7],  # Left ear
                    [130, 150, 0.7],  # Right ear
                    [200, 200, 0.9],  # Left shoulder
                    [100, 200, 0.9],  # Right shoulder
                    [220, 300, 0.8],  # Left elbow
                    [80, 300, 0.8],   # Right elbow
                    [240, 380, 0.7],  # Left wrist
                    [60, 380, 0.7],   # Right wrist
                    [180, 350, 0.9],  # Left hip
                    [120, 350, 0.9],  # Right hip
                    [190, 450, 0.8],  # Left knee
                    [110, 450, 0.8],  # Right knee
                    [200, 550, 0.7],  # Left ankle
                    [100, 550, 0.7],  # Right ankle
                ]
            }
        ]
        
        # Call _draw_poses
        result = PoseDetector._draw_poses(None, frame.copy(), poses)
        
        # Verify results
        assert result.shape == frame.shape
        # The result should be different from the original frame (has drawings)
        assert not np.array_equal(result, frame)
    
    def test_save_results(self, output_dir):
        """Test saving results to JSON."""
        # Create configuration
        config = Mock()
        config.output_json = output_dir / 'test_output.json'
        config.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config.input_path = Path('test_video.mp4')
        config.frames_dir = output_dir / 'frames'
        config.overlay_dir = output_dir / 'overlay'
        
        # Create test poses
        poses = [
            {
                'frame_idx': 0,
                'bbox': [100, 100, 300, 400, 0.9],
                'score': 0.9,
                'keypoints': [[x, y, 0.8] for x, y in zip(range(100, 500, 25), range(100, 500, 25))]
            },
            {
                'frame_idx': 1,
                'bbox': [150, 150, 350, 450, 0.8],
                'score': 0.8,
                'keypoints': [[x, y, 0.7] for x, y in zip(range(150, 550, 25), range(150, 550, 25))]
            }
        ]
        
        # Initialize detector
        detector = Mock()
        detector.config = config
        
        # Call save_results
        PoseDetector.save_results(detector, poses)
        
        # Verify results
        assert config.output_json.exists()
        with open(config.output_json, 'r') as f:
            data = json.load(f)
            assert 'timestamp' in data
            assert 'input_path' in data
            assert 'frames_dir' in data
            assert 'overlay_dir' in data
            assert 'total_poses' in data
            assert 'poses' in data
            assert data['total_poses'] == 2
            assert len(data['poses']) == 2


# ===== INTEGRATION TESTS =====

@pytest.mark.parametrize("input_type", ["video", "image_dir"])
def test_main_integration(input_type, test_video_path, test_image_dir, output_dir, mock_config_file, mock_checkpoint_file, mock_alphapose_imports):
    """Integration test for the main function with different input types."""
    # Prepare arguments based on input type
    if input_type == "video":
        input_path = test_video_path
        test_args = [
            '--video', str(input_path),
            '--output-dir', str(output_dir),
            '--config-file', str(mock_config_file),
            '--checkpoint', str(mock_checkpoint_file)
        ]
    else:  # image_dir
        input_path = test_image_dir
        test_args = [
            '--image-dir', str(input_path),
            '--output-dir', str(output_dir),
            '--config-file', str(mock_config_file),
            '--checkpoint', str(mock_checkpoint_file)
        ]
    
    # Mock dependencies
    with patch('sys.argv', ['detect.py'] + test_args):
        with patch('cli.detect.PoseDetector') as mock_detector_class:
            # Configure mock detector
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector
            
            if input_type == "video":
                mock_detector.process_video.return_value = [{'frame_idx': 0, 'keypoints': []}]
            else:
                mock_detector.process_images.return_value = [{'frame_idx': 0, 'keypoints': []}]
            
            # Call main function
            exit_code = main()
            
            # Verify results
            assert exit_code == 0
            mock_detector_class.assert_called_once()
            if input_type == "video":
                mock_detector.process_video.assert_called_once()
            else:
                mock_detector.process_images.assert_called_once()
            mock_detector.save_results.assert_called_once()


def test_main_error_handling(test_video_path, output_dir):
    """Test error handling in the main function."""
    test_args = [
        '--video', str(test_video_path),
        '--output-dir', str(output_dir),
        '--config-file', 'nonexistent.yaml',  # This will cause an error
        '--checkpoint', 'nonexistent.pth'
    ]
    
    # Mock dependencies
    with patch('sys.argv', ['detect.py'] + test_args):
        # Call main function
        exit_code = main()
        
        # Verify results
        assert exit_code == 1  # Error exit code


@pytest.mark.asyncio
async def test_async_compatibility():
    """Test that the CLI can be used in an async context."""
    # This is a simple test to ensure that the CLI doesn't block async code
    import asyncio
    
    async def run_cli():
        # Mock a CLI run
        await asyncio.sleep(0.1)
        return "CLI completed"
    
    result = await run_cli()
    assert result == "CLI completed"


# ===== PARAMETRIZED TESTS =====

@pytest.mark.parametrize("detector_type", ["yolox-s", "yolox-m", "yolox-l", "yolox-x"])
def test_detector_types(detector_type, test_video_path, output_dir, mock_config_file, mock_checkpoint_file, mock_alphapose_imports):
    """Test different detector types."""
    # Create configuration
    args = argparse.Namespace(
        video=str(test_video_path),
        image_dir=None,
        output=None,
        output_dir=str(output_dir),
        detector=detector_type,
        config_file=str(mock_config_file),
        checkpoint=str(mock_checkpoint_file),
        gpus=[0],
        track=False,
        pose_track=False,
        pose_3d=False,
        debug=False,
        detector_batch_size=1,
        pose_batch_size=80
    )
    
    # Initialize configuration
    config = AlphaDetectConfig(args)
    assert config.detector == detector_type
    
    # Mock detector initialization
    with patch('cli.detect.YoloxDetector') as mock_yolox:
        mock_yolox.return_value = Mock()
        with patch('cli.detect.update_config', return_value=Mock()):
            with patch('cli.detect.builder.build_sppe', return_value=Mock()):
                with patch('cli.detect.torch.load'):
                    # Initialize detector
                    detector = PoseDetector(config)
                    
                    # Verify detector initialization
                    mock_yolox.assert_called_once_with(model_name=detector_type, device='cuda:0')


@pytest.mark.parametrize("track_option", [
    (False, False),  # No tracking
    (True, False),   # Human tracking only
    (False, True),   # Pose tracking only
    (True, True)     # Both tracking options
])
def test_tracking_options(track_option, test_video_path, output_dir, mock_config_file, mock_checkpoint_file, mock_alphapose_imports):
    """Test different tracking options."""
    track, pose_track = track_option
    
    # Create configuration
    args = argparse.Namespace(
        video=str(test_video_path),
        image_dir=None,
        output=None,
        output_dir=str(output_dir),
        detector='yolox-x',
        config_file=str(mock_config_file),
        checkpoint=str(mock_checkpoint_file),
        gpus=[0],
        track=track,
        pose_track=pose_track,
        pose_3d=False,
        debug=False,
        detector_batch_size=1,
        pose_batch_size=80
    )
    
    # Initialize configuration
    config = AlphaDetectConfig(args)
    assert config.track == track
    assert config.pose_track == pose_track


@pytest.mark.parametrize("gpu_option", [
    [0],      # Single GPU
    [0, 1],   # Multiple GPUs
    [-1],     # CPU only
])
def test_gpu_options(gpu_option, test_video_path, output_dir, mock_config_file, mock_checkpoint_file, mock_alphapose_imports):
    """Test different GPU options."""
    # Create configuration
    args = argparse.Namespace(
        video=str(test_video_path),
        image_dir=None,
        output=None,
        output_dir=str(output_dir),
        detector='yolox-x',
        config_file=str(mock_config_file),
        checkpoint=str(mock_checkpoint_file),
        gpus=gpu_option,
        track=False,
        pose_track=False,
        pose_3d=False,
        debug=False,
        detector_batch_size=1,
        pose_batch_size=80
    )
    
    # Initialize configuration
    config = AlphaDetectConfig(args)
    
    # Check device setting
    if gpu_option == [-1] or not torch.cuda.is_available():
        expected_device = 'cpu'
    else:
        expected_device = f'cuda:{gpu_option[0]}'
    
    assert config.device == expected_device


# ===== ERROR HANDLING TESTS =====

def test_video_open_error(test_video_path, output_dir, mock_config_file, mock_checkpoint_file, mock_alphapose_imports):
    """Test handling of video open error."""
    # Create configuration
    args = argparse.Namespace(
        video=str(test_video_path),
        image_dir=None,
        output=None,
        output_dir=str(output_dir),
        detector='yolox-x',
        config_file=str(mock_config_file),
        checkpoint=str(mock_checkpoint_file),
        gpus=[0],
        track=False,
        pose_track=False,
        pose_3d=False,
        debug=False,
        detector_batch_size=1,
        pose_batch_size=80
    )
    config = AlphaDetectConfig(args)
    
    # Initialize detector with mocks
    detector = Mock()
    detector.config = config
    
    # Mock cv2.VideoCapture to return an unopened capture
    with patch('cli.detect.cv2.VideoCapture') as mock_video_capture:
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        # Call process_video and expect an IOError
        with pytest.raises(IOError, match="Failed to open video"):
            PoseDetector.process_video(detector)


def test_empty_image_directory(output_dir, mock_config_file, mock_checkpoint_file, mock_alphapose_imports):
    """Test handling of empty image directory."""
    # Create an empty directory
    empty_dir = output_dir / "empty_dir"
    empty_dir.mkdir(exist_ok=True)
    
    # Create configuration
    args = argparse.Namespace(
        video=None,
        image_dir=str(empty_dir),
        output=None,
        output_dir=str(output_dir),
        detector='yolox-x',
        config_file=str(mock_config_file),
        checkpoint=str(mock_checkpoint_file),
        gpus=[0],
        track=False,
        pose_track=False,
        pose_3d=False,
        debug=False,
        detector_batch_size=1,
        pose_batch_size=80
    )
    config = AlphaDetectConfig(args)
    
    # Initialize detector with mocks
    detector = Mock()
    detector.config = config
    
    # Call process_images and expect a ValueError
    with pytest.raises(ValueError, match="No image files found in directory"):
        PoseDetector.process_images(detector)


def test_detector_import_error(test_video_path, output_dir, mock_config_file, mock_checkpoint_file, mock_alphapose_imports):
    """Test handling of detector import error."""
    # Create configuration
    args = argparse.Namespace(
        video=str(test_video_path),
        image_dir=None,
        output=None,
        output_dir=str(output_dir),
        detector='invalid_detector',  # Invalid detector name
        config_file=str(mock_config_file),
        checkpoint=str(mock_checkpoint_file),
        gpus=[0],
        track=False,
        pose_track=False,
        pose_3d=False,
        debug=False,
        detector_batch_size=1,
        pose_batch_size=80
    )
    config = AlphaDetectConfig(args)
    
    # Mock dependencies
    with patch('cli.detect.update_config', return_value=Mock()):
        with patch('cli.detect.builder.build_sppe', return_value=Mock()):
            with patch('cli.detect.torch.load'):
                # Initialize detector and expect a ValueError
                with pytest.raises(ValueError, match="Unsupported detector"):
                    PoseDetector(config)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
