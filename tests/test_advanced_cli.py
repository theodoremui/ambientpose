"""
Tests for advanced CLI functionality in AmbientPose.

This module tests:
- Advanced CLI parameter parsing and validation
- Backend-specific configuration handling  
- Overlay video generation
- Toronto gait format output
- Comprehensive frame extraction
- Verbose logging functionality
"""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import cv2

import sys
sys.path.append(str(Path(__file__).parent.parent))

from cli.detect import (
    AmbientPoseConfig, 
    PoseDetector,
    MediaPipeDetector,
    UltralyticsDetector,
    OpenPoseDetector,
    AlphaPoseDetector,
    parse_args
)


class TestAdvancedCLIParsing:
    """Test advanced CLI parameter parsing."""
    
    def test_confidence_threshold_alias(self):
        """Test that --confidence-threshold works as alias for --min-confidence."""
        with patch('sys.argv', ['detect.py', '--video', 'test.mp4', '--confidence-threshold', '0.8']):
            args = parse_args()
            assert args.min_confidence == 0.8
    
    def test_net_resolution_parsing(self):
        """Test network resolution parameter parsing."""
        with patch('sys.argv', ['detect.py', '--video', 'test.mp4', '--net-resolution', '832x512']):
            args = parse_args()
            assert args.net_resolution == '832x512'
    
    def test_model_pose_parsing(self):
        """Test pose model parameter parsing."""
        with patch('sys.argv', ['detect.py', '--video', 'test.mp4', '--model-pose', 'BODY_25']):
            args = parse_args()
            assert args.model_pose == 'BODY_25'
    
    def test_overlay_video_parsing(self):
        """Test overlay video parameter parsing."""
        with patch('sys.argv', ['detect.py', '--video', 'test.mp4', '--overlay-video', 'output.mp4']):
            args = parse_args()
            assert args.overlay_video == 'output.mp4'
    
    def test_toronto_gait_format_parsing(self):
        """Test Toronto gait format flag parsing."""
        with patch('sys.argv', ['detect.py', '--video', 'test.mp4', '--toronto-gait-format']):
            args = parse_args()
            assert args.toronto_gait_format is True
    
    def test_extract_comprehensive_frames_parsing(self):
        """Test comprehensive frame extraction flag parsing."""
        with patch('sys.argv', ['detect.py', '--video', 'test.mp4', '--extract-comprehensive-frames']):
            args = parse_args()
            assert args.extract_comprehensive_frames is True
    
    def test_verbose_parsing(self):
        """Test verbose flag parsing."""
        with patch('sys.argv', ['detect.py', '--video', 'test.mp4', '--verbose']):
            args = parse_args()
            assert args.verbose is True


class TestAmbientPoseConfigAdvanced:
    """Test advanced configuration handling."""
    
    def create_mock_args(self, **kwargs):
        """Create mock arguments with defaults."""
        defaults = {
            'video': 'test.mp4',
            'image_dir': None,
            'output': None,
            'output_dir': 'outputs',
            'backend': 'auto',
            'min_confidence': 0.5,
            'net_resolution': None,
            'model_pose': None,
            'overlay_video': None,
            'toronto_gait_format': False,
            'extract_comprehensive_frames': False,
            'verbose': False,
            'debug': False
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)
    
    def test_backend_defaults(self):
        """Test backend-specific default configurations."""
        args = self.create_mock_args()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_video = Path(temp_dir) / "test.mp4"
            test_video.touch()
            args.video = str(test_video)
            
            config = AmbientPoseConfig(args)
            
            # Test MediaPipe defaults
            mp_config = config.get_backend_config('mediapipe')
            assert mp_config['net_resolution'] == '256x256'
            assert mp_config['model_pose'] == 'POSE_LANDMARKS'
            
            # Test Ultralytics defaults
            ultra_config = config.get_backend_config('ultralytics')
            assert ultra_config['net_resolution'] == '640x640'
            assert ultra_config['model_pose'] == 'yolov8n-pose.pt'
            
            # Test OpenPose defaults
            op_config = config.get_backend_config('openpose')
            assert op_config['net_resolution'] == '656x368'
            assert op_config['model_pose'] == 'BODY_25'
            
            # Test AlphaPose defaults
            alpha_config = config.get_backend_config('alphapose')
            assert alpha_config['net_resolution'] == '256x192'
            assert alpha_config['model_pose'] == 'COCO'
    
    def test_net_resolution_validation(self):
        """Test network resolution format validation."""
        args = self.create_mock_args(net_resolution='656x368')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_video = Path(temp_dir) / "test.mp4"
            test_video.touch()
            args.video = str(test_video)
            
            config = AmbientPoseConfig(args)
            assert config.net_resolution == '656x368'
    
    def test_invalid_net_resolution(self):
        """Test invalid network resolution handling."""
        args = self.create_mock_args(net_resolution='invalid')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_video = Path(temp_dir) / "test.mp4"
            test_video.touch()
            args.video = str(test_video)
            
            with pytest.raises(ValueError, match="Invalid network resolution format"):
                AmbientPoseConfig(args)
    
    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation."""
        args = self.create_mock_args(min_confidence=1.5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_video = Path(temp_dir) / "test.mp4"
            test_video.touch()
            args.video = str(test_video)
            
            with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
                AmbientPoseConfig(args)
    
    def test_overlay_video_path_validation(self):
        """Test overlay video path validation."""
        args = self.create_mock_args(overlay_video='/invalid/path/output.mp4')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_video = Path(temp_dir) / "test.mp4"
            test_video.touch()
            args.video = str(test_video)
            
            # Should create parent directory
            config = AmbientPoseConfig(args)
            assert config.overlay_video_path == '/invalid/path/output.mp4'


class TestBackendSpecificConfigurations:
    """Test backend-specific parameter handling."""
    
    def create_mock_config(self, backend='mediapipe', **kwargs):
        """Create mock configuration for testing."""
        defaults = {
            'min_confidence': 0.5,
            'net_resolution': None,
            'model_pose': None,
            'overlay_video_path': None,
            'toronto_gait_format': False,
            'extract_comprehensive_frames': False,
            'verbose': False,
            'debug': False
        }
        defaults.update(kwargs)
        
        config = Mock()
        config.min_confidence = defaults['min_confidence']
        config.net_resolution = defaults['net_resolution']
        config.model_pose = defaults['model_pose']
        config.overlay_video_path = defaults['overlay_video_path']
        config.toronto_gait_format = defaults['toronto_gait_format']
        config.extract_comprehensive_frames = defaults['extract_comprehensive_frames']
        config.verbose = defaults['verbose']
        config.debug = defaults['debug']
        config.get_backend_config = Mock(return_value=defaults)
        
        return config
    
    @patch('cli.detect.MEDIAPIPE_AVAILABLE', True)
    @patch('cli.detect.mp')
    def test_mediapipe_custom_resolution(self, mock_mp):
        """Test MediaPipe with custom resolution handling."""
        config = self.create_mock_config(net_resolution='512x512')
        
        # Mock MediaPipe components
        mock_mp.solutions.pose = Mock()
        mock_mp.solutions.drawing_utils = Mock()
        mock_pose_class = Mock()
        mock_mp.solutions.pose.Pose = mock_pose_class
        
        detector = MediaPipeDetector(config)
        
        # Should adjust model complexity for larger resolution
        mock_pose_class.assert_called_once()
        call_args = mock_pose_class.call_args[1]
        assert call_args['model_complexity'] == 2  # Full model for 512x512
    
    @patch('cli.detect.ULTRALYTICS_AVAILABLE', True)
    @patch('cli.detect.YOLO')
    def test_ultralytics_custom_model(self, mock_yolo):
        """Test Ultralytics with custom model selection."""
        config = self.create_mock_config(model_pose='yolov8l-pose.pt')
        
        detector = UltralyticsDetector(config)
        
        mock_yolo.assert_called_once_with('yolov8l-pose.pt')
    
    @patch('cli.detect.OPENPOSE_AVAILABLE', True)
    def test_openpose_custom_parameters(self):
        """Test OpenPose with custom parameters."""
        config = self.create_mock_config(
            net_resolution='832x512',
            model_pose='COCO'
        )
        
        with patch.dict('os.environ', {'OPENPOSE_HOME': '/fake/openpose'}):
            with patch('pathlib.Path.exists', return_value=False):
                with pytest.raises(RuntimeError):
                    OpenPoseDetector(config)
    
    @patch('cli.detect.ALPHAPOSE_AVAILABLE', True)
    def test_alphapose_custom_resolution(self):
        """Test AlphaPose with custom resolution."""
        config = self.create_mock_config(net_resolution='384x288')
        
        # Mock the AlphaPose dependencies
        with patch('cli.detect.alphapose_path', Path('/fake/alphapose')):
            with patch('pathlib.Path.exists', return_value=False):
                with pytest.raises(FileNotFoundError):
                    AlphaPoseDetector(config)


class TestOverlayVideoGeneration:
    """Test overlay video generation functionality."""
    
    def test_overlay_video_writer_initialization(self):
        """Test overlay video writer setup."""
        # Create mock PoseDetector
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Mock()
            config.overlay_video_path = 'test_overlay.mp4'
            config.verbose = True
            config.frames_dir = Path(temp_dir) / 'frames'
            config.frames_dir.mkdir(exist_ok=True)
            config.input_path = Path(temp_dir) / 'input.mp4'
            config.input_path.touch()
            config.overlay_dir = Path(temp_dir) / 'overlay'
            config.overlay_dir.mkdir(exist_ok=True)
            
            detector = PoseDetector.__new__(PoseDetector)  # Create without __init__
            detector.config = config
            detector.backend_name = 'test'
            detector.video_metadata = {}
            detector.all_converted_poses = []
            detector.comprehensive_frame_data = []
            
            # Mock OpenCV VideoWriter
            with patch('cv2.VideoWriter') as mock_writer:
                mock_writer_instance = Mock()
                mock_writer.return_value = mock_writer_instance
                
                # Mock video capture
                with patch('cv2.VideoCapture') as mock_cap:
                    mock_cap_instance = Mock()
                    mock_cap.return_value = mock_cap_instance
                    mock_cap_instance.isOpened.return_value = True
                    mock_cap_instance.get.side_effect = lambda prop: {
                        cv2.CAP_PROP_FRAME_COUNT: 10,
                        cv2.CAP_PROP_FPS: 30.0,
                        cv2.CAP_PROP_FRAME_WIDTH: 640,
                        cv2.CAP_PROP_FRAME_HEIGHT: 480
                    }[prop]
                    mock_cap_instance.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8))] * 10 + [(False, None)]
                    
                    # Mock detector methods
                    detector.detector = Mock()
                    detector.detector.detect_poses.return_value = []
                    detector.detector.draw_poses.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
                    detector.person_tracker = Mock()
                    detector.person_tracker.update.return_value = []
                    
                    # Mock file operations
                    with patch('cv2.imwrite'):
                        with patch('pathlib.Path.mkdir'):
                            # Call process_video which should initialize overlay video writer
                            detector.process_video()
                    
                    # Verify VideoWriter was called correctly
                    mock_writer.assert_called_once_with(
                        'test_overlay.mp4',
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        30.0,
                        (640, 480)
                    )
                    
                    # Verify frames were written
                    assert mock_writer_instance.write.call_count == 10
                    mock_writer_instance.release.assert_called_once()


class TestTorontoGaitFormat:
    """Test Toronto gait format output functionality."""
    
    def test_toronto_gait_format_structure(self):
        """Test Toronto gait format JSON structure."""
        # Create mock pose data
        mock_poses = []
        mock_converted_poses = [
            {
                'person_id': 0,
                'frame_number': 0,
                'timestamp': 0.0,
                'confidence': 0.8,
                'joints': [
                    {
                        'name': 'left_hip',
                        'joint_id': 11,
                        'keypoint': {'x': 100.0, 'y': 200.0, 'confidence': 0.9}
                    },
                    {
                        'name': 'right_hip',
                        'joint_id': 12,
                        'keypoint': {'x': 120.0, 'y': 200.0, 'confidence': 0.9}
                    },
                    {
                        'name': 'left_ankle',
                        'joint_id': 15,
                        'keypoint': {'x': 95.0, 'y': 350.0, 'confidence': 0.8}
                    }
                ]
            }
        ]
        
        class SimpleConfig:
            def __init__(self, output_json, input_path):
                self.output_json = output_json
                self.min_confidence = 0.5
                self.net_resolution = '640x480'
                self.model_pose = 'COCO'
                self.input_path = input_path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_output_toronto_gait.json'
            input_path = Path(temp_dir) / 'input.mp4'
            input_path.touch()
            config = SimpleConfig(Path(temp_dir) / 'test_output.json', input_path)
            
            detector = PoseDetector.__new__(PoseDetector)
            detector.config = config
            detector.backend_name = 'test'
            detector.video_metadata = {'width': 640, 'height': 480, 'fps': 30.0}
            detector.all_converted_poses = mock_converted_poses
            
            detector.save_toronto_gait_format(mock_poses)
            
            # Verify file was created
            assert output_path.exists() or (Path(temp_dir) / 'test_output.json').with_name('test_output_toronto_gait.json').exists()
            
            # Verify JSON structure
            with open((Path(temp_dir) / 'test_output.json').with_name('test_output_toronto_gait.json'), 'r') as f:
                data = json.load(f)
            
            assert 'metadata' in data
            assert 'summary' in data
            assert 'gait_analysis' in data
            assert data['metadata']['format'] == 'Toronto Gait Analysis v1.0'
            assert data['summary']['total_people_analyzed'] >= 0


class TestComprehensiveFrameExtraction:
    """Test comprehensive frame extraction functionality."""
    
    def test_frame_statistics_calculation(self):
        """Test frame statistics calculation."""
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        config = Mock()
        config.extract_comprehensive_frames = True
        config.verbose = False
        
        detector = PoseDetector.__new__(PoseDetector)
        detector.config = config
        detector.backend_name = 'test'
        detector.comprehensive_frame_data = []
        
        # Call the extraction method
        detector._extract_comprehensive_frame_data(test_frame, 0, 0.0, [])
        
        # Verify data was captured
        assert len(detector.comprehensive_frame_data) == 1
        frame_data = detector.comprehensive_frame_data[0]
        
        assert 'frame_statistics' in frame_data
        assert 'brightness' in frame_data['frame_statistics']
        assert 'contrast' in frame_data['frame_statistics']
        assert 'sharpness' in frame_data['frame_statistics']
        assert 'motion_blur' in frame_data['frame_statistics']
    
    def test_pose_quality_metrics(self):
        """Test pose quality metrics calculation."""
        # Create mock pose data
        mock_poses = [
            {
                'score': 0.8,
                'bbox': [100, 100, 200, 300, 0.8],
                'keypoints': [
                    [150, 120, 0.9],  # High confidence
                    [160, 130, 0.7],  # Medium confidence
                    [0, 0, 0.0],      # Invalid keypoint
                    [170, 140, 0.8]   # High confidence
                ]
            }
        ]
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        config = Mock()
        config.extract_comprehensive_frames = True
        config.verbose = False
        
        detector = PoseDetector.__new__(PoseDetector)
        detector.config = config
        detector.backend_name = 'test'
        detector.comprehensive_frame_data = []
        
        detector._extract_comprehensive_frame_data(test_frame, 0, 0.0, mock_poses)
        
        frame_data = detector.comprehensive_frame_data[0]
        pose_analysis = frame_data['pose_analysis']
        
        assert pose_analysis['total_poses'] == 1
        assert len(pose_analysis['pose_quality_metrics']) == 1
        
        quality_metrics = pose_analysis['pose_quality_metrics'][0]
        assert 'completeness' in quality_metrics
        assert 'spread_ratio' in quality_metrics
        assert 'average_confidence' in quality_metrics
        assert 'valid_keypoints' in quality_metrics
        
        # Should have 3 valid keypoints out of 4 total
        assert quality_metrics['valid_keypoints'] == 3
        assert quality_metrics['total_keypoints'] == 4
    
    def test_comprehensive_frame_analysis_save(self):
        """Test saving comprehensive frame analysis."""
        config = Mock()
        config.output_json = Path('test_output.json')
        config.min_confidence = 0.5
        config.net_resolution = '640x480'
        config.model_pose = 'COCO'
        
        detector = PoseDetector.__new__(PoseDetector)
        detector.config = config
        detector.backend_name = 'test'
        detector.video_metadata = {'width': 640, 'height': 480, 'fps': 30.0}
        detector.comprehensive_frame_data = [
            {
                'frame_number': 0,
                'timestamp': 0.0,
                'frame_statistics': {'brightness': 128.0, 'contrast': 50.0},
                'pose_analysis': {'total_poses': 1, 'confidence_scores': [0.8]},
                'motion_analysis': None,
                'processing_metadata': {'backend': 'test'}
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_output_comprehensive_frames.json'
            config.output_json = Path(temp_dir) / 'test_output.json'
            
            detector.save_comprehensive_frame_analysis()
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify JSON structure
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert 'metadata' in data
            assert 'summary' in data
            assert 'frame_analysis' in data
            assert data['metadata']['format'] == 'AmbientPose Comprehensive Frame Analysis v1.0'
            assert data['summary']['total_frames_analyzed'] == 1


class TestVerboseLogging:
    """Test verbose logging functionality."""
    
    def test_verbose_configuration_logging(self):
        """Test verbose mode configuration logging."""
        args = argparse.Namespace(
            video='test.mp4',
            image_dir=None,
            output=None,
            output_dir='outputs',
            backend='auto',
            min_confidence=0.5,
            net_resolution='832x512',
            model_pose='BODY_25',
            overlay_video='overlay.mp4',
            toronto_gait_format=True,
            extract_comprehensive_frames=True,
            verbose=True,
            debug=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_video = Path(temp_dir) / "test.mp4"
            test_video.touch()
            args.video = str(test_video)
            
            with patch('cli.detect.logger') as mock_logger:
                config = AmbientPoseConfig(args)
                
                # Verify verbose logging was called
                mock_logger.info.assert_called()
                
                # Check that configuration details were logged
                call_args = [call[0][0] for call in mock_logger.info.call_args_list]
                assert any("AmbientPose Configuration Details" in arg for arg in call_args)


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    def test_all_advanced_features_enabled(self):
        """Test workflow with all advanced features enabled."""
        # This would be a comprehensive integration test
        # that tests the complete pipeline with all features
        args = argparse.Namespace(
            video='test.mp4',
            image_dir=None,
            output=None,
            output_dir='outputs',
            backend='auto',
            min_confidence=0.3,
            net_resolution='656x368',
            model_pose='BODY_25',
            overlay_video='overlay.mp4',
            toronto_gait_format=True,
            extract_comprehensive_frames=True,
            verbose=True,
            debug=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_video = Path(temp_dir) / "test.mp4"
            test_video.touch()
            args.video = str(test_video)
            
            # This test verifies that all parameters are correctly parsed
            # and validated without actually running the detector
            config = AmbientPoseConfig(args)
            
            assert config.net_resolution == '656x368'
            assert config.model_pose == 'BODY_25'
            assert config.overlay_video_path == 'overlay.mp4'
            assert config.toronto_gait_format is True
            assert config.extract_comprehensive_frames is True
            assert config.verbose is True


if __name__ == '__main__':
    pytest.main([__file__]) 