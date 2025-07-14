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
import types

import sys
sys.path.append(str(Path(__file__).parent.parent))

from cli.detect import (
    AmbientPoseConfig, 
    PoseDetector,
    MediaPipeDetector,
    UltralyticsDetector,
    OpenPoseDetector,
    AlphaPoseDetector,
    parse_args,
    convert_pose_to_joints_format
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
            import os
            try:
                assert os.path.normpath(config.overlay_video_path) == os.path.normpath('/invalid/path/output.mp4')
            finally:
                config.close_log_sink()


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
            # Initialize new attributes for original joint stats
            detector.original_low_conf_joints_per_frame = {}
            detector.original_total_joints_per_frame = {}
            detector.original_low_conf_joints_per_person = {}
            detector.original_total_joints_per_person = {}
            detector.original_low_conf_joints_global = 0
            detector.original_total_joints_global = 0
            
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
            config.run_output_dir = Path(temp_dir)
            
            detector = PoseDetector.__new__(PoseDetector)
            detector.config = config
            detector.backend_name = 'test'
            detector.video_metadata = {'width': 640, 'height': 480, 'fps': 30.0}
            detector.all_converted_poses = mock_converted_poses
            
            detector.save_toronto_gait_format(mock_poses)
            
            # Verify file was created in run_output_dir as toronto_gait.json
            expected_path = config.run_output_dir / 'toronto_gait.json'
            assert expected_path.exists(), f"Expected {expected_path} to exist"
            
            # Verify JSON structure
            with open(expected_path, 'r') as f:
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
            config.run_output_dir = Path(temp_dir)
            
            detector.save_comprehensive_frame_analysis()
            
            # Verify file was created in run_output_dir as comprehensive_frames.json
            expected_path = config.run_output_dir / 'comprehensive_frames.json'
            assert expected_path.exists(), f"Expected {expected_path} to exist"
            
            # Verify JSON structure
            with open(expected_path, 'r') as f:
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
            import os
            try:
                assert config.net_resolution == '656x368'
                assert config.model_pose == 'BODY_25'
                assert os.path.normpath(config.overlay_video_path) == os.path.normpath('overlay.mp4')
                assert config.toronto_gait_format is True
                assert config.extract_comprehensive_frames is True
            finally:
                config.close_log_sink()


class TestJointConfidenceFilteringAndInterpolation:
    """Test joint filtering and interpolation logic for min_joint_confidence."""
    def make_config(self, min_joint_confidence=0.3):
        cfg = types.SimpleNamespace()
        cfg.min_joint_confidence = min_joint_confidence
        return cfg

    def test_filter_low_confidence_joints(self):
        """Joints below threshold are included as low-confidence if no interpolation possible."""
        joint_history = {0: {0: {0: (10, 10, 0.2)}}}  # Only one frame, low confidence
        pose = {'person_id': 0, 'keypoints': [[10, 10, 0.2]]}
        result = convert_pose_to_joints_format(pose, 0, 0.0, joint_history, self.make_config(0.3))
        # Should include the joint, marked as low_confidence
        assert len(result['joints']) == 1
        joint = result['joints'][0]
        assert joint['keypoint']['low_confidence'] is True
        assert joint['keypoint']['interpolated'] is False
        assert joint['keypoint']['confidence'] == 0.2

    def test_interpolate_joint_between_frames(self):
        """Joints below threshold are interpolated if adjacent frames have high confidence."""
        joint_history = {
            0: {
                0: {
                    0: (10, 10, 0.8),  # prev
                    1: (0, 0, 0.1),    # current (low)
                    2: (20, 20, 0.7)   # next
                }
            }
        }
        pose = {'person_id': 0, 'keypoints': [[0, 0, 0.1]]}
        result = convert_pose_to_joints_format(pose, 1, 1.0, joint_history, self.make_config(0.3))
        # Should interpolate between (10,10) and (20,20)
        assert len(result['joints']) == 1
        joint = result['joints'][0]
        assert joint['keypoint']['interpolated'] is True
        assert abs(joint['keypoint']['x'] - 15) < 1e-3
        assert abs(joint['keypoint']['y'] - 15) < 1e-3
        assert abs(joint['keypoint']['confidence'] - 0.7) < 1e-3

    def test_no_interpolation_if_no_high_conf_neighbors(self):
        """Joints below threshold are not interpolated if no high-confidence neighbors exist."""
        joint_history = {
            0: {
                0: {
                    0: (10, 10, 0.2),  # prev low
                    1: (0, 0, 0.1),    # current low
                    2: (20, 20, 0.2)   # next low
                }
            }
        }
        pose = {'person_id': 0, 'keypoints': [[0, 0, 0.1]]}
        result = convert_pose_to_joints_format(pose, 1, 1.0, joint_history, self.make_config(0.3))
        # Should be empty since no interpolation possible
        assert result['joints'] == []

    def test_overlay_marks_interpolated_joints(self):
        """Overlay drawing uses magenta for interpolated joints."""
        # Use UltralyticsDetector for test
        import tempfile
        from cli.detect import AmbientPoseConfig
        import argparse
        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_video = Path(temp_dir) / "dummy.mp4"
            dummy_video.touch()
            args = argparse.Namespace(
                video=str(dummy_video), image_dir=None, output=None, output_dir=temp_dir,
                backend='ultralytics', min_confidence=0.5, net_resolution=None, model_pose=None,
                overlay_video=None, toronto_gait_format=False, extract_comprehensive_frames=False,
                verbose=False, debug=False, min_joint_confidence=0.3
            )
            config = AmbientPoseConfig(args)
            try:
                detector = UltralyticsDetector(config)
                # One pose, one joint, interpolated
                pose = {'bbox': [0,0,100,100,1.0], 'score': 1.0, 'keypoints': [[15, 15, 0.7]], 'backend': 'ultralytics'}
                converted_pose = {'joints': [{'keypoint': {'x': 15, 'y': 15, 'confidence': 0.7, 'interpolated': True}}]}
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                out = detector.draw_poses(img.copy(), [pose], [converted_pose])
                # Check that magenta pixel is present (255,0,255)
                assert ((out[:,:,0] == 255) & (out[:,:,1] == 0) & (out[:,:,2] == 255)).any()
            finally:
                config.close_log_sink()

    def test_stats_count_low_confidence_and_interpolated(self):
        """PoseDetector counts low-confidence/interpolated joints correctly."""
        import tempfile
        from cli.detect import AmbientPoseConfig
        import argparse
        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_video = Path(temp_dir) / "dummy.mp4"
            dummy_video.touch()
            args = argparse.Namespace(
                video=str(dummy_video), image_dir=None, output=None, output_dir=temp_dir,
                backend='ultralytics', min_confidence=0.5, net_resolution=None, model_pose=None,
                overlay_video=None, toronto_gait_format=False, extract_comprehensive_frames=False,
                verbose=False, debug=False, min_joint_confidence=0.3
            )
            config = AmbientPoseConfig(args)
            try:
                pd = PoseDetector(config)
                pd.all_converted_poses = [
                    {'joints': [
                        {'keypoint': {'x': 1, 'y': 1, 'confidence': 0.2, 'interpolated': False}},
                        {'keypoint': {'x': 2, 'y': 2, 'confidence': 0.8, 'interpolated': False}},
                        {'keypoint': {'x': 3, 'y': 3, 'confidence': 0.7, 'interpolated': True}},
                    ]}
                ]
                pd.low_conf_joints_count = 0
                pd.total_joints_count = 0
                for pose in pd.all_converted_poses:
                    for joint in pose['joints']:
                        pd.total_joints_count += 1
                        if joint['keypoint'].get('interpolated', False):
                            pd.low_conf_joints_count += 1
                        elif joint['keypoint']['confidence'] < pd.config.min_joint_confidence:
                            pd.low_conf_joints_count += 1
                assert pd.total_joints_count == 3
                assert pd.low_conf_joints_count == 2
            finally:
                config.close_log_sink()

    def test_zero_joints_no_division_by_zero(self):
        """If there are zero joints, percentage is 0 and no ZeroDivisionError occurs."""
        import tempfile
        from cli.detect import AmbientPoseConfig
        import argparse
        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_video = Path(temp_dir) / "dummy.mp4"
            dummy_video.touch()
            args = argparse.Namespace(
                video=str(dummy_video), image_dir=None, output=None, output_dir=temp_dir,
                backend='ultralytics', min_confidence=0.5, net_resolution=None, model_pose=None,
                overlay_video=None, toronto_gait_format=False, extract_comprehensive_frames=False,
                verbose=False, debug=False, min_joint_confidence=0.3
            )
            config = AmbientPoseConfig(args)
            try:
                pd = PoseDetector(config)
                pd.all_converted_poses = []
                pd.low_conf_joints_count = 0
                pd.total_joints_count = 0
                # Should not raise
                pd.save_results([])
            finally:
                config.close_log_sink()

    def test_summary_json_no_interpolated_joints(self):
        """If there are no interpolated joints, the summary JSON is correct and well-structured."""
        import tempfile
        from cli.detect import AmbientPoseConfig
        import argparse
        import json as _json
        from loguru import logger
        from io import StringIO
        import re
        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_video = Path(temp_dir) / "dummy.mp4"
            dummy_video.touch()
            args = argparse.Namespace(
                video=str(dummy_video), image_dir=None, output=None, output_dir=temp_dir,
                backend='ultralytics', min_confidence=0.5, net_resolution=None, model_pose=None,
                overlay_video=None, toronto_gait_format=False, extract_comprehensive_frames=False,
                verbose=False, debug=False, min_joint_confidence=0.3
            )
            config = AmbientPoseConfig(args)
            try:
                pd = PoseDetector(config)
                pd.all_converted_poses = []
                pd.low_conf_joints_count = 0
                pd.total_joints_count = 0
                # Use a Loguru sink to capture all log output, including SUCCESS
                sink = StringIO()
                sink_id = logger.add(sink, level="SUCCESS")
                pd.save_results([])
                logger.remove(sink_id)
                sink.seek(0)
                log_output = sink.read()
                # Find the summary JSON in the log (handle multi-line pretty-printed JSON)
                lines = log_output.splitlines()
                for i, line in enumerate(lines):
                    if "Summary:" in line:
                        # Find the first '{' in this or subsequent lines
                        json_lines = []
                        found_brace = False
                        for l in lines[i:]:
                            if not found_brace:
                                if '{' in l:
                                    found_brace = True
                                    json_lines.append(l[l.index('{'):])
                            else:
                                json_lines.append(l)
                            if '}' in l:
                                # Assume last '}' closes the JSON
                                break
                        if json_lines:
                            json_str = '\n'.join(json_lines)
                            summary_json = _json.loads(json_str)
                            break
                else:
                    assert False, "Summary JSON not found in log"
                # Check only the new summary keys
                assert set(summary_json.keys()) == {"original_total_joints", "original_low_confidence_joints", "original_low_confidence_joints_per_frame", "note"}
                # Check that all values are zero/empty as expected
                assert summary_json["original_total_joints"] == 0
                assert summary_json["original_low_confidence_joints"] == 0
                assert summary_json["original_low_confidence_joints_per_frame"] == {} or summary_json["original_low_confidence_joints_per_frame"] == {0: 0}
                assert "original_low_confidence_joints_per_frame" in summary_json["note"]
            finally:
                config.close_log_sink()

    def test_summary_json_joint_counting(self):
        """Test that original joint counts are correct in the summary JSON."""
        import tempfile
        from cli.detect import AmbientPoseConfig
        import argparse
        import json as _json
        from loguru import logger
        from io import StringIO
        import re
        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_video = Path(temp_dir) / "dummy.mp4"
            dummy_video.touch()
            args = argparse.Namespace(
                video=str(dummy_video), image_dir=None, output=None, output_dir=temp_dir,
                backend='ultralytics', min_confidence=0.5, net_resolution=None, model_pose=None,
                overlay_video=None, toronto_gait_format=False, extract_comprehensive_frames=False,
                verbose=False, debug=False, min_joint_confidence=0.3
            )
            config = AmbientPoseConfig(args)
            try:
                pd = PoseDetector(config)
                # Simulate original joint stats
                pd.original_total_joints_global = 5
                pd.original_low_conf_joints_global = 2
                pd.original_low_conf_joints_per_frame = {0: 1, 1: 1}
                pd.original_total_joints_per_frame = {0: 3, 1: 2}
                # Simulate output joints (all high-confidence after interpolation)
                pd.all_converted_poses = [
                    {'person_id': 0, 'frame_number': 0, 'joints': [
                        {'keypoint': {'x': 1, 'y': 1, 'confidence': 0.2, 'interpolated': False}},
                        {'keypoint': {'x': 2, 'y': 2, 'confidence': 0.8, 'interpolated': False}},
                        {'keypoint': {'x': 3, 'y': 3, 'confidence': 0.7, 'interpolated': True}},
                    ]},
                    {'person_id': 1, 'frame_number': 1, 'joints': [
                        {'keypoint': {'x': 4, 'y': 4, 'confidence': 0.9, 'interpolated': False}},
                        {'keypoint': {'x': 5, 'y': 5, 'confidence': 0.1, 'interpolated': True}},
                    ]}
                ]
                sink = StringIO()
                sink_id = logger.add(sink, level="SUCCESS")
                pd.save_results([])
                logger.remove(sink_id)
                sink.seek(0)
                log_output = sink.read()
                # Find the summary JSON in the log (handle multi-line pretty-printed JSON)
                lines = log_output.splitlines()
                for i, line in enumerate(lines):
                    if "Summary:" in line:
                        json_lines = []
                        found_brace = False
                        for l in lines[i:]:
                            if not found_brace:
                                if '{' in l:
                                    found_brace = True
                                    json_lines.append(l[l.index('{'):])
                            else:
                                json_lines.append(l)
                            if '}' in l:
                                break
                        if json_lines:
                            json_str = '\n'.join(json_lines)
                            summary_json = _json.loads(json_str)
                            break
                else:
                    assert False, "Summary JSON not found in log"
                # Check only the new summary keys
                assert set(summary_json.keys()) == {"original_total_joints", "original_low_confidence_joints", "original_low_confidence_joints_per_frame", "note"}
                # Check global
                assert summary_json["original_total_joints"] == 5
                assert summary_json["original_low_confidence_joints"] == 2
                # Check per-frame
                assert summary_json["original_low_confidence_joints_per_frame"] == {"0": 1, "1": 1} or summary_json["original_low_confidence_joints_per_frame"] == {0: 1, 1: 1}
                # The sum across frames equals the global count
                assert sum(summary_json["original_low_confidence_joints_per_frame"].values()) == summary_json["original_low_confidence_joints"]
                # Note is present
                assert "original_low_confidence_joints_per_frame" in summary_json["note"]
            finally:
                config.close_log_sink()

    def test_original_low_confidence_joint_counting(self):
        """Test that original low-confidence joint counts are correct in summary (global, per-frame, per-person)."""
        import tempfile
        from cli.detect import AmbientPoseConfig
        import argparse
        import json as _json
        from loguru import logger
        from io import StringIO
        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_video = Path(temp_dir) / "dummy.mp4"
            dummy_video.touch()
            args = argparse.Namespace(
                video=str(dummy_video), image_dir=None, output=None, output_dir=temp_dir,
                backend='ultralytics', min_confidence=0.5, net_resolution=None, model_pose=None,
                overlay_video=None, toronto_gait_format=False, extract_comprehensive_frames=False,
                verbose=False, debug=False, min_joint_confidence=0.3
            )
            config = AmbientPoseConfig(args)
            try:
                pd = PoseDetector(config)
                # Simulate original joint stats
                pd.original_total_joints_global = 6
                pd.original_low_conf_joints_global = 2
                pd.original_total_joints_per_frame = {0: 3, 1: 3}
                pd.original_low_conf_joints_per_frame = {0: 1, 1: 1}
                pd.original_total_joints_per_person = {0: 4, 1: 2}
                pd.original_low_conf_joints_per_person = {0: 1, 1: 1}
                # Simulate output joints (all high-confidence after interpolation)
                pd.all_converted_poses = [
                    {'person_id': 0, 'frame_number': 0, 'joints': [
                        {'keypoint': {'x': 1, 'y': 1, 'confidence': 0.2, 'interpolated': True, 'low_confidence': False}},
                        {'keypoint': {'x': 2, 'y': 2, 'confidence': 0.8, 'interpolated': False, 'low_confidence': False}},
                    ]},
                    {'person_id': 0, 'frame_number': 1, 'joints': [
                        {'keypoint': {'x': 3, 'y': 3, 'confidence': 0.7, 'interpolated': False, 'low_confidence': False}},
                        {'keypoint': {'x': 4, 'y': 4, 'confidence': 0.9, 'interpolated': False, 'low_confidence': False}},
                    ]},
                    {'person_id': 1, 'frame_number': 1, 'joints': [
                        {'keypoint': {'x': 5, 'y': 5, 'confidence': 0.1, 'interpolated': True, 'low_confidence': False}},
                        {'keypoint': {'x': 6, 'y': 6, 'confidence': 0.5, 'interpolated': False, 'low_confidence': False}},
                    ]}
                ]
                sink = StringIO()
                sink_id = logger.add(sink, level="SUCCESS")
                pd.save_results([])
                logger.remove(sink_id)
                sink.seek(0)
                log_output = sink.read()
                # Find the summary JSON in the log
                lines = log_output.splitlines()
                for i, line in enumerate(lines):
                    if "Summary:" in line:
                        json_lines = []
                        found_brace = False
                        for l in lines[i:]:
                            if not found_brace:
                                if '{' in l:
                                    found_brace = True
                                    json_lines.append(l[l.index('{'):])
                            else:
                                json_lines.append(l)
                            if '}' in l:
                                break
                        if json_lines:
                            json_str = '\n'.join(json_lines)
                            summary_json = _json.loads(json_str)
                            break
                else:
                    assert False, "Summary JSON not found in log"
                # Check global
                assert summary_json["original_total_joints"] == 6
                assert summary_json["original_low_confidence_joints"] == 2
                # Check per-frame
                assert summary_json["original_low_confidence_joints_per_frame"] == {"0": 1, "1": 1} or summary_json["original_low_confidence_joints_per_frame"] == {0: 1, 1: 1}
                # The sum across frames equals the global count
                assert sum(summary_json["original_low_confidence_joints_per_frame"].values()) == summary_json["original_low_confidence_joints"]
                # Note is present
                assert "original_low_confidence_joints_per_frame" in summary_json["note"]
            finally:
                config.close_log_sink()

    def test_original_low_confidence_joint_counting_from_detection(self):
        """Test that original low-confidence joint counts are correct when counted immediately after detection."""
        import tempfile
        from cli.detect import AmbientPoseConfig, PoseDetector
        import argparse
        import json as _json
        from loguru import logger
        from io import StringIO
        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_video = Path(temp_dir) / "dummy.mp4"
            dummy_video.touch()
            args = argparse.Namespace(
                video=str(dummy_video), image_dir=None, output=None, output_dir=temp_dir,
                backend='ultralytics', min_confidence=0.5, net_resolution=None, model_pose=None,
                overlay_video=None, toronto_gait_format=False, extract_comprehensive_frames=False,
                verbose=False, debug=False, min_joint_confidence=0.3
            )
            config = AmbientPoseConfig(args)
            try:
                pd = PoseDetector(config)
                # Simulate detection output (before tracking)
                # Frame 0: person 0 (2 joints: 1 high, 1 low), person 1 (1 joint: low)
                # Frame 1: person 0 (1 joint: high), person 1 (2 joints: both low)
                pd.original_low_conf_joints_global = 0
                pd.original_total_joints_global = 0
                pd.original_low_conf_joints_per_frame = {}
                pd.original_total_joints_per_frame = {}
                pd.original_low_conf_joints_per_person = {}
                pd.original_total_joints_per_person = {}
                # Simulate what the new code would do
                # Frame 0
                pd.original_total_joints_per_frame[0] = 3
                pd.original_low_conf_joints_per_frame[0] = 2
                pd.original_total_joints_per_person[0] = 2
                pd.original_low_conf_joints_per_person[0] = 1
                pd.original_total_joints_per_person[1] = 1
                pd.original_low_conf_joints_per_person[1] = 1
                # Frame 1
                pd.original_total_joints_per_frame[1] = 3
                pd.original_low_conf_joints_per_frame[1] = 2
                pd.original_total_joints_per_person[0] += 1
                # person 0, frame 1: 1 high
                # person 1, frame 1: 2 low
                pd.original_total_joints_per_person[1] += 2
                pd.original_low_conf_joints_per_person[1] += 2
                pd.original_total_joints_global = 6
                pd.original_low_conf_joints_global = 4
                # Simulate output joints (all high-confidence after interpolation)
                pd.all_converted_poses = [
                    {'person_id': 0, 'frame_number': 0, 'joints': [
                        {'keypoint': {'x': 1, 'y': 1, 'confidence': 0.8, 'interpolated': False, 'low_confidence': False}},
                        {'keypoint': {'x': 2, 'y': 2, 'confidence': 0.2, 'interpolated': True, 'low_confidence': False}},
                    ]},
                    {'person_id': 1, 'frame_number': 0, 'joints': [
                        {'keypoint': {'x': 3, 'y': 3, 'confidence': 0.1, 'interpolated': True, 'low_confidence': False}},
                    ]},
                    {'person_id': 0, 'frame_number': 1, 'joints': [
                        {'keypoint': {'x': 4, 'y': 4, 'confidence': 0.9, 'interpolated': False, 'low_confidence': False}},
                    ]},
                    {'person_id': 1, 'frame_number': 1, 'joints': [
                        {'keypoint': {'x': 5, 'y': 5, 'confidence': 0.2, 'interpolated': True, 'low_confidence': False}},
                        {'keypoint': {'x': 6, 'y': 6, 'confidence': 0.1, 'interpolated': True, 'low_confidence': False}},
                    ]}
                ]
                sink = StringIO()
                sink_id = logger.add(sink, level="SUCCESS")
                pd.save_results([])
                logger.remove(sink_id)
                sink.seek(0)
                log_output = sink.read()
                # Find the summary JSON in the log
                lines = log_output.splitlines()
                for i, line in enumerate(lines):
                    if "Summary:" in line:
                        json_lines = []
                        found_brace = False
                        for l in lines[i:]:
                            if not found_brace:
                                if '{' in l:
                                    found_brace = True
                                    json_lines.append(l[l.index('{'):])
                            else:
                                json_lines.append(l)
                            if '}' in l:
                                break
                        if json_lines:
                            json_str = '\n'.join(json_lines)
                            summary_json = _json.loads(json_str)
                            break
                else:
                    assert False, "Summary JSON not found in log"
                # Check only the new summary keys
                assert set(summary_json.keys()) == {"original_total_joints", "original_low_confidence_joints", "original_low_confidence_joints_per_frame", "note"}
                # Check global
                assert summary_json["original_total_joints"] == 6
                assert summary_json["original_low_confidence_joints"] == 4
                # Check per-frame
                assert summary_json["original_low_confidence_joints_per_frame"] == {"0": 2, "1": 2} or summary_json["original_low_confidence_joints_per_frame"] == {0: 2, 1: 2}
                # The sum across frames equals the global count
                assert sum(summary_json["original_low_confidence_joints_per_frame"].values()) == summary_json["original_low_confidence_joints"]
                # Note is present
                assert "original_low_confidence_joints_per_frame" in summary_json["note"]
            finally:
                config.close_log_sink()


class TestOutputNamingScheme:
    def test_logical_output_json_naming(self):
        import re
        import argparse
        from cli.detect import AmbientPoseConfig
        with tempfile.TemporaryDirectory() as temp_dir:
            test_video = Path(temp_dir) / "myvideo.mp4"
            test_video.touch()
            args = argparse.Namespace(
                video=str(test_video),
                image_dir=None,
                output=None,
                output_dir=temp_dir,
                backend="ultralytics",
                min_confidence=0.5,
                net_resolution=None,
                model_pose=None,
                overlay_video=None,
                toronto_gait_format=False,
                extract_comprehensive_frames=False,
                verbose=False,
                debug=False,
                min_joint_confidence=0.3
            )
            config = AmbientPoseConfig(args)
            try:
                # Should be pose_points.json in the run directory
                assert config.output_json.name == "pose_points.json", f"Output JSON name {config.output_json.name} does not match expected 'pose_points.json'"
                # Parent should be a subdirectory of temp_dir matching the run dir pattern
                import re
                parent = config.output_json.parent
                assert parent.parent == Path(temp_dir), f"Parent dir {parent.parent} is not temp_dir {temp_dir}"
                pattern = re.compile(r"pose_myvideo_ultralytics_\d{8}_\d{6}")
                assert pattern.match(parent.name), f"Run dir {parent.name} does not match expected pattern"
            finally:
                config.close_log_sink()


if __name__ == '__main__':
    pytest.main([__file__]) 