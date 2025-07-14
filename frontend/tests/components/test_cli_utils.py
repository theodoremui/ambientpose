import sys
import os
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from frontend.components.cli_utils import build_detect_cli_command

def test_build_cli_command_minimal_video():
    cmd = build_detect_cli_command(
        input_type="video",
        input_path="/tmp/video.mp4",
        backend="ultralytics",
        min_confidence=0.7,
        min_joint_confidence=0.2
    )
    assert "--video" in cmd
    assert "/tmp/video.mp4" in cmd
    assert "--backend" in cmd
    assert "ultralytics" in cmd
    assert "--min-confidence" in cmd
    assert "0.7" in cmd
    assert "--min-joint-confidence" in cmd
    assert "0.2" in cmd

def test_build_cli_command_all_options():
    cmd = build_detect_cli_command(
        input_type="image-dir",
        input_path="/tmp/images.zip",
        backend="openpose",
        min_confidence=0.6,
        min_joint_confidence=0.1,
        net_resolution="832x512",
        model_pose="BODY_25",
        overlay_video="outputs/overlay.mp4",
        toronto_gait_format=True,
        extract_comprehensive_frames=True,
        debug=True,
        verbose=True,
        output_dir="outputs/test",
        output="outputs/test/result.json"
    )
    assert "--image-dir" in cmd
    assert "/tmp/images.zip" in cmd
    assert "--net-resolution" in cmd
    assert "832x512" in cmd
    assert "--model-pose" in cmd
    assert "BODY_25" in cmd
    assert "--overlay-video" in cmd
    assert "outputs/overlay.mp4" in cmd
    assert "--toronto-gait-format" in cmd
    assert "--extract-comprehensive-frames" in cmd
    assert "--debug" in cmd
    assert "--verbose" in cmd
    assert "--output-dir" in cmd
    assert "outputs/test" in cmd
    assert "--output" in cmd
    assert "outputs/test/result.json" in cmd 