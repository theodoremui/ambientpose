import sys
import os
import pytest
import tempfile
from frontend.components.viewer import OverlayVideoViewer
import builtins
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from frontend.components.uploader import FileUploader

class TestFileUploader:
    def test_initial_state(self):
        uploader = FileUploader()
        assert uploader.backend == "mediapipe"
        assert uploader.min_confidence == 0.5
        assert uploader.min_joint_confidence == 0.5
        assert uploader.uploaded_file is None

    def test_build_cli_command_minimal_video(self):
        uploader = FileUploader()
        uploader.input_type = "video"
        cmd = uploader._build_cli_command("/tmp/video.mp4")
        assert "--video" in cmd
        assert "/tmp/video.mp4" in cmd
        assert "--backend" in cmd
        assert uploader.backend in cmd
        assert "--min-confidence" in cmd
        assert str(uploader.min_confidence) in cmd
        assert "--min-joint-confidence" in cmd
        assert str(uploader.min_joint_confidence) in cmd

    def test_build_cli_command_all_options(self):
        uploader = FileUploader()
        FileUploader.input_type = "image-dir"
        FileUploader.net_resolution = "832x512"
        FileUploader.model_pose = "COCO"
        FileUploader.overlay_video = "outputs/overlay.mp4"
        FileUploader.toronto_gait_format = True
        FileUploader.extract_comprehensive_frames = True
        FileUploader.debug = True
        FileUploader.verbose = True
        FileUploader.output_dir = "outputs/test"
        FileUploader.output = "outputs/test/result.json"
        cmd = FileUploader._build_cli_command("/tmp/images.zip")
        assert "--image-dir" in cmd
        assert "/tmp/images.zip" in cmd
        assert "--net-resolution" in cmd
        assert "832x512" in cmd
        assert "--model-pose" in cmd
        assert "COCO" in cmd
        assert "--overlay-video" in cmd
        overlay_video_arg = cmd[cmd.index("--overlay-video") + 1]
        assert overlay_video_arg.endswith("_overlay.mp4")
        assert "--toronto-gait-format" in cmd
        assert "--extract-comprehensive-frames" in cmd
        assert "--debug" in cmd
        assert "--verbose" in cmd
        assert "--output-dir" in cmd
        output_dir_arg = cmd[cmd.index("--output-dir") + 1]
        assert output_dir_arg.startswith("outputs/test")
        assert "--output" in cmd
        output_json_arg = cmd[cmd.index("--output") + 1]
        assert output_json_arg.endswith(".json")
        prefix = os.path.normpath("outputs/test/pose_images_mediapipe_")
        assert os.path.normpath(output_json_arg).startswith(prefix) 