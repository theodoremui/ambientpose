import streamlit as st
from typing import Optional, List
import tempfile
import os
import sys
import subprocess
from .cli_utils import build_detect_cli_command

class FileUploader:
    # Backend-specific recommended defaults
    BACKEND_DEFAULTS = {
        "mediapipe": {"net_resolution": "256x256", "model_pose": "POSE_LANDMARKS"},
        "ultralytics": {"net_resolution": "640x640", "model_pose": "yolov8n-pose.pt"},
        "openpose": {"net_resolution": "656x368", "model_pose": "BODY_25"},
        "alphapose": {"net_resolution": "256x192", "model_pose": "COCO"},
    }
    # Store parameters as class variables for cross-method access
    uploaded_file = None
    backend = "ultralytics"
    min_confidence = 0.5
    min_joint_confidence = 0.5
    net_resolution = None
    model_pose = None
    overlay_video = None
    toronto_gait_format = False
    extract_comprehensive_frames = False
    debug = False
    verbose = False
    output_dir = "outputs"
    output = None
    input_type = "video"  # or "image-dir"
    input_path = None

    @classmethod
    def render_sidebar(cls):
        # Use Streamlit session state for sticky/auto-populated fields
        if 'backend' not in st.session_state:
            st.session_state['backend'] = 'ultralytics'
        if 'net_resolution' not in st.session_state:
            st.session_state['net_resolution'] = ""
        if 'model_pose' not in st.session_state:
            st.session_state['model_pose'] = ""
        if 'output_dir' not in st.session_state:
            st.session_state['output_dir'] = "outputs"

        # Backend selection
        backend = st.selectbox("Select Backend", list(cls.BACKEND_DEFAULTS.keys()), key="backend")
        cls.backend = backend
        recommended = cls.BACKEND_DEFAULTS[backend]

        # If backend changed, update recommended values if user hasn't typed anything
        if st.session_state.get('last_backend', None) != backend:
            if not st.session_state['net_resolution']:
                st.session_state['net_resolution'] = recommended['net_resolution']
            if not st.session_state['model_pose']:
                st.session_state['model_pose'] = recommended['model_pose']
            st.session_state['last_backend'] = backend

        # Detection/confidence sliders
        cls.min_confidence = st.slider("Minimum Detection Confidence", 0.0, 1.0, 0.5, 0.01, key="min_confidence")
        cls.min_joint_confidence = st.slider("Minimum Joint Confidence", 0.0, 1.0, 0.3, 0.01, key="min_joint_confidence")

        # Net resolution and model pose with recommended values
        cls.net_resolution = st.text_input(
            f"Network Resolution (recommended: {recommended['net_resolution']})",
            value=st.session_state['net_resolution'],
            key="net_resolution",
            help=f"Recommended for {backend}: {recommended['net_resolution']}"
        ) or None
        cls.model_pose = st.text_input(
            f"Model Pose (recommended: {recommended['model_pose']})",
            value=st.session_state['model_pose'],
            key="model_pose",
            help=f"Recommended for {backend}: {recommended['model_pose']}"
        ) or None

        cls.overlay_video = st.text_input("Overlay Video Output Path", value="", key="overlay_video") or None
        cls.toronto_gait_format = st.checkbox("Output Toronto Gait Format", value=False, key="toronto_gait_format")
        cls.extract_comprehensive_frames = st.checkbox("Extract Comprehensive Frame Metadata", value=False, key="extract_comprehensive_frames")
        cls.debug = st.checkbox("Enable Debug Mode", value=False, key="debug")
        cls.verbose = st.checkbox("Enable Verbose Logging", value=False, key="verbose")
        # Always default output_dir to 'outputs' unless user overrides
        cls.output_dir = st.text_input("Output Directory", value=st.session_state['output_dir'], key="output_dir")
        # DO NOT set st.session_state['output_dir'] after widget creation
        cls.output = st.text_input("Output JSON Path (optional)", value="", key="output") or None

    @classmethod
    def render_main(cls):
        st.subheader("Upload Video or Image Directory")
        input_type = st.radio("Input Type", ["Video", "Image Directory"], key="input_type")
        cls.input_type = "video" if input_type == "Video" else "image-dir"
        if cls.input_type == "video":
            cls.uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"], key="uploaded_file")
        else:
            cls.uploaded_file = st.file_uploader("Choose a zip of images", type=["zip"], key="uploaded_file")
        st.markdown("---")
        if st.button("Run Detection", disabled=cls.uploaded_file is None, key="run_detection"):
            cls._handle_run_detection()

    @classmethod
    def _build_cli_command(cls, input_path: str) -> List[str]:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        backend = cls.backend or 'auto'
        # Create a unique run folder for this run
        output_dir = cls.output_dir or 'outputs'
        run_folder = os.path.join(output_dir, f"pose_{input_name}_{backend}_{timestamp}")
        os.makedirs(run_folder, exist_ok=True)
        # Output JSON path
        output_json = os.path.join(run_folder, f"pose_{input_name}_{backend}_{timestamp}.json")
        # Only use overlay_video if user specified it in the sidebar
        overlay_video = cls.overlay_video if cls.overlay_video else None
        # Build the command directly to ensure correct input flag
        import sys
        cmd = [sys.executable, "cli/detect.py"]
        if cls.input_type == "image-dir":
            cmd += ["--image-dir", input_path]
        else:
            cmd += ["--video", input_path]
        if output_json:
            cmd += ["--output", output_json]
        if run_folder:
            cmd += ["--output-dir", run_folder]
        if backend:
            cmd += ["--backend", backend]
        if cls.min_confidence is not None:
            cmd += ["--min-confidence", str(cls.min_confidence)]
        if cls.net_resolution:
            cmd += ["--net-resolution", cls.net_resolution]
        if cls.model_pose:
            cmd += ["--model-pose", cls.model_pose]
        if overlay_video:
            cmd += ["--overlay-video", overlay_video]
        if cls.toronto_gait_format:
            cmd += ["--toronto-gait-format"]
        if cls.extract_comprehensive_frames:
            cmd += ["--extract-comprehensive-frames"]
        if cls.debug:
            cmd += ["--debug"]
        if cls.verbose:
            cmd += ["--verbose"]
        if cls.min_joint_confidence is not None:
            cmd += ["--min-joint-confidence", str(cls.min_joint_confidence)]
        return cmd

    @classmethod
    def _handle_run_detection(cls):
        st.info("Preparing to run detection...")
        if cls.uploaded_file is None:
            st.error("No file uploaded.")
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded file to disk
            input_path = os.path.join(tmpdir, cls.uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(cls.uploaded_file.getbuffer())
            st.success(f"File saved to {input_path}")
            # Build CLI command
            cmd = cls._build_cli_command(input_path)
            st.code(" ".join(cmd), language="bash")
            # Run CLI synchronously and stream output
            cls._run_cli_sync(cmd)

    @classmethod
    def _run_cli_sync(cls, cmd: List[str]):
        st.info("Running detection... (this may take a while)")
        output_area = st.empty()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        lines = []
        for line in process.stdout:
            lines.append(line.rstrip())
            output_area.text("\n".join(lines[-20:]))  # Show last 20 lines
        process.wait()
        if process.returncode == 0:
            st.success("Detection completed successfully.")
        else:
            st.error(f"Detection failed with exit code {process.returncode}.") 