from typing import List

def build_detect_cli_command(
    input_type: str,
    input_path: str,
    backend: str = "mediapipe",
    min_confidence: float = 0.5,
    min_joint_confidence: float = 0.5,
    net_resolution: str = None,
    model_pose: str = None,
    overlay_video: str = None,
    toronto_gait_format: bool = False,
    extract_comprehensive_frames: bool = False,
    debug: bool = False,
    verbose: bool = False,
    output_dir: str = "outputs",
    output: str = None,
    python_executable: str = None
) -> List[str]:
    import sys
    cmd = [python_executable or sys.executable, "cli/detect.py"]
    if input_type == "video":
        cmd += ["--video", input_path]
    else:
        cmd += ["--image-dir", input_path]
    if output:
        cmd += ["--output", output]
    if output_dir:
        cmd += ["--output-dir", output_dir]
    if backend:
        cmd += ["--backend", backend]
    if min_confidence is not None:
        cmd += ["--min-confidence", str(min_confidence)]
    if net_resolution:
        cmd += ["--net-resolution", net_resolution]
    if model_pose:
        cmd += ["--model-pose", model_pose]
    if overlay_video:
        cmd += ["--overlay-video", overlay_video]
    if toronto_gait_format:
        cmd += ["--toronto-gait-format"]
    if extract_comprehensive_frames:
        cmd += ["--extract-comprehensive-frames"]
    if debug:
        cmd += ["--debug"]
    if verbose:
        cmd += ["--verbose"]
    if min_joint_confidence is not None:
        cmd += ["--min-joint-confidence", str(min_joint_confidence)]
    return cmd 