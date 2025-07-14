import streamlit as st
import os
import pandas as pd
import json

class OutputViewer:
    """
    Displays processed images, run JSON output, and other results for a selected detection run.
    - Always shows the main run JSON output (summary and frame-level confidence info).
    - Allows frame-by-frame image navigation with slider and arrow buttons.
    - Displays CSV results if present.
    """
    def __init__(self):
        self.output_dir = "outputs"

    def render(self):
        """
        Render the output viewer UI for a selected detection run.
        """
        st.subheader("View Detection Outputs")
        # List available run folders (pose_* subfolders)
        if not os.path.exists(self.output_dir):
            st.info("No outputs found. Run detection first.")
            return
        folders = [f for f in os.listdir(self.output_dir)
                   if os.path.isdir(os.path.join(self.output_dir, f)) and f.startswith("pose_")]
        if not folders:
            st.info("No output folders found.")
            return
        folder = st.selectbox("Select Output Folder", folders, key="output_folder_select")
        folder_path = os.path.join(self.output_dir, folder)
        files = os.listdir(folder_path)
        # Get frames and overlays image lists
        frames_dir = os.path.join(folder_path, "frames")
        overlays_dir = os.path.join(folder_path, "overlays")
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith((".jpg", ".png"))]) if os.path.exists(frames_dir) else []
        overlays = sorted([f for f in os.listdir(overlays_dir) if f.endswith((".jpg", ".png"))]) if os.path.exists(overlays_dir) else []
        num_images = max(len(frames), len(overlays))
        jsons = [f for f in files if f.endswith('.json')]
        csvs = [f for f in files if f.endswith('.csv')]

        # Always show the main run JSON output (if present)
        main_json = None
        if jsons:
            # Prefer a JSON file matching the folder name or the most recent
            for js in jsons:
                if folder in js:
                    main_json = js
                    break
            if not main_json:
                main_json = sorted(jsons)[-1]
            pose_points_json_path = os.path.join(folder_path, main_json)
        else:
            st.info("No JSON results found in this output folder.")

        # Unified slider/arrows for frame index
        if num_images > 0:
            st.markdown("### Scan Frames and Overlays")
            slider_key = "frame_overlay_slider_idx"
            play_key = "play_frame_overlay"
            interval_key = "interval_frame_overlay"
            if slider_key not in st.session_state:
                st.session_state[slider_key] = 0
            if play_key not in st.session_state:
                st.session_state[play_key] = False
            if interval_key not in st.session_state:
                st.session_state[interval_key] = 1.0
            max_idx = num_images - 1
            col1, col2, col3, col4 = st.columns([1,6,1,2])
            with col1:
                if st.button("⬅️", key="img_left_frame_overlay", disabled=st.session_state[slider_key] <= 0):
                    st.session_state[slider_key] = max(0, st.session_state[slider_key] - 1)
            with col2:
                idx = st.slider("Frame Index", 0, max_idx, st.session_state[slider_key], key="image_slider_frame_overlay")
                st.session_state[slider_key] = idx
            with col3:
                if st.button("➡️", key="img_right_frame_overlay", disabled=st.session_state[slider_key] >= max_idx):
                    st.session_state[slider_key] = min(max_idx, st.session_state[slider_key] + 1)
            with col4:
                play = st.button("▶️ Play" if not st.session_state[play_key] else "⏸ Pause", key="play_btn_frame_overlay")
                interval = st.number_input("Interval (sec)", min_value=0.1, max_value=10.0, value=st.session_state[interval_key], step=0.5, key="interval_input_frame_overlay")
                st.session_state[interval_key] = interval
                if play:
                    st.session_state[play_key] = not st.session_state[play_key]
            # Show both images side by side if available
            colA, colB = st.columns(2)
            idx = st.session_state[slider_key]
            with colA:
                if idx < len(frames):
                    st.image(os.path.join(frames_dir, frames[idx]), caption=f"Frame {frames[idx]}", use_container_width=True)
                else:
                    st.info("No frame image for this index.")
            with colB:
                if idx < len(overlays):
                    st.image(os.path.join(overlays_dir, overlays[idx]), caption=f"Overlay {overlays[idx]}", use_container_width=True)
                else:
                    st.info("No overlay image for this index.")
            # Auto-advance if playing
            if st.session_state[play_key]:
                import time
                time.sleep(st.session_state[interval_key])
                if st.session_state[slider_key] < max_idx:
                    st.session_state[slider_key] += 1
                else:
                    st.session_state[play_key] = False
                st.rerun()
        else:
            st.info("No frame or overlay images found in this output folder.")

        # Show Pose Points Detection JSON below images, if available
        if jsons and main_json:
            with st.expander("Pose Points Detection", expanded=False):
                with open(pose_points_json_path) as f:
                    data = json.load(f)
                st.json(data)

        # CSV Results
        if csvs:
            st.markdown("### CSV Results")
            for csvf in csvs:
                df = pd.read_csv(os.path.join(folder_path, csvf))
                st.dataframe(df)

class OverlayVideoViewer:
    """
    Displays overlay videos for a selected detection run.
    - Allows the user to select an output folder and play the overlay video (if present).
    - Prefers overlay videos with 'overlay' in the filename or matching the folder name.
    """
    def __init__(self):
        self.output_dir = "outputs"

    def render(self):
        """
        Render the overlay video viewer UI for a selected detection run.
        """
        st.subheader("View Overlay Videos")
        # List available run folders (pose_* subfolders)
        if not os.path.exists(self.output_dir):
            st.info("No outputs found. Run detection first.")
            return
        folders = [f for f in os.listdir(self.output_dir)
                   if os.path.isdir(os.path.join(self.output_dir, f)) and f.startswith("pose_")]
        if not folders:
            st.info("No output folders found.")
            return
        folder = st.selectbox("Select Output Folder", folders, key="overlay_folder_select")
        folder_path = os.path.join(self.output_dir, folder)
        files = os.listdir(folder_path)
        overlay_video_path = os.path.join(folder_path, "pose_overlay.mp4")
        print(f"DEBUG: overlay_video_path={overlay_video_path}, exists={os.path.exists(overlay_video_path)}")
        if os.path.exists(overlay_video_path):
            print("DEBUG: Entered overlay video block")
            st.markdown("### Overlay Video")
            try:
                with open(overlay_video_path, "rb") as vf:
                    video_bytes = vf.read()
                print("DEBUG: About to call st.video")
                st.video(video_bytes)
            except Exception as e:
                print(f"DEBUG: About to call st.error: {e}")
                st.error(f"Failed to load or play overlay video: {e}")
        else:
            st.info("No overlay video (pose_overlay.mp4) found in this output folder.") 