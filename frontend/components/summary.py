import streamlit as st
import os
import json
import plotly.graph_objects as go

class SummaryPanel:
    def __init__(self):
        self.output_dir = "outputs"

    def render(self):
        st.subheader("Summary Statistics & Visualizations")
        # List available output folders
        if not os.path.exists(self.output_dir):
            st.info("No outputs found. Run detection first.")
            return
        folders = [f for f in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, f))]
        if not folders:
            st.info("No output folders found.")
            return
        folder = st.selectbox("Select Output Folder for Summary", folders)
        folder_path = os.path.join(self.output_dir, folder)
        jsons = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        if not jsons:
            st.info("No JSON summary found in this folder.")
            return
        js = st.selectbox("Select JSON File", jsons)
        with open(os.path.join(folder_path, js)) as f:
            data = json.load(f)
        st.markdown("#### Metadata")
        st.json(data.get("metadata", {}))
        st.markdown("#### Summary")
        st.json(data.get("summary", {}))
        # Example: Plot number of poses per frame if available
        poses = data.get("poses", [])
        if poses:
            frame_counts = {}
            for pose in poses:
                frame = pose.get("frame_number", 0)
                frame_counts[frame] = frame_counts.get(frame, 0) + 1
            frames = list(frame_counts.keys())
            counts = list(frame_counts.values())
            fig = go.Figure([go.Bar(x=frames, y=counts)])
            fig.update_layout(title="Poses Detected per Frame", xaxis_title="Frame", yaxis_title="# Poses")
            st.plotly_chart(fig, use_container_width=True) 