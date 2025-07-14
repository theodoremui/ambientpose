import streamlit as st
from components.uploader import FileUploader
from components.viewer import OutputViewer
from components.viewer import OverlayVideoViewer
from components.summary import SummaryPanel

st.set_page_config(page_title="AmbientPose Interactive Frontend", layout="wide")

# Main title with color and emoji
st.markdown('<h3>🍂 AmbientPose: Interactive Pose Detection Explorer 🕺🏻</h3>', unsafe_allow_html=True)

# Sidebar: navigation and advanced parameters only
with st.sidebar:
    st.header("Navigation")
    nav_pages = [
        ("Upload & Run Detection", "📤 Upload & Run Detection"),
        ("View Outputs", "🖼️ View Detection Outputs"),
        ("View Overlay Videos", "🎥 View Overlay Videos"),
        ("Summary", "📊 Summary Statistics & Visualizations"),
        ("About", "ℹ️ About")
    ]
    nav_keys = [key for key, _ in nav_pages]
    nav_labels = [label for _, label in nav_pages]
    if 'page' not in st.session_state:
        st.session_state['page'] = nav_keys[0]
    # Use st.radio for navigation with a visually hidden label for accessibility
    selected_label = st.radio(
        "Go to page:",
        nav_labels,
        index=nav_keys.index(st.session_state['page']),
        key="nav_radio",
        label_visibility="collapsed"
    )
    # Map label back to key
    label_to_key = {label: key for key, label in nav_pages}
    selected_page = label_to_key[selected_label]
    if st.session_state['page'] != selected_page:
        st.session_state['page'] = selected_page
        st.rerun()
    st.markdown("---")
    if selected_page == "Upload & Run Detection":
        st.subheader("Detection Parameters")
        FileUploader.render_sidebar()

page = st.session_state['page']

# Main panel: file upload and run detection action/results
if page == "About":
    st.markdown("""
  <h2>AmbientPose</h2>
  <h4>Modern End-to-End Human Pose Detection Platform</h4>
  <p>
    <b>Contact:</b> Theodore Mui &lt;<a href='mailto:theodoremui@gmail.com'>theodoremui@gmail.com</a>&gt;<br/>
    <b>GitHub:</b> <a href="https://github.com/theodoremui/ambientpose" target="_blank">github.com/theodoremui/ambientpose</a>
  </p>
  <h4 style="color:#009E73; font-weight:bold;">What is this project?</h4>
  <p style="font-size:1.1em; color:#444;">
    A project by <b>Theodore Mui</b> in collaboration with <b>Shardul Sapkota</b> (Ambient Intelligence Project, Stanford Institute for Human-centric AI) under the guidance of Professor <b>James Landay</b>.<br/><br/>
    <b>AmbientPose</b> is a modern, production-ready platform for human pose detection, powered by <b>AlphaPose</b>, <b>MediaPipe</b>, and <b>Ultralytics YOLO</b>.<br/>
    It offers a robust CLI, an asynchronous FastAPI backend, and a sleek Next.js/Tailwind frontend that work together to transform videos or image sequences into rich pose-estimation data and visualizations.
  </p>
    """, unsafe_allow_html=True)
elif page == "Upload & Run Detection":
    FileUploader.render_main()
elif page == "View Outputs":
    OutputViewer().render()
elif page == "View Overlay Videos":
    OverlayVideoViewer().render()
elif page == "Summary":
    SummaryPanel().render() 