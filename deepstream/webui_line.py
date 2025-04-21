import streamlit as st
import subprocess
import os
import tempfile
import sys
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")

st.title("DeepStream Video Processing with Line Crossing")
st.write("Upload a video, define a horizontal line using sliders, and process it.")
st.write("The pipeline will count vehicles crossing the defined line segment.")

# --- Configuration ---
TRAFBOARD_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "trafboard.py")
PGIE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_inferyolov8.txt")
TRACKER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_tracker.txt")
PYTHON_EXECUTABLE = sys.executable

# --- Session State ---
if 'line_coords' not in st.session_state: st.session_state.line_coords = None # (x1, y, x2, y)
if 'first_frame' not in st.session_state: st.session_state.first_frame = None
if 'frame_height' not in st.session_state: st.session_state.frame_height = 0
if 'frame_width' not in st.session_state: st.session_state.frame_width = 0
if 'uploaded_file_name' not in st.session_state: st.session_state.uploaded_file_name = None
if 'temp_input_path' not in st.session_state: st.session_state.temp_input_path = None
if 'processing_done' not in st.session_state: st.session_state.processing_done = False
if 'log_output' not in st.session_state: st.session_state.log_output = ""
if 'error_output' not in st.session_state: st.session_state.error_output = ""
if 'output_video_bytes' not in st.session_state: st.session_state.output_video_bytes = None
if 'output_file_basename' not in st.session_state: st.session_state.output_file_basename = None
# Slider states
if 'line_y_pos' not in st.session_state: st.session_state.line_y_pos = 0
if 'line_x1_pos' not in st.session_state: st.session_state.line_x1_pos = 0
if 'line_x2_pos' not in st.session_state: st.session_state.line_x2_pos = 100 # Default end

# --- Function to get first frame ---
def get_first_frame(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file.")
            return None, 0, 0
        st.session_state.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        st.session_state.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Initialize slider defaults based on frame size if not already set by user interaction
            if st.session_state.line_y_pos == 0:
                st.session_state.line_y_pos = st.session_state.frame_height // 2
            if st.session_state.line_x1_pos == 0 and st.session_state.line_x2_pos == 100:
                 st.session_state.line_x1_pos = st.session_state.frame_width // 4
                 st.session_state.line_x2_pos = st.session_state.frame_width * 3 // 4

            return frame_rgb, st.session_state.frame_height, st.session_state.frame_width
        else:
            st.error("Could not read the first frame.")
            return None, 0, 0
    except Exception as e:
        st.error(f"Error reading video with OpenCV: {e}")
        return None, 0, 0

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a video file (MP4 recommended)", type=["mp4", "mov", "avi", "mkv", "h264"])

if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_file_name:
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.line_coords = None
    st.session_state.first_frame = None
    st.session_state.temp_input_path = None
    st.session_state.processing_done = False
    st.session_state.log_output = ""
    st.session_state.error_output = ""
    st.session_state.output_video_bytes = None
    st.session_state.output_file_basename = None
    st.session_state.frame_height = 0
    st.session_state.frame_width = 0
    # Reset slider positions for new video
    st.session_state.line_y_pos = 0
    st.session_state.line_x1_pos = 0
    st.session_state.line_x2_pos = 100

    st.write("File Uploaded:", uploaded_file.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_input_file:
        temp_input_file.write(uploaded_file.getvalue())
        st.session_state.temp_input_path = temp_input_file.name

    st.session_state.first_frame, st.session_state.frame_height, st.session_state.frame_width = get_first_frame(st.session_state.temp_input_path)
    if st.session_state.first_frame is None:
        if os.path.exists(st.session_state.temp_input_path):
            os.remove(st.session_state.temp_input_path)
        st.session_state.temp_input_path = None
        st.session_state.uploaded_file_name = None

# --- Line Definition ---
if st.session_state.first_frame is not None:
    st.subheader("Define Crossing Line")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.write("Adjust Line Position and Extents:")
        st.session_state.line_y_pos = st.slider(
            "Line Y Position", 0, st.session_state.frame_height - 1, st.session_state.line_y_pos
        )
        # Use columns for x sliders to place them side-by-side
        x_col1, x_col2 = st.columns(2)
        with x_col1:
            st.session_state.line_x1_pos = st.slider(
                "Line Start X (x1)", 0, st.session_state.frame_width - 1, st.session_state.line_x1_pos
            )
        with x_col2:
            # Ensure x2 is always >= x1
            min_x2 = st.session_state.line_x1_pos
            st.session_state.line_x2_pos = st.slider(
                "Line End X (x2)", min_x2, st.session_state.frame_width - 1, max(min_x2, st.session_state.line_x2_pos)
            )

        # Update line coordinates in session state
        st.session_state.line_coords = (
            st.session_state.line_x1_pos,
            st.session_state.line_y_pos,
            st.session_state.line_x2_pos,
            st.session_state.line_y_pos # y1 and y2 are the same for horizontal line
        )
        st.write(f"Current Line: ({st.session_state.line_coords[0]}, {st.session_state.line_coords[1]}) to ({st.session_state.line_coords[2]}, {st.session_state.line_coords[3]})")


    with col1:
        st.write("First Frame with Line:")
        frame_with_line = st.session_state.first_frame.copy()
        if st.session_state.line_coords:
            x1, y, x2, _ = st.session_state.line_coords
            # Draw the line segment (Blue, thickness 2)
            cv2.line(frame_with_line, (x1, y), (x2, y), (0, 0, 255), 2)
        st.image(frame_with_line, caption="Frame with defined line", use_container_width=True)


    # --- Processing ---
    if st.session_state.temp_input_path:
        process_button_disabled = st.session_state.line_coords is None
        process_button = st.button("Process Video with Defined Line", disabled=process_button_disabled)

        if process_button:
            st.session_state.processing_done = False
            st.session_state.log_output = ""
            st.session_state.error_output = ""
            st.session_state.output_video_bytes = None
            st.session_state.output_file_basename = None

            if st.session_state.line_coords is None:
                st.warning("Cannot process without a defined line.")
            else:
                temp_output_dir = tempfile.mkdtemp()
                temp_output_path = os.path.join(temp_output_dir, "processed_" + os.path.splitext(st.session_state.uploaded_file_name)[0] + ".mp4")

                st.info("Processing video...")
                cmd = [
                    PYTHON_EXECUTABLE,
                    TRAFBOARD_SCRIPT_PATH,
                    "--pgie-config", PGIE_CONFIG_PATH,
                    "--tracker-config", TRACKER_CONFIG_PATH,
                    st.session_state.temp_input_path,
                    temp_output_path,
                    "--line", # Use the new argument
                    str(st.session_state.line_coords[0]),  # x1
                    str(st.session_state.line_coords[1]),  # y1 (same as y2)
                    str(st.session_state.line_coords[2]),  # x2
                    str(st.session_state.line_coords[3])   # y2 (same as y1)
                ]
                st.info(f"Running command: {' '.join(cmd)}")

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    st.session_state.log_output = result.stdout + "\n" + result.stderr
                    st.session_state.processing_done = True
                    st.success("Video processing complete!")

                    if os.path.exists(temp_output_path):
                        try:
                            with open(temp_output_path, 'rb') as video_file:
                                st.session_state.output_video_bytes = video_file.read()
                            st.session_state.output_file_basename = os.path.basename(temp_output_path)
                        except Exception as e:
                            st.error(f"Error reading or displaying processed video: {e}")
                    else:
                        st.error("Processed output file not found. Check the logs for errors.")

                except subprocess.CalledProcessError as e:
                    st.error("Video processing failed.")
                    st.session_state.error_output = e.stdout + "\n" + e.stderr
                    st.session_state.processing_done = True
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.error_output = str(e)
                    st.session_state.processing_done = True
                finally:
                    if os.path.exists(temp_output_path):
                        try: os.remove(temp_output_path)
                        except Exception as e: st.warning(f"Could not remove temp output file: {e}")
                    if os.path.exists(temp_output_dir):
                        try: os.rmdir(temp_output_dir)
                        except Exception as e: st.warning(f"Could not remove temp output dir: {e}")

        if st.session_state.processing_done:
            if st.session_state.error_output:
                st.subheader("DeepStream Error Log:")
                st.text_area("Error Log", st.session_state.error_output, height=300, key="error_log_area")
            if st.session_state.log_output:
                st.subheader("DeepStream Output Log:")
                st.text_area("Output Log", st.session_state.log_output, height=200, key="output_log_area")
            if st.session_state.output_video_bytes:
                st.subheader("Processed Video Output")
                st.video(st.session_state.output_video_bytes)
                st.download_button(
                    label="Download Processed Video",
                    data=st.session_state.output_video_bytes,
                    file_name=st.session_state.output_file_basename,
                    mime="video/mp4"
                )
            elif not st.session_state.error_output:
                st.error("Processed output file not found, but no specific error was logged.")

elif st.session_state.uploaded_file_name:
    st.info(f"File '{st.session_state.uploaded_file_name}' is ready. Define line and click Process.")
else:
    st.info("Please upload a video file to begin.")

# --- Clear Button ---
if st.button("Clear Uploaded File and Line"):
    if st.session_state.temp_input_path and os.path.exists(st.session_state.temp_input_path):
        try: os.remove(st.session_state.temp_input_path)
        except Exception as e: st.warning(f"Could not remove temp file: {e}")

    keys_to_delete = [
        'line_coords', 'first_frame', 'uploaded_file_name', 'temp_input_path',
        'output_video_bytes', 'output_file_basename', 'frame_height', 'frame_width',
        'line_y_pos', 'line_x1_pos', 'line_x2_pos'
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.processing_done = False
    st.session_state.log_output = ""
    st.session_state.error_output = ""
    # Re-initialize slider defaults
    st.session_state.line_y_pos = 0
    st.session_state.line_x1_pos = 0
    st.session_state.line_x2_pos = 100

    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception as e:
        st.warning(f"Could not clear Streamlit caches: {e}")

    st.rerun()
