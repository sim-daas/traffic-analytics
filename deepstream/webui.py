import streamlit as st
import subprocess
import os
import tempfile
import sys
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")

st.title("DeepStream Video Processing with ROI")
st.write("Upload a video, select a Region of Interest (ROI) in a separate window, and process it.")
st.write("The pipeline will count vehicles crossing the middle vertical line of the ROI.")

TRAFBOARD_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "trafboard.py")
PGIE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_inferyolov8.txt")
TRACKER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_tracker.txt")
PYTHON_EXECUTABLE = sys.executable

if 'roi_coords' not in st.session_state:
    st.session_state.roi_coords = None
if 'first_frame' not in st.session_state:
    st.session_state.first_frame = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'temp_input_path' not in st.session_state:
    st.session_state.temp_input_path = None
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'log_output' not in st.session_state:
    st.session_state.log_output = ""
if 'error_output' not in st.session_state:
    st.session_state.error_output = ""
if 'output_video_bytes' not in st.session_state:
    st.session_state.output_video_bytes = None
if 'output_file_basename' not in st.session_state:
    st.session_state.output_file_basename = None

def get_first_frame(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file.")
            return None
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            st.error("Could not read the first frame.")
            return None
    except Exception as e:
        st.error(f"Error reading video with OpenCV: {e}")
        return None

uploaded_file = st.file_uploader("Choose a video file (MP4 recommended)", type=["mp4", "mov", "avi", "mkv", "h264"])

if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_file_name:
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.roi_coords = None
    st.session_state.first_frame = None
    st.session_state.temp_input_path = None
    st.session_state.processing_done = False
    st.session_state.log_output = ""
    st.session_state.error_output = ""
    st.session_state.output_video_bytes = None
    st.session_state.output_file_basename = None

    st.write("File Uploaded:", uploaded_file.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_input_file:
        temp_input_file.write(uploaded_file.getvalue())
        st.session_state.temp_input_path = temp_input_file.name

    st.session_state.first_frame = get_first_frame(st.session_state.temp_input_path)
    if st.session_state.first_frame is None:
        if os.path.exists(st.session_state.temp_input_path):
            os.remove(st.session_state.temp_input_path)
        st.session_state.temp_input_path = None
        st.session_state.uploaded_file_name = None

if st.session_state.first_frame is not None:
    st.subheader("Select Region of Interest (ROI)")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("First Frame:")
        st.image(st.session_state.first_frame, caption="First frame of the video", use_container_width=True)

    with col2:
        st.write("Click the button below to open a window where you can draw the ROI.")
        if st.button("Select ROI using OpenCV Window"):
            st.info("Button clicked. Attempting to open OpenCV window...")
            try:
                st.info("Opening OpenCV window... Please select ROI and press ENTER or SPACE. Press C to cancel.")
                frame_bgr = cv2.cvtColor(st.session_state.first_frame, cv2.COLOR_RGB2BGR)
                roi = cv2.selectROI("Select ROI", frame_bgr, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select ROI")

                if roi != (0, 0, 0, 0):
                    x, y, w, h = roi
                    orig_frame_height, orig_frame_width, _ = st.session_state.first_frame.shape
                    x = max(0, min(x, orig_frame_width - 1))
                    y = max(0, min(y, orig_frame_height - 1))
                    w = max(1, min(w, orig_frame_width - x))
                    h = max(1, min(h, orig_frame_height - y))
                    st.session_state.roi_coords = (x, y, w, h)
                    st.success(f"ROI selected: {st.session_state.roi_coords}")
                    st.rerun()
                else:
                    st.warning("ROI selection cancelled or invalid.")
                    st.session_state.roi_coords = None

            except Exception as e:
                st.error(f"Error during ROI selection: {e}")
                st.error("Ensure you have a graphical environment (DISPLAY) available for OpenCV.")
                st.session_state.roi_coords = None

        if st.session_state.roi_coords:
            st.write(f"Current ROI: {st.session_state.roi_coords}")
            frame_with_roi = st.session_state.first_frame.copy()
            x, y, w, h = st.session_state.roi_coords
            cv2.rectangle(frame_with_roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
            mid_x = x + w // 2
            cv2.line(frame_with_roi, (mid_x, y), (mid_x, y + h), (0, 0, 255), 1)
            st.image(frame_with_roi, caption="Frame with selected ROI", use_container_width=True)
        else:
            st.write("No ROI selected yet.")

    if st.session_state.temp_input_path:
        process_button_disabled = st.session_state.roi_coords is None
        process_button = st.button("Process Video with Selected ROI", disabled=process_button_disabled)

        if process_button:
            st.session_state.processing_done = False
            st.session_state.log_output = ""
            st.session_state.error_output = ""
            st.session_state.output_video_bytes = None
            st.session_state.output_file_basename = None

            if st.session_state.roi_coords is None:
                st.warning("Cannot process without a selected ROI.")
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
                    "--roi",
                    str(st.session_state.roi_coords[0]),
                    str(st.session_state.roi_coords[1]),
                    str(st.session_state.roi_coords[2]),
                    str(st.session_state.roi_coords[3])
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
                        try:
                            os.remove(temp_output_path)
                            st.write(f"Cleaned up temporary output file: {temp_output_path}")
                        except Exception as e:
                            st.warning(f"Could not remove temporary output file {temp_output_path}: {e}")
                    if os.path.exists(temp_output_dir):
                        try:
                            os.rmdir(temp_output_dir)
                            st.write(f"Cleaned up temporary output directory: {temp_output_dir}")
                        except Exception as e:
                            st.warning(f"Could not remove temporary directory {temp_output_dir}: {e}")

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
                st.error("Processed output file not found, but no specific error was logged during processing.")

elif st.session_state.uploaded_file_name:
    st.info(f"File '{st.session_state.uploaded_file_name}' is ready. Select ROI and click Process.")
else:
    st.info("Please upload a video file to begin.")

if st.button("Clear Uploaded File and ROI"):
    if st.session_state.temp_input_path and os.path.exists(st.session_state.temp_input_path):
        try:
            os.remove(st.session_state.temp_input_path)
        except Exception as e:
            st.warning(f"Could not remove temp file {st.session_state.temp_input_path}: {e}")

    keys_to_delete = [
        'roi_coords', 'first_frame', 'uploaded_file_name', 'temp_input_path',
        'output_video_bytes', 'output_file_basename'
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.processing_done = False
    st.session_state.log_output = ""
    st.session_state.error_output = ""

    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception as e:
        st.warning(f"Could not clear Streamlit caches: {e}")

    st.rerun()
