import streamlit as st
import subprocess
import os
import tempfile
import sys
import cv2  # Import OpenCV
import numpy as np
from PIL import Image  # To display the frame in Streamlit

st.set_page_config(layout="wide")

st.title("DeepStream Video Processing with ROI")
st.write("Upload a video, select a Region of Interest (ROI) in a separate window, and process it.")
st.write("The pipeline will count vehicles crossing the middle vertical line of the ROI.")
st.warning("Note: Selecting the ROI will open a separate OpenCV window.")

# --- Configuration ---
TRAFBOARD_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "trafboard.py")
PGIE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_inferyolov8.txt")
TRACKER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_tracker.txt")
PYTHON_EXECUTABLE = sys.executable  # Use the same python that runs streamlit

# --- Session State ---
# Initialize session state variables if they don't exist
if 'roi_coords' not in st.session_state:
    st.session_state.roi_coords = None
if 'first_frame' not in st.session_state:
    st.session_state.first_frame = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'temp_input_path' not in st.session_state:
    st.session_state.temp_input_path = None

# --- Function to get first frame ---
def get_first_frame(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file.")
            return None
        ret, frame = cap.read()
        cap.release()
        if ret:
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            st.error("Could not read the first frame.")
            return None
    except Exception as e:
        st.error(f"Error reading video with OpenCV: {e}")
        return None

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a video file (MP4 recommended)", type=["mp4", "mov", "avi", "mkv", "h264"])

# Process uploaded file only if it's new or hasn't been processed yet
if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_file_name:
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.roi_coords = None  # Reset ROI when new file is uploaded
    st.session_state.first_frame = None
    st.session_state.temp_input_path = None

    st.write("File Uploaded:", uploaded_file.name)

    # Save to a temporary file to allow OpenCV to read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_input_file:
        temp_input_file.write(uploaded_file.getvalue())
        st.session_state.temp_input_path = temp_input_file.name

    # Get and store the first frame
    st.session_state.first_frame = get_first_frame(st.session_state.temp_input_path)
    if st.session_state.first_frame is None:
        # Cleanup if frame extraction failed
        if os.path.exists(st.session_state.temp_input_path):
            os.remove(st.session_state.temp_input_path)
        st.session_state.temp_input_path = None
        st.session_state.uploaded_file_name = None

# --- ROI Selection ---
if st.session_state.first_frame is not None:
    st.subheader("Select Region of Interest (ROI)")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("First Frame:")
        # Display the first frame using st.image
        st.image(st.session_state.first_frame, caption="First frame of the video", use_column_width=True)

    with col2:
        st.write("Click the button below to open a window where you can draw the ROI.")
        if st.button("Select ROI using OpenCV Window"):
            try:
                st.info("Opening OpenCV window... Please select ROI and press ENTER or SPACE. Press C to cancel.")
                # Convert RGB frame back to BGR for cv2.selectROI
                frame_bgr = cv2.cvtColor(st.session_state.first_frame, cv2.COLOR_RGB2BGR)
                # Use selectROI
                roi = cv2.selectROI("Select ROI", frame_bgr, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select ROI")  # Close the window immediately

                # roi is (x, y, w, h) or (0,0,0,0) if cancelled
                if roi != (0, 0, 0, 0):
                    # Clamp coordinates just in case
                    x, y, w, h = roi
                    orig_frame_height, orig_frame_width, _ = st.session_state.first_frame.shape
                    x = max(0, min(x, orig_frame_width - 1))
                    y = max(0, min(y, orig_frame_height - 1))
                    w = max(1, min(w, orig_frame_width - x))
                    h = max(1, min(h, orig_frame_height - y))
                    st.session_state.roi_coords = (x, y, w, h)
                    st.success(f"ROI selected: {st.session_state.roi_coords}")
                    # Force rerun to update display if needed, though state change might be enough
                    st.rerun()
                else:
                    st.warning("ROI selection cancelled or invalid.")
                    st.session_state.roi_coords = None

            except Exception as e:
                st.error(f"Error during ROI selection: {e}")
                st.error("Ensure you have a graphical environment (DISPLAY) available for OpenCV.")
                st.session_state.roi_coords = None

        # Display the currently selected ROI
        if st.session_state.roi_coords:
            st.write(f"Current ROI: {st.session_state.roi_coords}")
            # Optionally show the frame with ROI drawn here as well
            frame_with_roi = st.session_state.first_frame.copy()
            x, y, w, h = st.session_state.roi_coords
            cv2.rectangle(frame_with_roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
            mid_x = x + w // 2
            cv2.line(frame_with_roi, (mid_x, y), (mid_x, y + h), (0, 0, 255), 1)
            st.image(frame_with_roi, caption="Frame with selected ROI", use_column_width=True)
        else:
            st.write("No ROI selected yet.")

    # --- Processing ---
    if st.session_state.temp_input_path:  # Check if input file exists
        # Disable button if ROI is not selected
        process_button_disabled = st.session_state.roi_coords is None
        process_button = st.button("Process Video with Selected ROI", disabled=process_button_disabled)

        if process_button:
            # Redundant check, but safe
            if st.session_state.roi_coords is None:
                st.warning("Cannot process without a selected ROI.")
            else:
                # Define output path
                temp_output_dir = tempfile.mkdtemp()
                temp_output_path = os.path.join(temp_output_dir, "processed_" + os.path.splitext(st.session_state.uploaded_file_name)[0] + ".mp4")

                st.info("Processing video...")
                # Construct command with ROI arguments
                cmd = [
                    PYTHON_EXECUTABLE,
                    TRAFBOARD_SCRIPT_PATH,
                    "--pgie-config", PGIE_CONFIG_PATH,
                    "--tracker-config", TRACKER_CONFIG_PATH,
                    st.session_state.temp_input_path,  # Use stored temp input path
                    temp_output_path,
                    "--roi",
                    str(st.session_state.roi_coords[0]),  # x
                    str(st.session_state.roi_coords[1]),  # y
                    str(st.session_state.roi_coords[2]),  # w
                    str(st.session_state.roi_coords[3])  # h
                ]
                st.info(f"Running command: {' '.join(cmd)}")

                try:
                    # Run the modified trafboard.py script
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True  # Raise an exception if the command fails
                    )
                    st.success("Video processing complete!")
                    st.subheader("DeepStream Output Log:")
                    st.text_area("Log", result.stdout + "\n" + result.stderr, height=200)

                    # --- Display Output ---
                    if os.path.exists(temp_output_path):
                        st.subheader("Processed Video Output")
                        try:
                            with open(temp_output_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                            st.video(video_bytes)
                            # Provide download link
                            st.download_button(
                                label="Download Processed Video",
                                data=video_bytes,
                                file_name=os.path.basename(temp_output_path),
                                mime="video/mp4"
                            )
                        except Exception as e:
                            st.error(f"Error reading or displaying processed video: {e}")
                    else:
                        st.error("Processed output file not found. Check the logs for errors.")

                except subprocess.CalledProcessError as e:
                    st.error("Video processing failed.")
                    st.subheader("DeepStream Error Log:")
                    st.text_area("Log", e.stdout + "\n" + e.stderr, height=300)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                finally:
                    # --- Cleanup ---
                    # Keep input file until a new one is uploaded
                    # Clean up output dir/file after display/download attempt
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
            # Don't clean up temp_input_path here, keep it until new upload

elif st.session_state.uploaded_file_name:
    st.info(f"File '{st.session_state.uploaded_file_name}' is ready. Select ROI and click Process.")
else:
    st.info("Please upload a video file to begin.")

# Add a button to clear state if needed
if st.button("Clear Uploaded File and ROI"):
    if st.session_state.temp_input_path and os.path.exists(st.session_state.temp_input_path):
        try:
            os.remove(st.session_state.temp_input_path)
        except Exception as e:
            st.warning(f"Could not remove temp file {st.session_state.temp_input_path}: {e}")
    st.session_state.roi_coords = None
    st.session_state.first_frame = None
    st.session_state.uploaded_file_name = None
    st.session_state.temp_input_path = None
    st.rerun()
