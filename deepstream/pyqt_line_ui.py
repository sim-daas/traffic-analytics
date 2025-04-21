import sys
import os
import subprocess
import cv2
from PyQt6 import QtWidgets, QtGui, QtCore, QtMultimedia, QtMultimediaWidgets

# --- Configuration ---
TRAFBOARD_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "trafboard.py")
PGIE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_inferyolov8.txt")
TRACKER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_tracker.txt")
PYTHON_EXECUTABLE = sys.executable
OUTPUT_VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_VIDEO_DIR, "output.mp4")

# Theme colors (for preview drawing)
COLOR_LINE_PREVIEW = QtGui.QColor("#81a1c1") # Primary blue
COLOR_ROI_PREVIEW = QtGui.QColor("#88c0d0") # Secondary cyan
PREVIEW_LINE_THICKNESS = 2

class ImageLabel(QtWidgets.QLabel):
    """ Custom QLabel to handle mouse events for drawing lines and ROIs """
    line_drawn = QtCore.pyqtSignal(int, int, int, int) # x1, y, x2, y
    roi_drawn = QtCore.pyqtSignal(int, int, int, int) # x1, y1, x2, y2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_point = None
        self.end_point = None
        self.current_line_coords = None # Store the final line
        self.current_roi_coords = None # Store the final ROI
        self.original_pixmap = None
        self.drawing = False
        self.draw_mode = 'line' # Default mode: 'line' or 'roi'

    def setPixmap(self, pixmap: QtGui.QPixmap | None):
        self.original_pixmap = pixmap.copy() if pixmap else None
        super().setPixmap(pixmap)

    def set_draw_mode(self, mode):
        if mode in ['line', 'roi']:
            self.draw_mode = mode
            self.drawing = False # Stop any current drawing on mode change
            self.start_point = None
            self.end_point = None
            self.update() # Redraw to clear temporary shapes

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if self.original_pixmap and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.drawing = True
            # Clear previous shape of the current mode visually for redraw
            if self.draw_mode == 'line':
                self.current_line_coords = None
            else: # roi mode
                self.current_roi_coords = None
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self.original_pixmap and self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if self.original_pixmap and self.drawing and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.drawing = False
            self.end_point = event.pos()

            pixmap_width = self.original_pixmap.width() if self.original_pixmap else 0
            pixmap_height = self.original_pixmap.height() if self.original_pixmap else 0

            # Clamp start and end points
            sx = max(0, min(self.start_point.x(), pixmap_width - 1))
            sy = max(0, min(self.start_point.y(), pixmap_height - 1))
            ex = max(0, min(self.end_point.x(), pixmap_width - 1))
            ey = max(0, min(self.end_point.y(), pixmap_height - 1))

            if self.draw_mode == 'line':
                # Ensure line is horizontal (use start_point's y)
                y = sy
                x1 = min(sx, ex)
                x2 = max(sx, ex)
                self.current_line_coords = (x1, y, x2, y)
                self.line_drawn.emit(*self.current_line_coords)
            else: # roi mode
                x1 = min(sx, ex)
                y1 = min(sy, ey)
                x2 = max(sx, ex)
                y2 = max(sy, ey)
                # Ensure ROI has non-zero width and height
                if x1 < x2 and y1 < y2:
                    self.current_roi_coords = (x1, y1, x2, y2)
                    self.roi_drawn.emit(*self.current_roi_coords)
                else:
                    self.current_roi_coords = None # Discard zero-size ROI

            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        if not self.original_pixmap:
            super().paintEvent(event)
            return

        painter = QtGui.QPainter(self)
        painter.drawPixmap(self.rect(), self.original_pixmap)

        # Draw final shapes first (if they exist)
        if self.current_line_coords:
            pen = QtGui.QPen(COLOR_LINE_PREVIEW, PREVIEW_LINE_THICKNESS, QtCore.Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawLine(*self.current_line_coords)
        if self.current_roi_coords:
            pen = QtGui.QPen(COLOR_ROI_PREVIEW, PREVIEW_LINE_THICKNESS, QtCore.Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(QtCore.QRectF(QtCore.QPointF(self.current_roi_coords[0], self.current_roi_coords[1]),
                                           QtCore.QPointF(self.current_roi_coords[2], self.current_roi_coords[3])))

        # Draw temporary shape during drag
        if self.drawing and self.start_point and self.end_point:
            if self.draw_mode == 'line':
                pen = QtGui.QPen(COLOR_LINE_PREVIEW, PREVIEW_LINE_THICKNESS, QtCore.Qt.PenStyle.DotLine)
                painter.setPen(pen)
                y = self.start_point.y()
                x1 = min(self.start_point.x(), self.end_point.x())
                x2 = max(self.start_point.x(), self.end_point.x())
                painter.drawLine(x1, y, x2, y)
            else: # roi mode
                pen = QtGui.QPen(COLOR_ROI_PREVIEW, PREVIEW_LINE_THICKNESS, QtCore.Qt.PenStyle.DotLine)
                painter.setPen(pen)
                # Convert QPoint to QPointF for QRectF constructor
                start_point_f = QtCore.QPointF(self.start_point)
                end_point_f = QtCore.QPointF(self.end_point)
                painter.drawRect(QtCore.QRectF(start_point_f, end_point_f).normalized())

        painter.end()


class ProcessWorker(QtCore.QThread):
    log_message = QtCore.pyqtSignal(str)
    process_finished = QtCore.pyqtSignal(int) # Emit return code

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        try:
            # Use Popen to allow reading output incrementally
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Redirect stderr to stdout
                text=True,
                bufsize=1, # Line buffered
                universal_newlines=True
            )

            # Read output line by line
            for line in process.stdout:
                self.log_message.emit(line.strip())

            process.wait() # Wait for the process to complete
            self.process_finished.emit(process.returncode)

        except Exception as e:
            self.log_message.emit(f"Error starting/reading process: {e}")
            self.process_finished.emit(-1) # Indicate an error


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Line Definition for DeepStream (Dark)")
        self.setGeometry(100, 100, 1300, 800) # Adjusted size

        self.video_path = None
        self.original_frame = None
        self.original_width = 0
        self.original_height = 0
        self.display_width = 800 # Increased display width
        self.display_height = 0
        self.scale_x_display = 1.0
        self.scale_y_display = 1.0
        self.line_coords_original = None # Line coordinates relative to original frame
        self.roi_coords_original = None # Store drawn ROI coords
        self.output_video_path = None # Store path for video player
        self.processing_thread = None # To hold the worker thread

        # --- Widgets ---
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QGridLayout(self.central_widget)

        # --- Row 0: File selection ---
        self.btn_select_video = QtWidgets.QPushButton("Select Video")
        self.lbl_video_path = QtWidgets.QLabel("No video selected")
        self.layout.addWidget(self.btn_select_video, 0, 0)
        self.layout.addWidget(self.lbl_video_path, 0, 1, 1, 2) # Span 2 columns

        # --- Row 1: Image display and Video Player ---
        self.image_label = ImageLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.layout.addWidget(self.image_label, 1, 0)

        self.video_group = QtWidgets.QGroupBox("Output Video")
        self.video_layout = QtWidgets.QVBoxLayout()
        self.video_widget = QtMultimediaWidgets.QVideoWidget()
        self.media_player = QtMultimedia.QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        self.video_control_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.video_control_layout.addWidget(self.btn_play)
        self.video_control_layout.addWidget(self.btn_pause)
        self.video_control_layout.addWidget(self.btn_stop)
        self.video_layout.addWidget(self.video_widget, 1)
        self.video_layout.addLayout(self.video_control_layout)
        self.video_group.setLayout(self.video_layout)
        self.layout.addWidget(self.video_group, 1, 1, 1, 2)

        # --- Row 2: Controls and Logs ---
        self.controls_group = QtWidgets.QGroupBox("Controls")
        self.controls_layout = QtWidgets.QVBoxLayout()

        # --- Drawing Mode Selection ---
        self.draw_mode_group = QtWidgets.QGroupBox("Draw Mode")
        self.draw_mode_layout = QtWidgets.QHBoxLayout()
        self.radio_draw_line = QtWidgets.QRadioButton("Line")
        self.radio_draw_roi = QtWidgets.QRadioButton("ROI")
        self.radio_draw_line.setChecked(True) # Default to line
        self.draw_mode_layout.addWidget(self.radio_draw_line)
        self.draw_mode_layout.addWidget(self.radio_draw_roi)
        self.draw_mode_group.setLayout(self.draw_mode_layout)
        # --- End Drawing Mode Selection ---

        self.lbl_line_coords = QtWidgets.QLabel("Line Coords (Original): None")
        self.lbl_roi_coords = QtWidgets.QLabel("ROI Coords (Original): None") # Label to display ROI

        self.btn_process = QtWidgets.QPushButton("Process Video")
        self.btn_process.setEnabled(False) # Disabled until line is drawn

        self.controls_layout.addWidget(self.draw_mode_group) # Add draw mode selection
        self.controls_layout.addWidget(self.lbl_line_coords)
        self.controls_layout.addWidget(self.lbl_roi_coords) # Add ROI display label
        self.controls_layout.addWidget(self.btn_process)
        self.controls_layout.addStretch()
        self.controls_group.setLayout(self.controls_layout)
        self.layout.addWidget(self.controls_group, 2, 0)

        self.log_group = QtWidgets.QGroupBox("Logs")
        self.log_layout = QtWidgets.QVBoxLayout()
        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        # --- Set specific font for logs ---
        log_font = QtGui.QFont()
        log_font.setPointSize(10) # Slightly smaller than default if desired
        log_font.setBold(False)   # Ensure it's not bold
        self.log_output.setFont(log_font)
        # --- End set specific font ---
        self.log_layout.addWidget(self.log_output)
        self.log_group.setLayout(self.log_layout)
        self.layout.addWidget(self.log_group, 2, 1, 1, 2)

        # --- Row 3: Status Label ---
        self.lbl_status = QtWidgets.QLabel("")
        self.layout.addWidget(self.lbl_status, 3, 0, 1, 3)

        # --- Set column/row stretch factors ---
        self.layout.setColumnStretch(0, 2) # Image/Controls column wider
        self.layout.setColumnStretch(1, 1) # Video column start
        self.layout.setColumnStretch(2, 1) # Video column end
        self.layout.setRowStretch(1, 3) # Middle row (image/video) taller
        self.layout.setRowStretch(2, 1) # Log row

        # --- Connections ---
        self.btn_select_video.clicked.connect(self.select_video)
        self.image_label.line_drawn.connect(self.update_line_coords)
        self.image_label.roi_drawn.connect(self.update_roi_coords) # Connect ROI signal
        self.btn_process.clicked.connect(self.process_video)
        # Drawing mode connections
        self.radio_draw_line.toggled.connect(lambda: self.image_label.set_draw_mode('line'))
        self.radio_draw_roi.toggled.connect(lambda: self.image_label.set_draw_mode('roi'))
        # Video controls
        self.btn_play.clicked.connect(self.media_player.play)
        self.btn_pause.clicked.connect(self.media_player.pause)
        self.btn_stop.clicked.connect(self.media_player.stop)
        # Update UI based on media player state
        self.media_player.playbackStateChanged.connect(self.update_video_buttons)
        self.media_player.errorOccurred.connect(self.media_player_error)

        # Initial button states
        self.update_video_buttons()


    def select_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                             "Video Files (*.mp4 *.avi *.mov *.mkv *.h264);;All Files (*)")
        if file_path:
            self.video_path = file_path
            self.lbl_video_path.setText(os.path.basename(file_path))
            self.log_output.clear()
            self.line_coords_original = None
            self.roi_coords_original = None # Reset ROI on new video
            self.lbl_line_coords.setText("Line Coords (Original): None")
            self.lbl_roi_coords.setText("ROI Coords (Original): None") # Reset ROI label
            # Reset drawing shapes in ImageLabel
            self.image_label.current_line_coords = None
            self.image_label.current_roi_coords = None
            self.radio_draw_line.setChecked(True) # Reset draw mode
            self.btn_process.setEnabled(False)
            self.load_first_frame()
            self.media_player.stop() # Stop previous video
            self.media_player.setSource(QtCore.QUrl()) # Clear source
            self.output_video_path = None
            self.update_video_buttons()
            self.lbl_status.setText("") # Clear status

    def load_first_frame(self):
        if not self.video_path:
            return
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.show_error("Error opening video file.")
                return

            self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            ret, frame = cap.read()
            cap.release()

            if ret:
                self.original_frame = frame # Keep original BGR frame
                if self.original_width <= 0 or self.original_height <= 0:
                    self.show_error("Invalid video dimensions.")
                    return

                # Calculate display size and scaling
                self.scale_y_display = self.display_width / self.original_width
                self.display_height = int(self.original_height * self.scale_y_display)
                self.scale_x_display = self.scale_y_display # Assuming square pixels

                # Resize frame for display
                display_frame = cv2.resize(self.original_frame, (self.display_width, self.display_height))

                # Convert BGR to RGB for QPixmap
                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qt_image)

                self.image_label.setFixedSize(self.display_width, self.display_height)
                self.image_label.setPixmap(pixmap)

            else:
                self.show_error("Could not read the first frame.")
                self.image_label.clear()
                self.image_label.setFixedSize(self.display_width, 200) # Default size

        except Exception as e:
            self.show_error(f"Error loading frame: {e}")
            self.image_label.clear()
            self.image_label.setFixedSize(self.display_width, 200)

    def update_line_coords(self, x1_disp, y_disp, x2_disp, y_disp2):
        # Convert display coordinates back to original frame coordinates
        if self.scale_x_display > 0 and self.scale_y_display > 0:
            x1_orig = int(x1_disp / self.scale_x_display)
            y_orig = int(y_disp / self.scale_y_display)
            x2_orig = int(x2_disp / self.scale_x_display)

            # Clamp to original dimensions
            x1_orig = max(0, min(x1_orig, self.original_width - 1))
            y_orig = max(0, min(y_orig, self.original_height - 1))
            x2_orig = max(x1_orig, min(x2_orig, self.original_width - 1))

            self.line_coords_original = (x1_orig, y_orig, x2_orig, y_orig)
            self.lbl_line_coords.setText(f"Line Coords (Original): {self.line_coords_original}")
            self.btn_process.setEnabled(True) # Enable processing once line is drawn
        else:
            self.show_error("Cannot calculate original line coordinates (invalid scale).")
            self.line_coords_original = None
            self.btn_process.setEnabled(False)

    def update_roi_coords(self, x1_disp, y1_disp, x2_disp, y2_disp):
        """Slot to receive ROI coordinates from ImageLabel."""
        if self.scale_x_display > 0 and self.scale_y_display > 0:
            x1_orig = int(x1_disp / self.scale_x_display)
            y1_orig = int(y1_disp / self.scale_y_display)
            x2_orig = int(x2_disp / self.scale_x_display)
            y2_orig = int(y2_disp / self.scale_y_display)

            # Clamp to original dimensions
            x1_orig = max(0, min(x1_orig, self.original_width - 1))
            y1_orig = max(0, min(y1_orig, self.original_height - 1))
            x2_orig = max(x1_orig, min(x2_orig, self.original_width - 1))
            y2_orig = max(y1_orig, min(y2_orig, self.original_height - 1))

            # Ensure valid ROI after clamping
            if x1_orig < x2_orig and y1_orig < y2_orig:
                self.roi_coords_original = (x1_orig, y1_orig, x2_orig, y2_orig)
                self.lbl_roi_coords.setText(f"ROI Coords (Original): {self.roi_coords_original}")
            else:
                 self.show_error("ROI became invalid after clamping to video dimensions.")
                 self.roi_coords_original = None
                 self.lbl_roi_coords.setText("ROI Coords (Original): Invalid")
        else:
            self.show_error("Cannot calculate original ROI coordinates (invalid scale).")
            self.roi_coords_original = None
            self.lbl_roi_coords.setText("ROI Coords (Original): Error")

    def _validate_and_get_roi(self):
        """Returns the drawn ROI coordinates if available, otherwise None."""
        # Now primarily relies on the drawn ROI stored in self.roi_coords_original
        return self.roi_coords_original


    def process_video(self):
        if not self.video_path or not self.line_coords_original:
            self.show_error("Please select a video and draw a line first.")
            return
        if self.processing_thread and self.processing_thread.isRunning():
            self.show_error("Processing is already in progress.")
            return

        # --- Get ROI coordinates (now from drawing) ---
        current_roi = self._validate_and_get_roi()
        # No need for "ERROR" check as validation happens during drawing/update
        # --- End Get ROI ---

        # Ensure output directory exists
        try:
            os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
        except OSError as e:
            self.show_error(f"Failed to create output directory {OUTPUT_VIDEO_DIR}: {e}")
            return

        self.output_video_path = OUTPUT_VIDEO_PATH

        cmd = [
            PYTHON_EXECUTABLE,
            TRAFBOARD_SCRIPT_PATH,
            "--pgie-config", PGIE_CONFIG_PATH,
            "--tracker-config", TRACKER_CONFIG_PATH,
            self.video_path,
            self.output_video_path,
            "--line",
            str(self.line_coords_original[0]),
            str(self.line_coords_original[1]),
            str(self.line_coords_original[2]),
            str(self.line_coords_original[3])
        ]

        # --- Add ROI arguments if available ---
        if current_roi:
            cmd.extend([
                "--roi",
                str(current_roi[0]), # xmin
                str(current_roi[1]), # ymin
                str(current_roi[2]), # xmax
                str(current_roi[3])  # ymax
            ])
        # --- End Add ROI ---

        self.log_output.clear()
        self.log_output.append("Starting processing...")
        self.log_output.append(f"Command: {' '.join(cmd)}")
        self.btn_process.setEnabled(False)
        self.lbl_status.setText("Status: Processing...") # Show status

        self.media_player.stop()
        self.media_player.setSource(QtCore.QUrl())
        self.update_video_buttons()

        # Create and start the worker thread
        self.processing_thread = ProcessWorker(cmd)
        self.processing_thread.log_message.connect(self.append_log)
        self.processing_thread.process_finished.connect(self.handle_process_finished)
        self.processing_thread.start()

    @QtCore.pyqtSlot(str)
    def append_log(self, message):
        self.log_output.append(message)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum()) # Auto-scroll

    @QtCore.pyqtSlot(int)
    def handle_process_finished(self, return_code):
        self.lbl_status.setText(f"Status: Processing finished (Code: {return_code})")
        self.btn_process.setEnabled(self.line_coords_original is not None) # Re-enable if line exists

        if return_code == 0:
            self.log_output.append(f"\nProcessing finished successfully. Output saved to: {self.output_video_path}")
            if os.path.exists(self.output_video_path):
                self.media_player.setSource(QtCore.QUrl.fromLocalFile(self.output_video_path))
                self.log_output.append("Output video loaded. Press Play.")
                self.update_video_buttons() # Enable play button
            else:
                self.log_output.append("Output video file not found after processing.")
        else:
            self.log_output.append(f"\nProcessing failed with return code {return_code}.")

        self.processing_thread = None # Clear the thread reference

    def update_video_buttons(self):
        state = self.media_player.playbackState()
        has_source = not self.media_player.source().isEmpty()

        self.btn_play.setEnabled(has_source and state != QtMultimedia.QMediaPlayer.PlaybackState.PlayingState)
        self.btn_pause.setEnabled(state == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState)
        self.btn_stop.setEnabled(has_source and state != QtMultimedia.QMediaPlayer.PlaybackState.StoppedState)

    def media_player_error(self, error, error_string):
        self.show_error(f"Media Player Error: {error_string} (Code: {error})")
        self.log_output.append(f"ERROR (Media Player): {error_string}")
        self.update_video_buttons()

    def show_error(self, message):
        QtWidgets.QMessageBox.critical(self, "Error", message)
        self.log_output.append(f"ERROR: {message}")


def set_dark_theme(app):
    app.setStyle("Fusion")

    # Define colors from the provided palette
    background = QtGui.QColor("#1f2229") # Use background-alt for main window
    background_base = QtGui.QColor("#000000") # True black for base elements like text edit
    foreground = QtGui.QColor("#dfdfdf")
    primary = QtGui.QColor("#81a1c1")
    secondary = QtGui.QColor("#88c0d0") # Use for borders?
    alert = QtGui.QColor("#bf616a")
    disabled = QtGui.QColor("#707880")
    orange = QtGui.QColor("#FFA500") # Keep for potential future use

    dark_palette = QtGui.QPalette()

    # Base colors
    dark_palette.setColor(QtGui.QPalette.ColorRole.Window, background)
    dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, foreground)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Base, background_base) # Base for inputs
    dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, background) # Used in views like tables/lists
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, background)
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, foreground)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Text, foreground)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Button, background) # Button background controlled by stylesheet
    dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, foreground)
    dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, alert)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Link, primary)

    # Highlight colors
    dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, primary)
    dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, background_base) # Black text on primary highlight

    # Disabled colors
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text, disabled)
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText, disabled)
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.WindowText, disabled)
    # Slightly darker highlight for disabled items
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Highlight, background.darker(120))
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.HighlightedText, disabled)

    app.setPalette(dark_palette)

    # Define some derived colors for the stylesheet
    button_bg = background.lighter(115) # Slightly lighter than main background
    button_hover_bg = background.lighter(130)
    button_pressed_bg = background.lighter(105)
    button_disabled_bg = background.darker(110)
    border_color = secondary.darker(120).name() # Use a darker secondary for borders
    border_hover_color = secondary.name() # Lighter border on hover

    app.setStyleSheet(f"""
        QWidget {{
            color: {foreground.name()};
            background-color: {background.name()};
        }}
        QGroupBox {{
            border: 2px solid {border_color}; /* Increased border width */
            border-radius: 8px; /* Increased rounding */
            margin-top: 1em; /* Add space above groupbox title */
            margin-bottom: 5px; /* Add space below groupbox */
            margin-left: 5px; /* Add space to the left */
            margin-right: 5px; /* Add space to the right */
            padding: 15px 8px 8px 8px; /* Adjust padding top for title */
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left; /* Position title */
            left: 10px;
            padding: 0 5px 0 5px;
            color: {secondary.name()};
            /* Background behind title to cover the border */
            background-color: {background.name()};
        }}
        QPushButton {{
            border: 2px solid {border_color}; /* Increased border width */
            border-radius: 8px; /* Increased rounding */
            padding: 8px 10px;
            background-color: {button_bg.name()};
            min-width: 80px;
        }}
        QPushButton:hover {{
            background-color: {button_hover_bg.name()};
            border: 2px solid {border_hover_color}; /* Use hover border color */
        }}
        QPushButton:pressed {{
            background-color: {button_pressed_bg.name()};
        }}
        QPushButton:disabled {{
            background-color: {button_disabled_bg.name()};
            color: {disabled.name()};
            border: 2px solid {disabled.darker(110).name()};
        }}
        QTextEdit {{
            background-color: {background_base.name()};
            border: 2px solid {border_color}; /* Increased border width */
            border-radius: 8px; /* Added rounding */
            color: {foreground.name()};
            padding: 4px; /* Add some internal padding */
        }}
        QLabel {{
            border: none;
            padding: 2px;
        }}
        ImageLabel {{ /* Keep specific border for the image */
             border: 2px solid {border_color}; /* Increased border width */
             border-radius: 8px; /* Added rounding */
        }}
        QVideoWidget {{
            background-color: {background_base.name()};
            border: 2px solid {border_color}; /* Add border */
            border-radius: 8px; /* Add rounding */
        }}
        /* Style scrollbars */
        QScrollBar:vertical {{
            border: 1px solid {border_color};
            background: {background.name()};
            width: 18px; /* Slightly wider */
            margin: 18px 0 18px 0; /* Adjust margin for thicker buttons */
            border-radius: 9px;
        }}
        QScrollBar::handle:vertical {{
            background: {disabled.name()};
            min-height: 25px;
            border-radius: 8px; /* Rounded handle */
            margin: 1px; /* Add slight margin around handle */
        }}
         QScrollBar::handle:vertical:hover {{
            background: {disabled.lighter(120).name()};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: 2px solid {border_color}; /* Match button border */
            background: {button_bg.name()};
            height: 16px; /* Adjust height */
            border-radius: 8px; /* Match button rounding */
            subcontrol-position: top;
            subcontrol-origin: margin;
        }}
         QScrollBar::add-line:vertical {{
            subcontrol-position: bottom;
         }}
        QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{
             border: none;
             width: 6px; /* Adjust size */
             height: 6px; /* Adjust size */
             background: {foreground.name()};
             border-radius: 3px; /* Make arrows round */
             margin: 2px; /* Center arrows */
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
             background: none;
        }}
    """)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # --- Set Global Font ---
    default_font = QtGui.QFont()
    default_font.setPointSize(11) # Increase point size (adjust as needed)
    default_font.setBold(True)    # Make it bold
    app.setFont(default_font)
    # --- End Set Global Font ---

    set_dark_theme(app)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
