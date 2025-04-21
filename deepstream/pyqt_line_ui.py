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

class ImageLabel(QtWidgets.QLabel):
    """ Custom QLabel to handle mouse events for drawing lines (PyQt6) """
    line_drawn = QtCore.pyqtSignal(int, int, int, int) # x1, y, x2, y

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_point = None
        self.end_point = None
        self.current_line_coords = None # Store the final line
        self.original_pixmap = None # Store the clean pixmap without temporary lines
        self.drawing = False

    def setPixmap(self, pixmap: QtGui.QPixmap | None):
        self.original_pixmap = pixmap.copy() if pixmap else None
        super().setPixmap(pixmap)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if self.original_pixmap and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos() # Initialize end point
            self.drawing = True
            # Clear previous final line visually for redraw
            self.current_line_coords = None
            self.update() # Trigger paintEvent

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self.original_pixmap and self.drawing:
            self.end_point = event.pos()
            self.update() # Trigger paintEvent to draw temporary line

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if self.original_pixmap and self.drawing and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.drawing = False
            self.end_point = event.pos()

            # Ensure line is horizontal (use start_point's y)
            y = self.start_point.y()
            x1 = min(self.start_point.x(), self.end_point.x())
            x2 = max(self.start_point.x(), self.end_point.x())

            # Clamp coordinates to pixmap bounds
            pixmap_width = self.original_pixmap.width() if self.original_pixmap else 0
            pixmap_height = self.original_pixmap.height() if self.original_pixmap else 0
            x1 = max(0, min(x1, pixmap_width - 1))
            y = max(0, min(y, pixmap_height - 1))
            x2 = max(x1, min(x2, pixmap_width - 1)) # Ensure x2 >= x1

            self.current_line_coords = (x1, y, x2, y)
            self.line_drawn.emit(*self.current_line_coords)
            self.update() # Trigger paintEvent to draw final line

    def paintEvent(self, event: QtGui.QPaintEvent):
        if not self.original_pixmap:
            super().paintEvent(event)
            return

        # Draw the original pixmap first
        painter = QtGui.QPainter(self)
        painter.drawPixmap(self.rect(), self.original_pixmap)

        # Draw the current line (temporary during drag or final after release)
        line_to_draw = None
        if self.drawing and self.start_point and self.end_point:
            # Draw temporary horizontal line during drag
            y = self.start_point.y()
            x1 = min(self.start_point.x(), self.end_point.x())
            x2 = max(self.start_point.x(), self.end_point.x())
            line_to_draw = (x1, y, x2, y)
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.blue, 2, QtCore.Qt.PenStyle.DotLine) # Dotted blue line while drawing
        elif self.current_line_coords:
            # Draw final solid line after release
            line_to_draw = self.current_line_coords
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 2, QtCore.Qt.PenStyle.SolidLine) # Solid red line when finished

        if line_to_draw:
            painter.setPen(pen)
            painter.drawLine(line_to_draw[0], line_to_draw[1], line_to_draw[2], line_to_draw[3])

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
        self.setGeometry(100, 100, 1200, 800) # Adjusted size

        self.video_path = None
        self.original_frame = None
        self.original_width = 0
        self.original_height = 0
        self.display_width = 800 # Increased display width
        self.display_height = 0
        self.scale_x_display = 1.0
        self.scale_y_display = 1.0
        self.line_coords_original = None # Line coordinates relative to original frame
        self.output_video_path = None # Store path for video player
        self.processing_thread = None # To hold the worker thread

        # --- Widgets ---
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QGridLayout(self.central_widget)

        # Top row: File selection
        self.btn_select_video = QtWidgets.QPushButton("Select Video")
        self.lbl_video_path = QtWidgets.QLabel("No video selected")
        self.layout.addWidget(self.btn_select_video, 0, 0)
        self.layout.addWidget(self.lbl_video_path, 0, 1, 1, 2) # Span 2 columns

        # Middle row: Image display and controls
        self.image_label = ImageLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;") # Adjusted border color
        self.layout.addWidget(self.image_label, 1, 0, 2, 1) # Span 2 rows

        # Controls GroupBox
        self.controls_group = QtWidgets.QGroupBox("Controls")
        self.controls_layout = QtWidgets.QVBoxLayout()
        self.lbl_line_coords = QtWidgets.QLabel("Line Coords (Original): None")
        self.btn_process = QtWidgets.QPushButton("Process Video")
        self.btn_process.setEnabled(False)
        self.controls_layout.addWidget(self.lbl_line_coords)
        self.controls_layout.addWidget(self.btn_process)
        self.controls_layout.addStretch()
        self.controls_group.setLayout(self.controls_layout)
        self.layout.addWidget(self.controls_group, 1, 1) # Next to image

        # Video Player GroupBox
        self.video_group = QtWidgets.QGroupBox("Output Video")
        self.video_layout = QtWidgets.QVBoxLayout()
        self.video_widget = QtMultimediaWidgets.QVideoWidget()
        self.media_player = QtMultimedia.QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        # Add Play/Pause/Stop buttons
        self.video_control_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.video_control_layout.addWidget(self.btn_play)
        self.video_control_layout.addWidget(self.btn_pause)
        self.video_control_layout.addWidget(self.btn_stop)
        self.video_layout.addWidget(self.video_widget, 1) # Allow video to stretch
        self.video_layout.addLayout(self.video_control_layout)
        self.video_group.setLayout(self.video_layout)
        self.layout.addWidget(self.video_group, 1, 2) # Right column

        # Log Output GroupBox
        self.log_group = QtWidgets.QGroupBox("Logs")
        self.log_layout = QtWidgets.QVBoxLayout()
        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_layout.addWidget(self.log_output)
        self.log_group.setLayout(self.log_layout)
        self.layout.addWidget(self.log_group, 2, 1, 1, 2) # Bottom right, span 2 columns

        # Add a status label
        self.lbl_status = QtWidgets.QLabel("")
        self.layout.addWidget(self.lbl_status, 3, 0, 1, 3) # Bottom row, span all columns

        # Set column stretch factors
        self.layout.setColumnStretch(0, 3) # Image column wider
        self.layout.setColumnStretch(1, 1) # Controls column
        self.layout.setColumnStretch(2, 2) # Video column
        # Set row stretch factors
        self.layout.setRowStretch(1, 3) # Middle row (image/video) taller
        self.layout.setRowStretch(2, 1) # Log row

        # --- Connections ---
        self.btn_select_video.clicked.connect(self.select_video)
        self.image_label.line_drawn.connect(self.update_line_coords)
        self.btn_process.clicked.connect(self.process_video)
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
            self.lbl_line_coords.setText("Line Coords (Original): None")
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
            x2_orig = max(x1_orig, min(x2_orig, self.original_width - 1)) # Ensure x2 >= x1

            self.line_coords_original = (x1_orig, y_orig, x2_orig, y_orig) # Store as x1,y1,x2,y2
            self.lbl_line_coords.setText(f"Line Coords (Original): {self.line_coords_original}")
            self.btn_process.setEnabled(True)
        else:
            self.show_error("Cannot calculate original coordinates (invalid scale).")
            self.line_coords_original = None
            self.btn_process.setEnabled(False)


    def process_video(self):
        if not self.video_path or not self.line_coords_original:
            self.show_error("Please select a video and draw a line first.")
            return
        if self.processing_thread and self.processing_thread.isRunning():
            self.show_error("Processing is already in progress.")
            return

        # Ensure output directory exists
        try:
            os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
        except OSError as e:
            self.show_error(f"Failed to create output directory {OUTPUT_VIDEO_DIR}: {e}")
            return

        self.output_video_path = OUTPUT_VIDEO_PATH # Use fixed path

        cmd = [
            PYTHON_EXECUTABLE,
            TRAFBOARD_SCRIPT_PATH,
            "--pgie-config", PGIE_CONFIG_PATH,
            "--tracker-config", TRACKER_CONFIG_PATH,
            self.video_path,
            self.output_video_path, # Use fixed output path
            "--line",
            str(self.line_coords_original[0]), # x1
            str(self.line_coords_original[1]), # y1
            str(self.line_coords_original[2]), # x2
            str(self.line_coords_original[3])  # y2 (same as y1)
        ]

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

    dark_palette = QtGui.QPalette()

    dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(35, 35, 35))
    dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))

    dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black)

    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text, QtGui.QColor(127, 127, 127))
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(127, 127, 127))
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(127, 127, 127))
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(80, 80, 80))
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(127, 127, 127))

    app.setPalette(dark_palette)

    app.setStyleSheet("""
        QWidget {
            color: white;
            background-color: rgb(53, 53, 53);
        }
        QGroupBox {
            border: 1px solid gray;
            border-radius: 5px;
            margin-top: 0.5em;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QPushButton {
            border: 1px solid gray;
            border-radius: 4px;
            padding: 5px;
            background-color: rgb(70, 70, 70);
        }
        QPushButton:hover {
            background-color: rgb(85, 85, 85);
        }
        QPushButton:pressed {
            background-color: rgb(60, 60, 60);
        }
        QPushButton:disabled {
            background-color: rgb(45, 45, 45);
            color: gray;
        }
        QTextEdit {
            background-color: rgb(35, 35, 35);
            border: 1px solid gray;
        }
        QLabel {
            border: none;
        }
        QVideoWidget {
            background-color: black;
        }
    """)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    set_dark_theme(app)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
