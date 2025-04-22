import sys
import os
import subprocess
import cv2
from PyQt6 import QtWidgets, QtGui, QtCore, QtMultimedia, QtMultimediaWidgets
import itertools  # For color cycling

# --- Configuration ---
TRAFBOARD_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "trafboard.py")
PGIE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_inferyolov8.txt")
TRACKER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_tracker.txt")
PYTHON_EXECUTABLE = sys.executable
OUTPUT_VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_VIDEO_DIR, "output.mp4")

# Theme colors (for preview drawing)
PREVIEW_COLORS = itertools.cycle([
    QtGui.QColor("#81a1c1"),  # Primary blue
    QtGui.QColor("#a3be8c"),  # Green
    QtGui.QColor("#ebcb8b"),  # Yellow
    QtGui.QColor("#d08770"),  # Orange
    QtGui.QColor("#b48ead"),  # Purple
])
COLOR_ROI_PREVIEW = QtGui.QColor("#88c0d0")  # Secondary cyan
PREVIEW_LINE_THICKNESS = 2


class ImageLabel(QtWidgets.QLabel):
    """ Custom QLabel to handle mouse events for drawing lines and ROIs """
    line_drawn = QtCore.pyqtSignal(int, int, int, int, int)  # id, x1, y, x2, y
    roi_drawn = QtCore.pyqtSignal(int, int, int, int)  # x1, y1, x2, y2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_point = None
        self.end_point = None
        self.current_lines = {}  # Store lines as {id: (x1, y, x2, y)}
        self.current_roi_coords = None
        self.original_pixmap = None
        self.drawing = False
        self.draw_mode = 'roi'  # Default mode: 'roi' or 'line'
        self.current_line_id = 1  # ID for the next line to be drawn

    def setPixmap(self, pixmap: QtGui.QPixmap | None):
        self.original_pixmap = pixmap.copy() if pixmap else None
        super().setPixmap(pixmap)

    def set_draw_mode(self, mode, line_id=None):
        if mode in ['line', 'roi']:
            self.draw_mode = mode
            if mode == 'line':
                self.current_line_id = line_id if line_id is not None else 1
            self.drawing = False
            self.start_point = None
            self.end_point = None
            self.update()

    def clear_line(self, line_id):
        if line_id in self.current_lines:
            del self.current_lines[line_id]
            self.update()

    def clear_all_lines(self):
        self.current_lines.clear()
        self.update()

    def clear_roi(self):
        self.current_roi_coords = None
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if self.original_pixmap and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.drawing = True
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
            sx = max(0, min(self.start_point.x(), pixmap_width - 1))
            sy = max(0, min(self.start_point.y(), pixmap_height - 1))
            ex = max(0, min(self.end_point.x(), pixmap_width - 1))
            ey = max(0, min(self.end_point.y(), pixmap_height - 1))

            if self.draw_mode == 'line':
                y = sy
                x1 = min(sx, ex)
                x2 = max(sx, ex)
                line_coords = (x1, y, x2, y)
                self.current_lines[self.current_line_id] = line_coords
                self.line_drawn.emit(self.current_line_id, *line_coords)
            else:  # roi mode
                x1 = min(sx, ex)
                y1 = min(sy, ey)
                x2 = max(sx, ex)
                y2 = max(sy, ey)
                if x1 < x2 and y1 < y2:
                    self.current_roi_coords = (x1, y1, x2, y2)
                    self.roi_drawn.emit(*self.current_roi_coords)
                else:
                    self.current_roi_coords = None

            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        if not self.original_pixmap:
            super().paintEvent(event)
            return

        painter = QtGui.QPainter(self)
        painter.drawPixmap(self.rect(), self.original_pixmap)

        if self.current_roi_coords:
            pen = QtGui.QPen(COLOR_ROI_PREVIEW, PREVIEW_LINE_THICKNESS, QtCore.Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(QtCore.QRectF(QtCore.QPointF(self.current_roi_coords[0], self.current_roi_coords[1]),
                                           QtCore.QPointF(self.current_roi_coords[2], self.current_roi_coords[3])))

        color_map = {}
        sorted_line_ids = sorted(self.current_lines.keys())
        for line_id in sorted_line_ids:
            coords = self.current_lines[line_id]
            if line_id not in color_map:
                color_map[line_id] = next(PREVIEW_COLORS)
            pen = QtGui.QPen(color_map[line_id], PREVIEW_LINE_THICKNESS, QtCore.Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawLine(*coords)
            painter.drawText(coords[0] - 20, coords[1] + 5, str(line_id))

        if self.drawing and self.start_point and self.end_point:
            if self.draw_mode == 'line':
                current_color = color_map.get(self.current_line_id, next(PREVIEW_COLORS))
                pen = QtGui.QPen(current_color, PREVIEW_LINE_THICKNESS, QtCore.Qt.PenStyle.DotLine)
                painter.setPen(pen)
                y = self.start_point.y()
                x1 = min(self.start_point.x(), self.end_point.x())
                x2 = max(self.start_point.x(), self.end_point.x())
                painter.drawLine(x1, y, x2, y)
            else:
                pen = QtGui.QPen(COLOR_ROI_PREVIEW, PREVIEW_LINE_THICKNESS, QtCore.Qt.PenStyle.DotLine)
                painter.setPen(pen)
                start_point_f = QtCore.QPointF(self.start_point)
                end_point_f = QtCore.QPointF(self.end_point)
                painter.drawRect(QtCore.QRectF(start_point_f, end_point_f).normalized())

        painter.end()


class ProcessWorker(QtCore.QThread):
    log_message = QtCore.pyqtSignal(str)
    process_finished = QtCore.pyqtSignal(int)

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            for line in process.stdout:
                self.log_message.emit(line.strip())

            process.wait()
            self.process_finished.emit(process.returncode)

        except Exception as e:
            self.log_message.emit(f"Error starting/reading process: {e}")
            self.process_finished.emit(-1)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Line Definition for DeepStream (Dark)")
        self.setGeometry(100, 100, 1300, 800)

        self.video_path = None
        self.original_frame = None
        self.original_width = 0
        self.original_height = 0
        self.display_width = 800
        self.display_height = 0
        self.scale_x_display = 1.0
        self.scale_y_display = 1.0
        self.lines_original = {}
        self.roi_coords_original = None
        self.output_video_path = None
        self.processing_thread = None

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QGridLayout(self.central_widget)

        self.btn_select_video = QtWidgets.QPushButton("Select Video")
        self.lbl_video_path = QtWidgets.QLabel("No video selected")
        self.layout.addWidget(self.btn_select_video, 0, 0)
        self.layout.addWidget(self.lbl_video_path, 0, 1, 1, 2)

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

        self.controls_container = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QVBoxLayout(self.controls_container)

        self.line_mgmt_group = QtWidgets.QGroupBox("Lines")
        self.line_mgmt_layout = QtWidgets.QVBoxLayout()
        self.line_id_layout = QtWidgets.QHBoxLayout()
        self.lbl_line_id = QtWidgets.QLabel("Line ID:")
        self.spin_line_id = QtWidgets.QSpinBox()
        self.spin_line_id.setMinimum(1)
        self.spin_line_id.setMaximum(99)
        self.btn_draw_line = QtWidgets.QPushButton("Draw/Redraw Line")
        self.line_id_layout.addWidget(self.lbl_line_id)
        self.line_id_layout.addWidget(self.spin_line_id)
        self.line_id_layout.addWidget(self.btn_draw_line)

        self.list_lines = QtWidgets.QListWidget()
        self.list_lines.setFixedHeight(100)

        self.line_clear_layout = QtWidgets.QHBoxLayout()
        self.btn_clear_line = QtWidgets.QPushButton("Clear Selected Line")
        self.btn_clear_all_lines = QtWidgets.QPushButton("Clear All Lines")
        self.line_clear_layout.addWidget(self.btn_clear_line)
        self.line_clear_layout.addWidget(self.btn_clear_all_lines)

        self.line_mgmt_layout.addLayout(self.line_id_layout)
        self.line_mgmt_layout.addWidget(self.list_lines)
        self.line_mgmt_layout.addLayout(self.line_clear_layout)
        self.line_mgmt_group.setLayout(self.line_mgmt_layout)

        self.roi_mgmt_group = QtWidgets.QGroupBox("Region of Interest (ROI)")
        self.roi_mgmt_layout = QtWidgets.QVBoxLayout()
        self.btn_draw_roi = QtWidgets.QPushButton("Draw/Redraw ROI")
        self.lbl_roi_coords = QtWidgets.QLabel("ROI Coords (Original): None")
        self.btn_clear_roi = QtWidgets.QPushButton("Clear ROI")
        self.roi_mgmt_layout.addWidget(self.btn_draw_roi)
        self.roi_mgmt_layout.addWidget(self.lbl_roi_coords)
        self.roi_mgmt_layout.addWidget(self.btn_clear_roi)
        self.roi_mgmt_group.setLayout(self.roi_mgmt_layout)

        self.btn_process = QtWidgets.QPushButton("Process Video")
        self.btn_process.setEnabled(False)

        self.controls_layout.addWidget(self.line_mgmt_group)
        self.controls_layout.addWidget(self.roi_mgmt_group)
        self.controls_layout.addWidget(self.btn_process)
        self.controls_layout.addStretch()

        self.controls_scroll_area = QtWidgets.QScrollArea()
        self.controls_scroll_area.setWidgetResizable(True)
        self.controls_scroll_area.setWidget(self.controls_container)
        self.layout.addWidget(self.controls_scroll_area, 2, 0)

        self.log_group = QtWidgets.QGroupBox("Logs")
        self.log_layout = QtWidgets.QVBoxLayout()
        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        log_font = QtGui.QFont()
        log_font.setPointSize(10)
        log_font.setBold(False)
        self.log_output.setFont(log_font)
        self.log_layout.addWidget(self.log_output)
        self.log_group.setLayout(self.log_layout)
        self.layout.addWidget(self.log_group, 2, 1, 1, 2)

        self.lbl_status = QtWidgets.QLabel("")
        self.layout.addWidget(self.lbl_status, 3, 0, 1, 3)

        self.layout.setColumnStretch(0, 2)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 1)
        self.layout.setRowStretch(1, 4)
        self.layout.setRowStretch(2, 2)

        self.btn_select_video.clicked.connect(self.select_video)
        self.image_label.line_drawn.connect(self.update_line_coords)
        self.image_label.roi_drawn.connect(self.update_roi_coords)
        self.btn_process.clicked.connect(self.process_video)
        self.btn_draw_line.clicked.connect(self.set_draw_line_mode)
        self.btn_draw_roi.clicked.connect(lambda: self.image_label.set_draw_mode('roi'))
        self.btn_clear_line.clicked.connect(self.clear_selected_line)
        self.btn_clear_all_lines.clicked.connect(self.clear_all_lines)
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        self.btn_play.clicked.connect(self.media_player.play)
        self.btn_pause.clicked.connect(self.media_player.pause)
        self.btn_stop.clicked.connect(self.media_player.stop)
        self.media_player.playbackStateChanged.connect(self.update_video_buttons)
        self.media_player.errorOccurred.connect(self.media_player_error)

        self.update_video_buttons()

    def select_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                             "Video Files (*.mp4 *.avi *.mov *.mkv *.h264);;All Files (*)")
        if file_path:
            self.video_path = file_path
            self.lbl_video_path.setText(os.path.basename(file_path))
            self.log_output.clear()
            self.lines_original.clear()
            self.roi_coords_original = None
            self.list_lines.clear()
            self.lbl_roi_coords.setText("ROI Coords (Original): None")
            self.image_label.clear_all_lines()
            self.image_label.clear_roi()
            self.image_label.set_draw_mode('roi')
            self.btn_process.setEnabled(False)
            self.load_first_frame()
            self.media_player.stop()
            self.media_player.setSource(QtCore.QUrl())
            self.output_video_path = None
            self.update_video_buttons()
            self.lbl_status.setText("")

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
                self.original_frame = frame
                if self.original_width <= 0 or self.original_height <= 0:
                    self.show_error("Invalid video dimensions.")
                    return

                self.scale_y_display = self.display_width / self.original_width
                self.display_height = int(self.original_height * self.scale_y_display)
                self.scale_x_display = self.scale_y_display

                display_frame = cv2.resize(self.original_frame, (self.display_width, self.display_height))

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
                self.image_label.setFixedSize(self.display_width, 200)

        except Exception as e:
            self.show_error(f"Error loading frame: {e}")
            self.image_label.clear()
            self.image_label.setFixedSize(self.display_width, 200)

    def set_draw_line_mode(self):
        line_id = self.spin_line_id.value()
        self.image_label.set_draw_mode('line', line_id)

    def update_line_coords(self, line_id, x1_disp, y_disp, x2_disp, y_disp2):
        if self.scale_x_display > 0 and self.scale_y_display > 0:
            x1_orig = int(x1_disp / self.scale_x_display)
            y_orig = int(y_disp / self.scale_y_display)
            x2_orig = int(x2_disp / self.scale_x_display)
            x1_orig = max(0, min(x1_orig, self.original_width - 1))
            y_orig = max(0, min(y_orig, self.original_height - 1))
            x2_orig = max(x1_orig, min(x2_orig, self.original_width - 1))

            line_coords_orig = (x1_orig, y_orig, x2_orig, y_orig)
            self.lines_original[line_id] = line_coords_orig

            self.update_line_list_widget()

            self.btn_process.setEnabled(bool(self.lines_original))
        else:
            self.show_error("Cannot calculate original line coordinates (invalid scale).")
            self.btn_process.setEnabled(bool(self.lines_original))

    def update_roi_coords(self, x1_disp, y1_disp, x2_disp, y2_disp):
        if self.scale_x_display > 0 and self.scale_y_display > 0:
            x1_orig = int(x1_disp / self.scale_x_display)
            y1_orig = int(y1_disp / self.scale_y_display)
            x2_orig = int(x2_disp / self.scale_x_display)
            y2_orig = int(y2_disp / self.scale_y_display)
            x1_orig = max(0, min(x1_orig, self.original_width - 1))
            y1_orig = max(0, min(y1_orig, self.original_height - 1))
            x2_orig = max(x1_orig, min(x2_orig, self.original_width - 1))
            y2_orig = max(y1_orig, min(y2_orig, self.original_height - 1))

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

    def update_line_list_widget(self):
        self.list_lines.clear()
        for line_id in sorted(self.lines_original.keys()):
            coords = self.lines_original[line_id]
            self.list_lines.addItem(f"Line {line_id}: {coords}")

    def clear_selected_line(self):
        selected_items = self.list_lines.selectedItems()
        if not selected_items:
            return
        item_text = selected_items[0].text()
        try:
            line_id = int(item_text.split(":")[0].split(" ")[1])
            if line_id in self.lines_original:
                del self.lines_original[line_id]
                self.image_label.clear_line(line_id)
                self.update_line_list_widget()
                self.btn_process.setEnabled(bool(self.lines_original))
        except (IndexError, ValueError):
            self.show_error("Could not parse line ID from selected item.")

    def clear_all_lines(self):
        self.lines_original.clear()
        self.image_label.clear_all_lines()
        self.update_line_list_widget()
        self.btn_process.setEnabled(False)

    def clear_roi(self):
        self.roi_coords_original = None
        self.image_label.clear_roi()
        self.lbl_roi_coords.setText("ROI Coords (Original): None")

    def _validate_and_get_roi(self):
        return self.roi_coords_original

    def process_video(self):
        if not self.video_path or not self.lines_original:
            self.show_error("Please select a video and draw at least one line first.")
            return
        if self.processing_thread and self.processing_thread.isRunning():
            self.show_error("Processing is already in progress.")
            return

        current_roi = self._validate_and_get_roi()

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
        ]

        if self.lines_original:
            cmd.append("--lines")
            for line_id in sorted(self.lines_original.keys()):
                coords = self.lines_original[line_id]
                cmd.extend([
                    str(line_id),
                    str(coords[0]),
                    str(coords[1]),
                    str(coords[2]),
                    str(coords[3])
                ])

        if current_roi:
            cmd.extend([
                "--roi",
                str(current_roi[0]), str(current_roi[1]),
                str(current_roi[2]), str(current_roi[3])
            ])

        self.log_output.clear()
        self.log_output.append("Starting processing...")
        self.log_output.append(f"Command: {' '.join(cmd)}")
        self.btn_process.setEnabled(False)
        self.lbl_status.setText("Status: Processing...")

        self.media_player.stop()
        self.media_player.setSource(QtCore.QUrl())
        self.update_video_buttons()

        self.processing_thread = ProcessWorker(cmd)
        self.processing_thread.log_message.connect(self.append_log)
        self.processing_thread.process_finished.connect(self.handle_process_finished)
        self.processing_thread.start()

    @QtCore.pyqtSlot(str)
    def append_log(self, message):
        self.log_output.append(message)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    @QtCore.pyqtSlot(int)
    def handle_process_finished(self, return_code):
        self.lbl_status.setText(f"Status: Processing finished (Code: {return_code})")
        self.btn_process.setEnabled(bool(self.lines_original))

        if return_code == 0:
            self.log_output.append(f"\nProcessing finished successfully. Output saved to: {self.output_video_path}")
            if os.path.exists(self.output_video_path):
                self.media_player.setSource(QtCore.QUrl.fromLocalFile(self.output_video_path))
                self.log_output.append("Output video loaded. Press Play.")
                self.update_video_buttons()
            else:
                self.log_output.append("Output video file not found after processing.")
        else:
            self.log_output.append(f"\nProcessing failed with return code {return_code}.")

        self.processing_thread = None

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

    background = QtGui.QColor("#1f2229")
    background_base = QtGui.QColor("#000000")
    foreground = QtGui.QColor("#dfdfdf")
    primary = QtGui.QColor("#81a1c1")
    secondary = QtGui.QColor("#88c0d0")
    alert = QtGui.QColor("#bf616a")
    disabled = QtGui.QColor("#707880")
    orange = QtGui.QColor("#FFA500")

    dark_palette = QtGui.QPalette()

    dark_palette.setColor(QtGui.QPalette.ColorRole.Window, background)
    dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, foreground)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Base, background_base)
    dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, background)
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, background)
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, foreground)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Text, foreground)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Button, background)
    dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, foreground)
    dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, alert)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Link, primary)

    dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, primary)
    dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, background_base)

    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text, disabled)
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText, disabled)
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.WindowText, disabled)
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Highlight, background.darker(120))
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.HighlightedText, disabled)

    app.setPalette(dark_palette)

    button_bg = background.lighter(115)
    button_hover_bg = background.lighter(130)
    button_pressed_bg = background.lighter(105)
    button_disabled_bg = background.darker(110)
    border_color = secondary.darker(120).name()
    border_hover_color = secondary.name()

    app.setStyleSheet(f"""
        QWidget {{
            color: {foreground.name()};
            background-color: {background.name()};
        }}
        QGroupBox {{
            border: 2px solid {border_color};
            border-radius: 8px;
            margin-top: 1em;
            margin-bottom: 5px;
            margin-left: 5px;
            margin-right: 5px;
            padding: 15px 8px 8px 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px 0 5px;
            color: {secondary.name()};
            background-color: {background.name()};
        }}
        QPushButton {{
            border: 2px solid {border_color};
            border-radius: 8px;
            padding: 8px 10px;
            background-color: {button_bg.name()};
            min-width: 80px;
        }}
        QPushButton:hover {{
            background-color: {button_hover_bg.name()};
            border: 2px solid {border_hover_color};
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
            border: 2px solid {border_color};
            border-radius: 8px;
            color: {foreground.name()};
            padding: 4px;
        }}
        QLabel {{
            border: none;
            padding: 2px;
        }}
        ImageLabel {{
             border: 2px solid {border_color};
             border-radius: 8px;
        }}
        QVideoWidget {{
            background-color: {background_base.name()};
            border: 2px solid {border_color};
            border-radius: 8px;
        }}
        QScrollArea {{
            border: 2px solid {border_color};
            border-radius: 8px;
        }}
        QScrollBar:vertical {{
            border: 1px solid {border_color};
            background: {background.name()};
            width: 18px;
            margin: 18px 0 18px 0;
            border-radius: 9px;
        }}
        QScrollBar::handle:vertical {{
            background: {disabled.name()};
            min-height: 25px;
            border-radius: 8px;
            margin: 1px;
        }}
         QScrollBar::handle:vertical:hover {{
            background: {disabled.lighter(120).name()};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: 2px solid {border_color};
            background: {button_bg.name()};
            height: 16px;
            border-radius: 8px;
            subcontrol-position: top;
            subcontrol-origin: margin;
        }}
         QScrollBar::add-line:vertical {{
            subcontrol-position: bottom;
         }}
        QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{
             border: none;
             width: 6px;
             height: 6px;
             background: {foreground.name()};
             border-radius: 3px;
             margin: 2px;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
             background: none;
        }}
    """)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    default_font = QtGui.QFont()
    default_font.setPointSize(11)
    default_font.setBold(True)
    app.setFont(default_font)

    set_dark_theme(app)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
