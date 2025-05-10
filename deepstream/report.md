# Traffic Analytics System Report

## 1. Abstract

This report details a comprehensive traffic analytics system designed for monitoring and analyzing road traffic using computer vision and deep learning techniques. The system leverages NVIDIA DeepStream for high-performance video inference, ROS2 for modular communication, and PyQt6 for a user-friendly graphical interface. It provides real-time insights into vehicle movements, including counts, speeds, lane usage, and compliance with designated lane rules. The primary goal is to offer a robust and extensible platform for traffic management, planning, and safety analysis.

## 2. Motivation and Need

Efficient traffic management is crucial for modern urban environments. Congestion, safety hazards, and inefficient road usage lead to economic losses, environmental pollution, and reduced quality of life. Traditional methods like manual counting or inductive loops are often expensive, inflexible, or lack the granularity needed for detailed analysis.

A computer vision-based system offers several advantages:
- **Cost-Effectiveness**: Utilizes existing camera infrastructure or relatively inexpensive cameras.
- **Rich Data**: Provides detailed information beyond simple counts, including vehicle types, speeds, trajectories, and interactions.
- **Flexibility**: Easily adaptable to different road layouts and monitoring requirements through software configuration.
- **Scalability**: Can be deployed across multiple locations and integrated into larger traffic management systems.

This project addresses the need for an accessible, high-performance, and feature-rich traffic analytics tool that can be easily configured and deployed for various traffic monitoring scenarios. The integration of a user-friendly GUI further lowers the barrier to entry for users who may not be experts in command-line tools or DeepStream configuration.

## 3. Technology Stack

The system is built upon a combination of specialized technologies chosen for performance, flexibility, and ease of development:

- **NVIDIA DeepStream SDK (6.2+)**: The core engine for video processing. It provides a streaming analytics toolkit optimized for NVIDIA GPUs. Key features utilized include hardware-accelerated decoding/encoding, TensorRT-optimized inference, multi-object tracking, and efficient batch processing of video streams. Its GStreamer integration allows for building complex, high-throughput pipelines.
- **ROS2 (Humble Hawksbill)**: A widely adopted middleware for robotics and distributed systems. In this project, it primarily provides the node structure (`rclpy.node.Node`) for the DeepStream pipeline classes, enabling standardized logging and lifecycle management. While direct ROS2 topic communication is minimal in the current implementation (a basic publisher exists but isn't central to the analytics), the ROS2 structure facilitates future integration with larger robotic systems or distributed sensor networks.
- **Python 3 (3.8+)**: The primary programming language. Python's extensive libraries and ease of use make it ideal for orchestrating the DeepStream pipeline, implementing custom analytics logic, developing the GUI, and handling data processing. Bindings like PyGObject (for GStreamer/DeepStream) and PyQt6 enable seamless interaction with the underlying C/C++ libraries.
- **GStreamer**: A powerful pipeline-based multimedia framework. DeepStream is built on top of GStreamer. Understanding GStreamer concepts (elements, pads, linking, probes, bus messages) is essential for customizing and debugging the video processing pipeline.
- **YOLOv8 (You Only Look Once)**: The chosen object detection model (`yolov8m` variant). It offers an excellent balance between real-time performance and detection accuracy for common road objects. Its integration involves converting the model to ONNX format and then optimizing it with TensorRT for deployment within DeepStream's `nvinfer` element.
- **TensorRT**: NVIDIA's SDK for high-performance deep learning inference. It optimizes trained models for deployment on NVIDIA GPUs, significantly reducing latency and increasing throughput compared to running inference directly within frameworks like PyTorch or TensorFlow. DeepStream leverages TensorRT heavily.
- **PyQt6**: A comprehensive set of Python bindings for the Qt cross-platform application framework. It is used to build the entire graphical user interface, providing widgets for video display, user input (drawing lines/ROI, selecting files), configuration, log display, and embedding Matplotlib charts.
- **Matplotlib**: The standard Python library for creating static, animated, and interactive visualizations. It is integrated into the PyQt6 GUI via the `FigureCanvasQTAgg` backend to display the various statistical charts (line, pie, bar) in the "Statistics" tab.
- **OpenCV (cv2)**: A fundamental library for computer vision tasks. In this project, its primary role is within the PyQt GUI (`pyqt_line_ui.py`) to read the first frame of the selected video file. This frame is then displayed in the `ImageLabel` widget, allowing the user to draw lines and ROIs accurately based on the video's content and perspective. It also handles the necessary color space conversions (BGR to RGB) for display in Qt.
- **NumPy**: Used for numerical operations, particularly within the statistics calculation and plotting components (e.g., creating arrays for Matplotlib charts).
- **JSON**: The format used for saving the structured traffic statistics generated by the `TrafficStats` class. Its human-readable and machine-parseable nature makes it suitable for data storage and interchange.

## 4. System Architecture and Core Classes

The system employs a modular design centered around the DeepStream GStreamer pipeline, controlled and extended by Python classes. A PyQt6 application provides the user interface for configuration and visualization.

### Pipeline Class Hierarchy

The core video processing logic is encapsulated within a hierarchy of Python classes that manage the GStreamer pipeline construction and execution.

1.  **`NodeFileSinkPipeline` (`deepstream_class.py`)**:
    *   **Role**: This class serves as the foundation for pipelines that process video from an input file and save the annotated output to another file, while also integrating basic ROS2 node capabilities (like logging).
    *   **Inheritance**: Inherits from `rclpy.node.Node`.
    *   **Functionality**:
        *   Initializes GStreamer (`Gst.init`).
        *   Creates and configures standard DeepStream GStreamer elements for a file-to-file workflow: `filesrc` -> `qtdemux` -> parser (`h264parse`) -> `nvv4l2decoder` -> `nvstreammux` -> `nvinfer` (PGIE) -> `nvtracker` -> `nvvideoconvert` -> `nvdsosd` -> `nvvideoconvert` -> encoder (`nvv4l2h264enc`) -> parser (`h264parse`) -> muxer (`qtmux`) -> `filesink`.
        *   Handles the dynamic linking required by `qtdemux` via the `on_pad_added` callback method. This method ensures that once the demuxer identifies the video stream type (e.g., H.264), it correctly links its output pad to the appropriate parser's input pad, establishing the data flow.
        *   Reads tracker configuration from a file and applies it to the `nvtracker` element.
        *   Sets properties for various elements like `nvstreammux` dimensions, batch size, `nvinfer` config file path, encoder bitrate, and `filesink` location.
        *   Provides a `run()` method that sets the pipeline state to `PLAYING`, starts a `GLib.MainLoop`, and attaches a message handler (`bus_call`) to the pipeline's bus to listen for End-of-Stream (EOS) or ERROR messages, which trigger the loop to quit. It ensures the pipeline state is set back to `NULL` upon completion or error.
        *   Includes a basic ROS2 publisher (`self.publisher_`) for potential future use, though not central to the current file-saving pipeline's core function.

2.  **`TrafboardPipeline` (`trafboard.py`)**:
    *   **Role**: This class specializes the `NodeFileSinkPipeline` specifically for the traffic analytics task. It inherits the basic file-to-file pipeline structure and adds the custom logic required for counting, tracking, and analysis.
    *   **Inheritance**: Inherits from `NodeFileSinkPipeline`.
    *   **Functionality**:
        *   **Initialization**: Calls the parent constructor (`super().__init__`) and adds initialization specific to traffic analysis:
            *   Parses additional command-line arguments (using `argparse` in `main`) for defining crossing lines (`--lines`), ROI (`--roi`), lane designations (`--inner-lanes`, `--outer-lanes`), and the statistics output directory (`--stats-dir`).
            *   Instantiates the `TrafficStats` class, passing the line and lane designation information.
            *   Calculates scaling factors (`scale_w`, `scale_h`) to map coordinates between the original video resolution (used in the GUI) and the pipeline's processing resolution (defined by `nvstreammux`).
        *   **Custom Probe Attachment**: The key extension point. It retrieves the sink pad of the `nvdsosd` element (`self.osdsinkpad`) and attaches the `custom_probe` method to it using `add_probe(Gst.PadProbeType.BUFFER, ...)`. This allows intercepting and analyzing the metadata associated with each batch of frames *before* it's rendered by the OSD but *after* detection and tracking have occurred.
        *   **`custom_probe` Method**: Contains the core analytics logic executed per batch: accesses metadata (`NvDsBatchMeta`, `NvDsFrameMeta`, `NvDsObjectMeta`), filters objects (by class ID using `is_vehicle_class`, by ROI), calculates centroids, updates tracking history (`past_tracking_points`), performs line crossing checks, calls methods on the `self.stats` object (`update_vehicle_crossing`, `update_vehicle_position`, `update_lane_compliance`, etc.), and adds custom visualizations (lines, ROI, trails) to the frame's display metadata (`NvDsDisplayMeta`).

This inheritance structure allows `TrafboardPipeline` to focus solely on the analytics logic while reusing the complex GStreamer pipeline setup and execution management provided by `NodeFileSinkPipeline`.

### Detailed Class Descriptions:

1.  **`NodeFileSinkPipeline` (`deepstream_class.py`)**:
    *   **Base Class**: `rclpy.node.Node`
    *   **Description**: 
        *   Initializes GStreamer and sets up the pipeline for processing video from a file to a file sink.
        *   Handles dynamic pads for demuxers and configures various elements like stream muxer, inference, tracker, and encoders.
        *   Contains a `run()` method to start the pipeline and a ROS2 publisher (currently not used).
    *   **Key Methods**:
        *   `__init__(self, input_file, output_file, pgie_config, tracker_config, batch_size=1, width=640, height=480)`: Constructor that sets up the pipeline elements and their properties.
        *   `on_pad_added(self, src, new_pad)`: Callback to handle dynamic pad linking for the demuxer.
        *   `bus_call(self, bus, message)`: Callback to handle messages from the GStreamer bus.
        *   `run(self)`: Starts the pipeline.

2.  **`TrafboardPipeline` (`trafboard.py`)**:
    *   **Base Class**: `NodeFileSinkPipeline`
    *   **Description**: 
        *   Specialized pipeline for traffic analysis, extending the file sink pipeline with traffic-specific processing.
        *   Parses additional parameters for traffic lines, ROIs, lane designations, and statistics output.
        *   Implements custom probing for analytics.
    *   **Key Methods**Isaac Sim/Robotics Weekly Livestream:
        *   `__init__(self, input_file, output_file, pgie_config, tracker_config, lines, roi, inner_lanes, outer_lanes, stats_dir, batch_size=1, width=640, height=480)`: Constructor that initializes the base class and sets up traffic-specific parameters.
        *   `custom_probe(self, pad, info)`: Method attached to the OSD sink pad for custom analytics processing.

3.  **`TrafficStats` (`traffic_stats.py`)**:
    *   **Description**: 
        *   Manages and stores traffic statistics like counts, speeds, and lane compliance.
        *   Updates statistics based on the processed video frames and detected objects.
    *   **Key Methods**:
        *   `__init__(self, lines, inner_lanes, outer_lanes)`: Constructor that initializes the statistics storage.
        *   `update_vehicle_crossing(self, vehicle_id, crossing_id)`: Updates when a vehicle crosses a line.
        *   `update_vehicle_position(self, vehicle_id, position)`: Updates the position of a vehicle.
        *   `update_lane_compliance(self, vehicle_id, lane_id, compliant)`: Updates lane compliance status for a vehicle.

4.  **`ImageLabel` (`pyqt_line_ui.py`)**:
    *   **Description**: 
        *   Custom QLabel used in the PyQt GUI to display video frames and handle drawing of lines and ROIs.
        *   Converts between OpenCV and Qt color formats.
    *   **Key Methods**:
        *   `__init__(self, parent=None)`: Constructor.
        *   `paintEvent(self, event)`: Handles the painting event to draw lines and ROIs.
        *   `draw_line(self, start, end)`: Draws a line on the label.
        *   `draw_roi(self, points)`: Draws a region of interest.

5.  **`StatsTab` (`pyqt_stats_ui.py`)**:
    *   **Description**: 
        *   Tab in the PyQt GUI dedicated to displaying various statistics in chart form.
        *   Uses Matplotlib to render charts.
    *   **Key Methods**:
        *   `__init__(self, parent=None)`: Constructor.
        *   `plot_data(self, data)`: Plots the given data on the chart.

6.  **`MainWindow` (`pyqt_main.py`)**:
    *   **Description**: 
        *   Main window of the PyQt application.
        *   Contains instances of ImageLabel and StatsTab.
        *   Manages the layout and overall application flow.
    *   **Key Methods**:
        *   `__init__(self)`: Constructor.
        *   `setup_ui(self)`: Sets up the user interface components.
        *   `load_video(self, file)`: Loads a video file for processing.
        *   `update_statistics(self, stats)`: Updates the statistics tab with new data.

### System Architecture Diagram

```text
+---------------------+      +--------------------------------+      +---------------------+
| PyQt GUI            |<---->| User Input                     |      |                     |
| (pyqt_line_ui.py)   |      | - Select Video                 |      |                     |
| - MainWindow        |      | - Draw Lines/ROI               |      |                     |
| - ImageLabel        |      | - Set Lanes                    |      |                     |
| - StatsTab          |      | - Start Process                |      |                     |
+---------+-----------+      +--------------------------------+      +---------------------+
          | ▲                                                          | ▲
          | | Stats Data (JSON)                                        | | Processed Video
          | | Control/Config                                           | | File
          | |                                                          | |
+---------v-----------+      +--------------------------------+      +---v-----------------+
| DeepStream Pipeline |<-----| ProcessWorker (in PyQt GUI)    |----->| Output Video File   |
| (trafboard.py)      |      | (Launches trafboard.py)        |      | (videos/*.mp4)      |
| - Decode, Mux       |      +--------------------------------+      +---------------------+
| - Infer (YOLOv8)    |
| - Track (nvtracker) |
| - Custom Probe      |
|   (Analytics Logic) |
| - OSD, Encode       |
+---------+-----------+
          |
          | Metadata & Events
          v
+---------------------+      +--------------------------------+
| TrafficStats Class  |----->| Statistics Output              |
| (in trafboard.py)   |      | (stats/*.json)                 |
| - Count Vehicles    |      +--------------------------------+
| - Calc Compliance   |
| - Calc Speed        |
| - Aggregate Data    |
+---------------------+

```

## 5. Features Implemented

- Real-time vehicle detection and counting.
- Vehicle speed estimation.
- Lane adherence monitoring.
- Data recording and playback.
- User-friendly interface for configuration and monitoring.
- Visualization of statistics and analytics.

## 6. Future Improvements

- Integration of additional sensors (e.g., radar, LIDAR) for enhanced data.
- Advanced analytics (e.g., predictive modeling for traffic flow).
- Cloud integration for data storage and processing.
- Mobile application for remote monitoring and configuration.
- Support for more camera types and video formats.

## 7. Acknowledgments

- NVIDIA for DeepStream SDK
- Ultralytics for YOLOv8
- PyQt team for the GUI framework
- ROS2 community for the middleware platform

