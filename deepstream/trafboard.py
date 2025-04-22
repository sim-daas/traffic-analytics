import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import argparse
from std_msgs.msg import String
import rclpy
from collections import deque, defaultdict
import cv2 # Import OpenCV
import itertools # For color cycling
import json
import os
import time
import numpy as np
from deepstream_class import NodeFileSinkPipeline

# Define colors based on theme (adjust alpha as needed)
# Use a cycle of colors for multiple lines in DeepStream output
DS_LINE_COLORS = itertools.cycle([
    (0.506, 0.631, 0.757, 0.8), # Primary blue
    (0.639, 0.749, 0.549, 0.8), # Green
    (0.921, 0.796, 0.545, 0.8), # Yellow
    (0.816, 0.529, 0.439, 0.8), # Orange
    (0.706, 0.541, 0.678, 0.8), # Purple
])
COLOR_ROI_DS = (0.533, 0.753, 0.816, 0.6) # Cyan
COLOR_BBOX_DS = (1.0, 1.0, 0.0, 1.0) # Yellow
# Thickness
DS_LINE_THICKNESS = 2
DS_ROI_THICKNESS = 2
DS_BBOX_THICKNESS = 2
DS_TRAIL_THICKNESS = 2

# Define class labels based on YOLO configuration
VEHICLE_CLASSES = {
    0: 'person',       # Not a vehicle but useful to track
    1: 'bicycle',      
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
    # Add other vehicle classes as needed
}

class TrafficStats:
    """Class for tracking and storing traffic statistics"""
    def __init__(self, lines, output_dir='./stats'):
        self.start_time = time.time()
        
        # Create stats directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.stats_file = os.path.join(output_dir, f"traffic_stats_{int(self.start_time)}.json")
        self.latest_stats_file = os.path.join(output_dir, "latest_stats.json")
        
        # Vehicle counts per line and direction
        self.line_counts = {}
        for line in lines:
            line_id = line['id']
            self.line_counts[line_id] = {
                'inbound': defaultdict(int),   # Vehicles moving down
                'outbound': defaultdict(int),  # Vehicles moving up
                'total': defaultdict(int)      # Total regardless of direction
            }
            
        # Time series data (each entry is a timestamp with counts)
        self.time_series = []
        self.last_time_sample = self.start_time
        self.time_sample_interval = 1.0  # Record counts every second
        
        # Speed tracking data
        self.speeds = defaultdict(list)  # Speed by vehicle class
        self.avg_speeds = {}             # Will be calculated at the end
        
        # Lane occupancy tracking
        self.lane_occupancy = defaultdict(float) # Percentage time each lane is occupied
        self.lane_time_blocks = {}               # Track time blocks for occupancy calculation
        
        # Vehicle position tracking for speed calculation
        self.vehicle_positions = {}  # {object_id: [(time, pos_x, pos_y), ...]}
        
        # Initialize the current time slice
        self._update_time_slice()
        
    def _update_time_slice(self):
        """Initialize or update the current time slice for time series data"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Create a new time slice if interval has passed
        if current_time - self.last_time_sample >= self.time_sample_interval:
            # Add the current counts to time series
            time_slice = {
                'timestamp': elapsed_time,
                'counts': {}
            }
            
            # Record counts for each line
            for line_id, counts in self.line_counts.items():
                time_slice['counts'][str(line_id)] = {
                    'inbound': dict(counts['inbound']),
                    'outbound': dict(counts['outbound']),
                    'total': dict(counts['total'])
                }
                
            self.time_series.append(time_slice)
            self.last_time_sample = current_time
    
    def update_vehicle_crossing(self, line_id, object_id, vehicle_class, is_inbound):
        """Update statistics when a vehicle crosses a line"""
        # Increase counts for the appropriate line and direction
        direction = 'inbound' if is_inbound else 'outbound'
        class_name = VEHICLE_CLASSES.get(vehicle_class, f'class_{vehicle_class}')
        
        self.line_counts[line_id][direction][class_name] += 1
        self.line_counts[line_id]['total'][class_name] += 1
        
        # Update time series data
        self._update_time_slice()
        
    def update_vehicle_position(self, object_id, class_id, position_x, position_y):
        """Track vehicle positions for speed calculation"""
        current_time = time.time()
        if object_id not in self.vehicle_positions:
            self.vehicle_positions[object_id] = []
        
        # Add current position with timestamp
        self.vehicle_positions[object_id].append((current_time, position_x, position_y))
        
        # Keep only the last 10 positions per vehicle (to limit memory usage)
        if len(self.vehicle_positions[object_id]) > 10:
            self.vehicle_positions[object_id].pop(0)
            
        # Calculate speed if we have multiple positions
        if len(self.vehicle_positions[object_id]) >= 2:
            self._calculate_vehicle_speed(object_id, class_id)
    
    def _calculate_vehicle_speed(self, object_id, class_id):
        """Calculate vehicle speed from position history"""
        positions = self.vehicle_positions[object_id]
        if len(positions) < 2:
            return
            
        # Get the oldest and newest positions
        oldest = positions[0]
        newest = positions[-1]
        
        # Calculate time difference
        time_diff = newest[0] - oldest[0]
        if time_diff <= 0:
            return
            
        # Calculate distance (in pixels)
        dx = newest[1] - oldest[1]
        dy = newest[2] - oldest[2]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Calculate speed (pixels per second)
        speed = distance / time_diff
        
        # Store speed for this vehicle class
        class_name = VEHICLE_CLASSES.get(class_id, f'class_{class_id}')
        self.speeds[class_name].append(speed)
    
    def update_lane_occupancy(self, lane_id, is_occupied):
        """Update lane occupancy statistics"""
        current_time = time.time()
        
        # Initialize tracking for this lane if needed
        if lane_id not in self.lane_time_blocks:
            self.lane_time_blocks[lane_id] = {
                'last_update': current_time,
                'total_occupied_time': 0.0
            }
            
        # If state changed or periodic update
        if is_occupied:
            # Lane is occupied, accumulate time
            elapsed = current_time - self.lane_time_blocks[lane_id]['last_update']
            self.lane_time_blocks[lane_id]['total_occupied_time'] += elapsed
            
        # Update last update time
        self.lane_time_blocks[lane_id]['last_update'] = current_time
        
        # Calculate and update occupancy percentage
        total_time = current_time - self.start_time
        if total_time > 0:
            occupancy_pct = (self.lane_time_blocks[lane_id]['total_occupied_time'] / total_time) * 100
            self.lane_occupancy[lane_id] = occupancy_pct
    
    def calculate_final_stats(self):
        """Calculate final statistics before saving"""
        # Calculate average speeds
        for vehicle_class, speeds in self.speeds.items():
            if speeds:
                self.avg_speeds[vehicle_class] = sum(speeds) / len(speeds)
        
        # Final update of lane occupancy
        for lane_id in self.lane_time_blocks:
            self.update_lane_occupancy(lane_id, False)
            
        # Final time slice update
        self._update_time_slice()
    
    def save_stats(self):
        """Save all statistics to file"""
        self.calculate_final_stats()
        
        # Prepare data structure for JSON
        stats_data = {
            'start_time': self.start_time,
            'end_time': time.time(),
            'duration': time.time() - self.start_time,
            'line_counts': {str(k): v for k, v in self.line_counts.items()},
            'time_series': self.time_series,
            'avg_speeds': self.avg_speeds,
            'lane_occupancy': {str(k): v for k, v in self.lane_occupancy.items()},
        }
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats_json = json.dumps(stats_data, indent=2)
        
        # Ensure directory exists again (just to be safe)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save to timestamp-based file
        try:
            with open(self.stats_file, 'w') as f:
                f.write(stats_json)
        except Exception as e:
            print(f"Error saving to timestamped file: {e}")
            
        # Save to latest file (always overwrite)
        try:
            with open(self.latest_stats_file, 'w') as f:
                f.write(stats_json)
        except Exception as e:
            print(f"Error saving to latest file: {e}")
            
        return self.stats_file


class TrafboardPipeline(NodeFileSinkPipeline):
    def __init__(self, pgie_config, input_file_path, tracker_config_path, output_file_path,
                 lines_data, roi_coords,
                 stats_dir='./stats',
                 node_name='trafboard_line_sink_node'):
        super().__init__(pgie_config, input_file_path, tracker_config_path, output_file_path, node_name=node_name)

        try:
            cap = cv2.VideoCapture(input_file_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {input_file_path}")
            self.source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if self.source_width <= 0 or self.source_height <= 0:
                 raise ValueError("Invalid source video dimensions obtained.")
            self.get_logger().info(f"Source video resolution: {self.source_width}x{self.source_height}")
        except Exception as e:
            self.get_logger().error(f"Failed to get source video resolution: {e}. Using pipeline defaults.")
           
            self.source_width = self.streammux.get_property('width')
            self.source_height = self.streammux.get_property('height')

      
        self.pipeline_width = self.streammux.get_property('width')
        self.pipeline_height = self.streammux.get_property('height')
        self.get_logger().info(f"Pipeline processing resolution: {self.pipeline_width}x{self.pipeline_height}")

      
        if self.source_width > 0 and self.source_height > 0:
            self.scale_x = self.pipeline_width / self.source_width
            self.scale_y = self.pipeline_height / self.source_height
        else:
            self.get_logger().warning("Source dimensions are zero, cannot calculate scale factors. Assuming scale=1.")
            self.scale_x = 1.0
            self.scale_y = 1.0
        self.get_logger().info(f"Scaling factors (pipeline/source): scale_x={self.scale_x:.4f}, scale_y={self.scale_y:.4f}")

        # --- Lines Setup (Multiple) ---
        self.lines = [] # Store line info: [{'id': id, 'orig': (x1,y,x2,y), 'scaled': (sx1,sy,sx2,sy)}]
        self.line_colors = {} # Store color per line ID: {line_id: (r,g,b,a)}

        if lines_data:
            for line_info in lines_data:
                line_id = line_info['id']
                orig_coords = line_info['coords']
                orig_x1 = min(orig_coords[0], orig_coords[2])
                orig_y = orig_coords[1] # y1 == y2 enforced by UI/argparse
                orig_x2 = max(orig_coords[0], orig_coords[2])

                scaled_x1 = int(orig_x1 * self.scale_x)
                scaled_y = int(orig_y * self.scale_y)
                scaled_x2 = int(orig_x2 * self.scale_x)

                scaled_x1 = max(0, min(scaled_x1, self.pipeline_width - 1))
                scaled_y = max(0, min(scaled_y, self.pipeline_height - 1))
                scaled_x2 = max(scaled_x1, min(scaled_x2, self.pipeline_width - 1))

                scaled_coords = (scaled_x1, scaled_y, scaled_x2, scaled_y)
                self.lines.append({'id': line_id, 'orig': (orig_x1, orig_y, orig_x2, orig_y), 'scaled': scaled_coords})
                self.line_colors[line_id] = next(DS_LINE_COLORS) # Assign color

                self.get_logger().info(f"Line ID {line_id}: Original=({orig_x1},{orig_y})-({orig_x2},{orig_y}), Scaled=({scaled_x1},{scaled_y})-({scaled_x2},{scaled_y})")
        else:
             self.get_logger().warning("No line coordinates provided. Line crossing disabled.")
        # --- End Lines Setup ---

        # --- ROI Setup ---
        self.roi_enabled = False
        self.orig_roi_x1, self.orig_roi_y1, self.orig_roi_x2, self.orig_roi_y2 = 0, 0, 0, 0
        self.scaled_roi_x1, self.scaled_roi_y1, self.scaled_roi_x2, self.scaled_roi_y2 = 0, 0, 0, 0

        if roi_coords and len(roi_coords) == 4:
            self.orig_roi_x1 = min(roi_coords[0], roi_coords[2])
            self.orig_roi_y1 = min(roi_coords[1], roi_coords[3])
            self.orig_roi_x2 = max(roi_coords[0], roi_coords[2])
            self.orig_roi_y2 = max(roi_coords[1], roi_coords[3])

            self.scaled_roi_x1 = int(self.orig_roi_x1 * self.scale_x)
            self.scaled_roi_y1 = int(self.orig_roi_y1 * self.scale_y)
            self.scaled_roi_x2 = int(self.orig_roi_x2 * self.scale_x)
            self.scaled_roi_y2 = int(self.orig_roi_y2 * self.scale_y)

            self.scaled_roi_x1 = max(0, min(self.scaled_roi_x1, self.pipeline_width - 1))
            self.scaled_roi_y1 = max(0, min(self.scaled_roi_y1, self.pipeline_height - 1))
            self.scaled_roi_x2 = max(self.scaled_roi_x1, min(self.scaled_roi_x2, self.pipeline_width - 1))
            self.scaled_roi_y2 = max(self.scaled_roi_y1, min(self.scaled_roi_y2, self.pipeline_height - 1))

            self.roi_enabled = True
            self.get_logger().info(f"Original ROI: ({self.orig_roi_x1}, {self.orig_roi_y1}) to ({self.orig_roi_x2}, {self.orig_roi_y2})")
            self.get_logger().info(f"Scaled ROI (for {self.pipeline_width}x{self.pipeline_height}): ({self.scaled_roi_x1}, {self.scaled_roi_y1}) to ({self.scaled_roi_x2}, {self.scaled_roi_y2})")
        else:
            self.get_logger().warning("ROI coordinates invalid or not provided. ROI filtering disabled.")

        # --- Traffic Statistics Setup ---
        self.stats = TrafficStats(self.lines, stats_dir) if self.lines else None
        self.lane_ids = {i+1: line['id'] for i, line in enumerate(self.lines)} # Map lane numbers to line IDs
        # --- End Statistics Setup ---

        # --- Crossing Counters (Multiple Lines) ---
        self.vehicles_crossed = {line['id']: 0 for line in self.lines} # Count per line ID
        self.object_crossed_line = {} # {object_id: {line_id: True}}
        # --- End Crossing Counters ---

        self.past_tracking_points = {}
        self.max_trail_length = 40

        # Store last frame timestamp
        self.last_frame_time = time.time()
        
        # Store frames where object IDs appear
        self.object_frames = defaultdict(int)
        self.frame_count = 0
        
        # Track if we've already saved stats
        self.stats_saved = False

        if self.osdsinkpad:
            self.probe_id = self.osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.custom_probe, 0)
            self.get_logger().info("Custom probe attached successfully.")
        else:
            self.get_logger().error("OSD sink pad not found during Trafboard init. Cannot add custom probe.")

        self.get_logger().info(f"TrafboardPipeline (Sink Mode - {node_name}) post-initialization complete.")
    
    def custom_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            self.get_logger().warn("Unable to get GstBuffer in custom_probe")
            return Gst.PadProbeReturn.OK

        # Update frame count for statistics
        self.frame_count += 1
        current_time = time.time()
        frame_delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            # --- Draw Crossing Lines (Multiple) ---
            display_meta_lines = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta_lines.num_lines = len(self.lines)
            line_idx = 0
            for line in self.lines:
                line_params = display_meta_lines.line_params[line_idx]
                scaled_coords = line['scaled']
                line_id = line['id']
                color = self.line_colors.get(line_id, (1.0, 1.0, 1.0, 1.0)) # Default white

                line_params.x1 = scaled_coords[0]
                line_params.y1 = scaled_coords[1]
                line_params.x2 = scaled_coords[2]
                line_params.y2 = scaled_coords[3]
                line_params.line_width = DS_LINE_THICKNESS
                line_params.line_color.set(*color)
                line_idx += 1
            if display_meta_lines.num_lines > 0:
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta_lines)

            # --- Draw ROI Rectangle ---
            if self.roi_enabled:
                display_meta_roi = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                display_meta_roi.num_rects = 1
                rect_params_roi = display_meta_roi.rect_params[0]
                rect_params_roi.left = self.scaled_roi_x1
                rect_params_roi.top = self.scaled_roi_y1
                rect_params_roi.width = self.scaled_roi_x2 - self.scaled_roi_x1
                rect_params_roi.height = self.scaled_roi_y2 - self.scaled_roi_y1
                rect_params_roi.border_width = DS_ROI_THICKNESS
                rect_params_roi.border_color.set(*COLOR_ROI_DS)
                rect_params_roi.has_bg_color = 0
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta_roi)

            # --- Draw Vehicle Count Text (Multiple Lines) ---
            display_meta_text = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            num_labels = len(self.vehicles_crossed)
            display_meta_text.num_labels = num_labels if num_labels > 0 else 1 # Show at least one label
            y_offset_start = 10
            label_idx = 0
            if num_labels > 0:
                for line_id, count in sorted(self.vehicles_crossed.items()):
                    text_params = display_meta_text.text_params[label_idx]
                    text_params.display_text = f"Line {line_id} Crossed: {count}"
                    text_params.x_offset = 10
                    text_params.y_offset = y_offset_start + label_idx * 30 # Stack labels
                    text_params.font_params.font_name = "Sans" # Non-bold
                    text_params.font_params.font_size = 16 # Smaller font
                    text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                    text_params.set_bg_clr = 1 # Enable background color
                    text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6) # Black background
                    label_idx += 1
            else: # No lines defined
                 text_params = display_meta_text.text_params[0]
                 text_params.display_text = "No lines defined"
                 text_params.x_offset = 10
                 text_params.y_offset = y_offset_start
                 text_params.font_params.font_name = "Sans"
                 text_params.font_params.font_size = 16
                 text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                 text_params.set_bg_clr = 1
                 text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta_text)
            # --- End Text Drawing ---

            # --- Process detected objects ---
            l_obj=frame_meta.obj_meta_list
            
            # Track which lanes are occupied this frame
            lane_occupied = defaultdict(bool)
            
            while l_obj is not None:
                try:
                    obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                    object_id = obj_meta.object_id
                    class_id = obj_meta.class_id
                    rect_params = obj_meta.rect_params
                    top = int(rect_params.top)
                    left = int(rect_params.left)
                    width = int(rect_params.width)
                    height = int(rect_params.height)

                    current_point_x = int(left + width / 2)
                    current_point_y = int(top + height / 2)
                    
                    # Update statistics for this object
                    if self.stats and class_id in VEHICLE_CLASSES:
                        self.stats.update_vehicle_position(object_id, class_id, current_point_x, current_point_y)
                    
                    # Count frame appearance for this object ID
                    self.object_frames[object_id] += 1

                    # --- ROI Filtering ---
                    is_within_roi = True
                    if self.roi_enabled:
                        is_within_roi = (self.scaled_roi_x1 <= current_point_x <= self.scaled_roi_x2 and
                                         self.scaled_roi_y1 <= current_point_y <= self.scaled_roi_y2)

                    if not is_within_roi:
                        rect_params.border_width = 0
                        txt_params = obj_meta.text_params
                        if txt_params:
                            txt_params.display_text = ""
                        try:
                            l_obj=l_obj.next
                        except StopIteration:
                            break
                        continue

                    # --- Trail tracking and drawing ---
                    if object_id not in self.past_tracking_points:
                        self.past_tracking_points[object_id] = deque(maxlen=self.max_trail_length)
                    self.past_tracking_points[object_id].append((current_point_x, current_point_y))

                    trail_points = self.past_tracking_points[object_id]

                    colors = [ (1.0, 1.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0),
                               (1.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 1.0), (1.0, 0.5, 0.0, 1.0) ]
                    color_index = object_id % len(colors)
                    color = colors[color_index]

                    if len(trail_points) >= 2:
                        for i in range(len(trail_points) - 1):
                            start_point = trail_points[i]
                            end_point = trail_points[i+1]
                            display_meta_trail = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                            display_meta_trail.num_lines = 1
                            line_params_trail = display_meta_trail.line_params[0]
                            line_params_trail.x1, line_params_trail.y1 = start_point[0], start_point[1]
                            line_params_trail.x2, line_params_trail.y2 = end_point[0], end_point[1]
                            line_params_trail.line_width = DS_TRAIL_THICKNESS
                            line_params_trail.line_color.set(color[0], color[1], color[2], color[3])
                            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta_trail)

                    # --- Line Crossing Check and Statistics ---
                    if self.lines and len(trail_points) >= 2:
                        prev_point_y = trail_points[-2][1]

                        for line in self.lines:
                            line_id = line['id']
                            scaled_line_y = line['scaled'][1]
                            scaled_line_x1 = line['scaled'][0]
                            scaled_line_x2 = line['scaled'][2]
                            
                            # Check which lane this object is in
                            if scaled_line_x1 <= current_point_x <= scaled_line_x2:
                                # Mark this lane as occupied
                                for lane_num, lid in self.lane_ids.items():
                                    if lid == line_id:
                                        lane_occupied[lane_num] = True
                                        if self.stats:
                                            self.stats.update_lane_occupancy(lane_num, True)

                            # Skip if already counted for this line
                            if object_id in self.object_crossed_line and line_id in self.object_crossed_line[object_id]:
                                continue

                            # Check if crossing occurred
                            crossed_y = ((prev_point_y < scaled_line_y <= current_point_y) or
                                        (current_point_y < scaled_line_y <= prev_point_y))

                            if crossed_y:
                                is_within_x_bounds = (scaled_line_x1 <= current_point_x <= scaled_line_x2)

                                if is_within_x_bounds:
                                    # Determine direction (inbound = top to bottom, outbound = bottom to top)
                                    is_inbound = prev_point_y < scaled_line_y
                                    
                                    # Update statistics
                                    if self.stats and class_id in VEHICLE_CLASSES:
                                        self.stats.update_vehicle_crossing(line_id, object_id, class_id, is_inbound)
                                    
                                    # Update crossing counters
                                    self.vehicles_crossed[line_id] += 1
                                    if object_id not in self.object_crossed_line:
                                        self.object_crossed_line[object_id] = {}
                                    self.object_crossed_line[object_id][line_id] = True
                                    
                                    self.get_logger().info(
                                        f"Object ID {object_id} (class: {VEHICLE_CLASSES.get(class_id, class_id)}) "
                                        f"crossed Line {line_id} {'inbound' if is_inbound else 'outbound'}. "
                                        f"Count for Line {line_id}: {self.vehicles_crossed[line_id]}"
                                    )

                    rect_params.border_width = DS_BBOX_THICKNESS
                    rect_params.border_color.set(*COLOR_BBOX_DS)

                except StopIteration:
                    break
                try:
                    l_obj=l_obj.next
                except StopIteration:
                    break
                    
            # Update unoccupied lanes
            if self.stats:
                for lane_num in self.lane_ids.keys():
                    if not lane_occupied[lane_num]:
                        self.stats.update_lane_occupancy(lane_num, False)

            try:
                l_frame=l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK
        
    def run(self):
        """Override run method to save statistics at the end"""
        try:
            super().run()
        finally:
            if self.stats and not self.stats_saved:
                stats_file = self.stats.save_stats()
                self.stats_saved = True
                self.get_logger().info(f"Traffic statistics saved to {stats_file}")


def main(args=None):
    rclpy.init(args=args)
    pipeline_node = None

    parser = argparse.ArgumentParser(description='Run Trafboard deepstream pipeline with line crossing and save output.')
    parser.add_argument('--pgie-config', type=str, default='config_inferyolov8.txt', help='Path to PGIE config')
    parser.add_argument('--tracker-config', type=str, default='config_tracker.txt', help='Path to tracker config')
    parser.add_argument('input_video', type=str, help='Path to input video')
    parser.add_argument('output_video', type=str, help='Path to save output video')
    parser.add_argument('--lines', type=int, nargs='+', metavar='ID X1 Y1 X2 Y2', default=[], # Changed metavar to a single string
                        help='Crossing line segments. Provide as ID X1 Y1 X2 Y2 [ID X1 Y1 X2 Y2 ...]. Y1 must equal Y2.')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('Xmin', 'Ymin', 'Xmax', 'Ymax'), default=None,
                        help='Region of Interest coordinates (xmin, ymin, xmax, ymax) based on ORIGINAL video resolution.')
    parser.add_argument('--stats-dir', type=str, default='./stats', 
                        help='Directory to save traffic statistics JSON files')

    parsed_args = parser.parse_args()

    lines_input = parsed_args.lines
    lines_data = [] # To store [{'id': id, 'coords': (x1,y1,x2,y2)}, ...]
    if lines_input:
        if len(lines_input) % 5 != 0:
            print("Error: --lines requires arguments in groups of 5 (ID, X1, Y1, X2, Y2).")
            sys.exit(1)
        for i in range(0, len(lines_input), 5):
            try:
                line_id = lines_input[i]
                x1 = lines_input[i+1]
                y1 = lines_input[i+2]
                x2 = lines_input[i+3]
                y2 = lines_input[i+4]
                if y1 != y2:
                     print(f"Error: Line ID {line_id} is not horizontal (Y1={y1}, Y2={y2}). Only horizontal lines supported.")
                     sys.exit(1)
                lines_data.append({'id': line_id, 'coords': (x1, y1, x2, y2)})
            except IndexError:
                 print("Error parsing --lines arguments.")
                 sys.exit(1)

    roi_coordinates = tuple(parsed_args.roi) if parsed_args.roi else None
    if roi_coordinates:
        if len(roi_coordinates) != 4:
            print("Error: --roi requires 4 coordinates (Xmin, Ymin, Xmax, Ymax).")
            sys.exit(1)
        if roi_coordinates[0] >= roi_coordinates[2] or roi_coordinates[1] >= roi_coordinates[3]:
            print("Error: ROI coordinates invalid. Xmin must be < Xmax and Ymin must be < Ymax.")
            sys.exit(1)

    try:
        pipeline_node = TrafboardPipeline(
            pgie_config=parsed_args.pgie_config,
            input_file_path=parsed_args.input_video,
            tracker_config_path=parsed_args.tracker_config,
            output_file_path=parsed_args.output_video,
            lines_data=lines_data, # Pass parsed lines data
            roi_coords=roi_coordinates,
            stats_dir=parsed_args.stats_dir
        )
        pipeline_node.get_logger().info(f"Trafboard Pipeline Node (Sink Mode) created for {parsed_args.input_video} -> {parsed_args.output_video}")
        pipeline_node.run()

    except Exception as e:
        if pipeline_node:
            pipeline_node.get_logger().error(f"Error in Trafboard pipeline: {e}")
        else:
            print(f"Error creating Trafboard pipeline node: {e}")
    finally:
        if pipeline_node:
            pipeline_node.get_logger().info("Shutting down Trafboard Pipeline Node")
            if pipeline_node.pipeline:
                 pipeline_node.pipeline.set_state(Gst.State.NULL)
            pipeline_node.destroy_node()
        rclpy.shutdown()
        print("ROS shutdown complete.")

if __name__ == '__main__':
    main(sys.argv)