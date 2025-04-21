import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import argparse
from std_msgs.msg import String
import rclpy
from collections import deque
import cv2 # Import OpenCV

from deepstream_class import NodeFileSinkPipeline

# Define colors based on theme (adjust alpha as needed)
# Primary blue: #81a1c1 -> (129/255, 161/255, 193/255) -> (0.506, 0.631, 0.757)
COLOR_LINE_DS = (0.506, 0.631, 0.757, 0.8) # RGBA with alpha
# Secondary cyan: #88c0d0 -> (136/255, 192/255, 208/255) -> (0.533, 0.753, 0.816)
COLOR_ROI_DS = (0.533, 0.753, 0.816, 0.6) # RGBA with alpha
# Trail/BBox colors (can keep existing or adjust)
COLOR_BBOX_DS = (1.0, 1.0, 0.0, 1.0) # Yellow
# Thickness
DS_LINE_THICKNESS = 2
DS_ROI_THICKNESS = 2
DS_BBOX_THICKNESS = 2
DS_TRAIL_THICKNESS = 2

class TrafboardPipeline(NodeFileSinkPipeline):
    def __init__(self, pgie_config, input_file_path, tracker_config_path, output_file_path,
                 line_coords, roi_coords, # Add roi_coords
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


     
        self.line_enabled = False
        self.orig_line_x1, self.orig_line_y, self.orig_line_x2 = 0, -1, 0
        self.scaled_line_x1, self.scaled_line_y, self.scaled_line_x2 = 0, -1, 0

        if line_coords and len(line_coords) == 4 and line_coords[1] == line_coords[3]:
          
            self.orig_line_x1 = min(line_coords[0], line_coords[2])
            self.orig_line_y = line_coords[1]
            self.orig_line_x2 = max(line_coords[0], line_coords[2])

           
            self.scaled_line_x1 = int(self.orig_line_x1 * self.scale_x)
            self.scaled_line_y = int(self.orig_line_y * self.scale_y)
            self.scaled_line_x2 = int(self.orig_line_x2 * self.scale_x)

           
            self.scaled_line_x1 = max(0, min(self.scaled_line_x1, self.pipeline_width - 1))
            self.scaled_line_y = max(0, min(self.scaled_line_y, self.pipeline_height - 1))
            self.scaled_line_x2 = max(self.scaled_line_x1, min(self.scaled_line_x2, self.pipeline_width - 1)) 

            self.line_enabled = True
            self.get_logger().info(f"Original Line: ({self.orig_line_x1}, {self.orig_line_y}) to ({self.orig_line_x2}, {self.orig_line_y})")
            self.get_logger().info(f"Scaled Line (for {self.pipeline_width}x{self.pipeline_height}): ({self.scaled_line_x1}, {self.scaled_line_y}) to ({self.scaled_line_x2}, {self.scaled_line_y})")
        else:
            self.get_logger().warning("Line coordinates invalid or not provided. Line crossing disabled.")

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

        self.vehicles_crossed = 0
        self.object_crossed_line = {}

        self.past_tracking_points = {}
        self.max_trail_length = 40

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

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            if self.line_enabled:
                display_meta_line = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                display_meta_line.num_lines = 1
                line_params = display_meta_line.line_params[0]

                line_params.x1 = self.scaled_line_x1
                line_params.y1 = self.scaled_line_y
                line_params.x2 = self.scaled_line_x2
                line_params.y2 = self.scaled_line_y
                line_params.line_width = DS_LINE_THICKNESS
                line_params.line_color.set(*COLOR_LINE_DS)

                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta_line)

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

            display_meta_text = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta_text.num_labels = 1
            text_params = display_meta_text.text_params[0]
            text_params.display_text = f"Vehicles Crossed: {self.vehicles_crossed}"
            text_params.x_offset = 10
            text_params.y_offset = 10
            text_params.font_params.font_name = "Sans Bold"
            text_params.font_params.font_size = 22
            text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            text_params.set_bg_clr = 5
            text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta_text)

            l_obj=frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                    object_id = obj_meta.object_id
                    rect_params = obj_meta.rect_params
                    top = int(rect_params.top)
                    left = int(rect_params.left)
                    width = int(rect_params.width)
                    height = int(rect_params.height)

                    current_point_x = int(left + width / 2)
                    current_point_y = int(top + height / 2)

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

                    if self.line_enabled and object_id not in self.object_crossed_line:
                        if len(trail_points) >= 2:
                            prev_point_y = trail_points[-2][1]

                            crossed_y = ((prev_point_y < self.scaled_line_y <= current_point_y) or \
                                         (current_point_y < self.scaled_line_y <= prev_point_y))

                            if crossed_y:
                                is_within_x_bounds = (self.scaled_line_x1 <= current_point_x <= self.scaled_line_x2)

                                if is_within_x_bounds:
                                    self.vehicles_crossed += 1
                                    self.object_crossed_line[object_id] = True
                                    self.get_logger().info(f"Object ID {object_id} crossed the line. Count: {self.vehicles_crossed}")
                                    
                    rect_params.border_width = DS_BBOX_THICKNESS
                    rect_params.border_color.set(*COLOR_BBOX_DS)

                except StopIteration:
                    break
                try:
                    l_obj=l_obj.next
                except StopIteration:
                    break

            try:
                l_frame=l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK


def main(args=None):
    rclpy.init(args=args)
    pipeline_node = None

    parser = argparse.ArgumentParser(description='Run Trafboard deepstream pipeline with line crossing and save output.')
    parser.add_argument('--pgie-config', type=str, default='config_inferyolov8.txt', help='Path to PGIE config')
    parser.add_argument('--tracker-config', type=str, default='config_tracker.txt', help='Path to tracker config')
    parser.add_argument('input_video', type=str, help='Path to input video')
    parser.add_argument('output_video', type=str, help='Path to save output video')
    parser.add_argument('--line', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'), default=None,
                        help='Crossing line segment coordinates (x1, y1, x2, y2) based on ORIGINAL video resolution. y1 must equal y2.')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('Xmin', 'Ymin', 'Xmax', 'Ymax'), default=None,
                        help='Region of Interest coordinates (xmin, ymin, xmax, ymax) based on ORIGINAL video resolution.')

    parsed_args = parser.parse_args()

    line_coordinates = tuple(parsed_args.line) if parsed_args.line else None

    if line_coordinates and (len(line_coordinates) != 4 or line_coordinates[1] != line_coordinates[3]):
        print("Error: --line requires 4 coordinates (X1, Y1, X2, Y2) where Y1 must equal Y2 for a horizontal line.")
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
            line_coords=line_coordinates,
            roi_coords=roi_coordinates
        )
        pipeline_node.get_logger().info(f"Trafboard Pipeline Node (Sink Mode) created for {parsed_args.input_video} -> {parsed_args.output_video}")
        pipeline_node.run()

    except Exception as e:
        if pipeline_node:
            pipeline_node.get_logger().error(f"Error in Trafboard pipeline: {e}", exc_info=True)
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