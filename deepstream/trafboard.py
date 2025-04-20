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

from deepstream_class import NodeFileSinkPipeline

class TrafboardPipeline(NodeFileSinkPipeline):
    def __init__(self, pgie_config, input_file_path, tracker_config_path, output_file_path,
                 roi_coords,
                 node_name='trafboard_sink_node'):
        super().__init__(pgie_config, input_file_path, tracker_config_path, output_file_path, node_name=node_name)

        if roi_coords and len(roi_coords) == 4:
            self.roi_x, self.roi_y, self.roi_w, self.roi_h = roi_coords
            self.roi_enabled = True
            self.roi_mid_line_x = self.roi_x + self.roi_w // 2
            self.get_logger().info(f"ROI Enabled: x={self.roi_x}, y={self.roi_y}, w={self.roi_w}, h={self.roi_h}, mid_x={self.roi_mid_line_x}")
        else:
            self.roi_enabled = False
            self.get_logger().info("ROI Disabled or invalid coordinates provided.")
            self.roi_x, self.roi_y, self.roi_w, self.roi_h = 0, 0, 0, 0
            self.roi_mid_line_x = 0

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

            if self.roi_enabled:
                display_meta_roi = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                display_meta_roi.num_rects = 1
                display_meta_roi.num_lines = 1
                rect_params_roi = display_meta_roi.rect_params[0]
                line_params_mid = display_meta_roi.line_params[0]

                rect_params_roi.left = self.roi_x
                rect_params_roi.top = self.roi_y
                rect_params_roi.width = self.roi_w
                rect_params_roi.height = self.roi_h
                rect_params_roi.border_width = 2
                rect_params_roi.border_color.set(0.0, 1.0, 1.0, 0.5)

                line_params_mid.x1 = self.roi_mid_line_x
                line_params_mid.y1 = self.roi_y
                line_params_mid.x2 = self.roi_mid_line_x
                line_params_mid.y2 = self.roi_y + self.roi_h
                line_params_mid.line_width = 2
                line_params_mid.line_color.set(1.0, 0.0, 0.0, 0.5)

                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta_roi)

            display_meta_text = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta_text.num_labels = 1
            text_params = display_meta_text.text_params[0]
            text_params.display_text = f"Vehicles Crossed: {self.vehicles_crossed}"
            text_params.x_offset = 10
            text_params.y_offset = 10
            text_params.font_params.font_name = "Sans"
            text_params.font_params.font_size = 12
            text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            text_params.set_bg_clr = 1
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
                    current_point_y = int(top + height)

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
                            line_params = display_meta_trail.line_params[0]
                            line_params.x1, line_params.y1 = start_point[0], start_point[1]
                            line_params.x2, line_params.y2 = end_point[0], end_point[1]
                            line_params.line_width = 2
                            line_params.line_color.set(color[0], color[1], color[2], color[3])
                            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta_trail)

                    if self.roi_enabled and object_id not in self.object_crossed_line:
                        is_in_roi = (self.roi_x <= current_point_x < self.roi_x + self.roi_w and
                                     self.roi_y <= current_point_y < self.roi_y + self.roi_h)

                        if is_in_roi and len(trail_points) >= 2:
                            prev_point_x = trail_points[-2][0]

                            crossed_line = ((prev_point_x < self.roi_mid_line_x <= current_point_x) or \
                                            (current_point_x < self.roi_mid_line_x <= prev_point_x))

                            if crossed_line:
                                self.vehicles_crossed += 1
                                self.object_crossed_line[object_id] = True
                                self.get_logger().info(f"Object ID {object_id} crossed the line. Count: {self.vehicles_crossed}")

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

    parser = argparse.ArgumentParser(description='Run Trafboard deepstream pipeline with ROI and save output.')
    parser.add_argument('--pgie-config', type=str, default='config_inferyolov8.txt', help='Path to PGIE config')
    parser.add_argument('--tracker-config', type=str, default='config_tracker.txt', help='Path to tracker config')
    parser.add_argument('input_video', type=str, help='Path to input video')
    parser.add_argument('output_video', type=str, help='Path to save output video')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'), default=None,
                        help='Region of Interest coordinates (x, y, width, height)')

    parsed_args = parser.parse_args()

    roi_coordinates = tuple(parsed_args.roi) if parsed_args.roi else None

    try:
        pipeline_node = TrafboardPipeline(
            pgie_config=parsed_args.pgie_config,
            input_file_path=parsed_args.input_video,
            tracker_config_path=parsed_args.tracker_config,
            output_file_path=parsed_args.output_video,
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