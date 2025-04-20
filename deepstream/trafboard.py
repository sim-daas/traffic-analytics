import sys
sys.path.append('../') # Adjust if necessary
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import argparse
from std_msgs.msg import String # Needed for the probe function
import rclpy
from collections import deque # Import deque for efficient fixed-length trails
# Add path to deepstream_class if it's not in the standard Python path

# Assuming NodeFilePipeline is importable from deepstream_class
# Adjust the import path based on your project structure
from deepstream_class import NodeFilePipeline

class TrafboardPipeline(NodeFilePipeline):
    def __init__(self, pgie_config, file_path, tracker_config_path, node_name='trafboard_pipeline_node'):
        # Initialize the parent class (NodeFilePipeline)
        super().__init__(pgie_config, file_path, tracker_config_path)

        # Add the Trafboard-specific probe function
        # Make sure osdsinkpad is valid before adding probe
        if self.osdsinkpad:
            self.probe_id = self.osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)
        else:
             self.get_logger().error("OSD sink pad not found during initialization. Cannot add Trafboard probe.")
             # Handle error appropriately, maybe exit or raise exception
             sys.exit(1)

        # Storage for past tracking points: dict mapping object_id -> deque of (x, y) points
        self.past_tracking_points = {}
        self.max_trail_length = 40 # Keep the last 30 points for the trail

        self.get_logger().info(f"TrafboardPipeline ({node_name}) post-initialization complete.")

    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        msg = String()
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            self.get_logger().error("Unable to get GstBuffer")
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                # Get frame number for potential cleanup later (optional)
                frame_number = frame_meta.frame_num
            except StopIteration:
                break

            l_obj=frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                    rect_params = obj_meta.rect_params
                    top = int(rect_params.top)
                    left = int(rect_params.left)
                    width = int(rect_params.width)
                    height = int(rect_params.height)

                    # Calculate center point of the object for the trail
                    current_point_x = int(left + width / 2)
                    current_point_y = int(top + height / 2)

                    # Get object ID
                    object_id = obj_meta.object_id

                    # Update past points storage
                    if object_id not in self.past_tracking_points:
                        # Use deque for fixed-size efficient storage
                        self.past_tracking_points[object_id] = deque(maxlen=self.max_trail_length)
                    
                    # Add current point and frame number (optional)
                    self.past_tracking_points[object_id].append((current_point_x, current_point_y))

                    # --- Draw Bounding Box ---
                    x1 = int(left)
                    y1 = int(top)
                    x2 = int(left + width)
                    y2 = int(top + height)
                    result = str(x1) + ", " + str(x2) + ", " + str(y1) + ", " + str(y2)
                    msg.data = result
                    self.publisher_.publish(msg)

                    rect_params.border_width = 3
                    # Define a list of colors (R,G,B,A format)
                    colors = [
                        (1.0, 1.0, 0.0, 1.0),  # Yellow
                        (0.0, 1.0, 0.0, 1.0),  # Green
                        (0.0, 0.0, 1.0, 1.0),  # Blue
                        (1.0, 0.0, 1.0, 1.0),  # Magenta
                        (0.0, 1.0, 1.0, 1.0),  # Cyan
                        (1.0, 0.5, 0.0, 1.0)   # Orange
                    ]
                    # Use the object ID to select a color (modulo to cycle through the list)
                    color_index = object_id % len(colors)
                    color = colors[color_index]
                    # Set the border color
                    rect_params.border_color.set(color[0], color[1], color[2], color[3])

                    # --- Draw Trail ---
                    trail_points = self.past_tracking_points[object_id]
                    if len(trail_points) >= 2:
                        # Iterate through pairs of points to draw line segments
                        for i in range(len(trail_points) - 1):
                            start_point = trail_points[i]
                            end_point = trail_points[i+1]

                            # Acquire display meta from pool
                            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                            display_meta.num_lines = 1
                            line_params = display_meta.line_params[0]

                            # Set line coordinates
                            line_params.x1 = start_point[0]
                            line_params.y1 = start_point[1]
                            line_params.x2 = end_point[0]
                            line_params.y2 = end_point[1]

                            # Set line properties
                            line_params.line_width = 2 # Trail line width
                            # Use the same color as the bounding box for the trail
                            line_params.line_color.set(color[0], color[1], color[2], color[3])

                            # Add the line metadata to the frame
                            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

                    # --- Draw Text ---
                    txt_params = obj_meta.text_params
                    if txt_params:
                        txt_params.font_params.font_name = "Sans Bold"
                        # Optionally add object ID to text
                        txt_params.display_text = f"ID: {object_id}"
                        txt_params.font_params.font_size = 10
                        txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0) # White text
                        txt_params.set_bg_clr = 1
                        txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7) # Semi-transparent black background

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
    """
    Main function to initialize ROS, parse arguments, create and run the TrafboardPipeline.
    """
    rclpy.init(args=args)
    pipeline_node = None

    parser = argparse.ArgumentParser(description='Run Trafboard deepstream pipeline with a video file')
    parser.add_argument('--pgie-config', type=str, default='config_inferyolov8.txt',
                        help='Path to the Primary GPU Inference Engine config file')
    parser.add_argument('--tracker-config', type=str, default='config_tracker.txt',
                        help='Path to the tracker config file')
    parser.add_argument('video', type=str, help='Path to the video file')
    parsed_args, remaining_args = parser.parse_known_args()

    try:
        pipeline_node = TrafboardPipeline(
            pgie_config=parsed_args.pgie_config,
            file_path=parsed_args.video,
            tracker_config_path=parsed_args.tracker_config
        )
        pipeline_node.get_logger().info("Trafboard Pipeline Node created")
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