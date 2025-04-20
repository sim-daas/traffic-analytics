import sys
sys.path.append('../') # Adjust if necessary
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import argparse
from std_msgs.msg import String # Needed for the probe function
import rclpy
# Add path to deepstream_class if it's not in the standard Python path

# Assuming NodeFilePipeline is importable from deepstream_class
# Adjust the import path based on your project structure
from deepstream_class import NodeFilePipeline

class TrafboardPipeline(NodeFilePipeline):
    def __init__(self, pgie_config, file_path, tracker_config_path, node_name='trafboard_pipeline_node'):
        # Initialize the parent class (NodeFilePipeline)
        super().__init__(pgie_config, file_path, tracker_config_path)

        # Add the Trafboard-specific probe function
        self.osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

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

                    x1 = int(left)
                    y1 = int(top)
                    x2 = int(left + width)
                    y2 = int(top + height)
                    result = str(x1) + ", " + str(x2) + ", " + str(y1) + ", " + str(y2)
                    msg.data = result
                    self.publisher_.publish(msg)

                    rect_params.border_width = 3
                    rect_params.border_color.set(1.0, 1.0, 0.0, 1.0) # Yellow

                    txt_params = obj_meta.text_params
                    if txt_params:
                        txt_params.font_params.font_name = "Sans Bold"

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