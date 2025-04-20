#!/usr/bin/env python3
import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from deepstream_class import Pipeline, Pipeline_tracker , VideoPipeline, NodePipeline, NodeFilePipeline, NodeFileSinkPipeline
import pyds
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import argparse

def osd_sink_pad_buffer_probe(self, pad,info,u_data):
        msg = String()
        gst_buffer = info.get_buffer()
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
                    rect_params.border_color.set(1.0, 1.0, 0.0, 1.0)

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

def main():
    pipeline = Pipeline('sample_720p.h264', 'config_inferyolov8.txt')
    pipeline.osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    pipeline.run()

def main2():
    pipeline = Pipeline_tracker('sample_720p.h264', 'config_inferyolov8.txt', 'config_tracker.txt')
    pipeline.osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    pipeline.run()

def main3():
    rclpy.init()
    pipeline = VideoPipeline('config_inferyolov8.txt', '/dev/video0', 'config_tracker.txt')
    pipeline.osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    pipeline.run()
    rclpy.shutdown()

def main4():
    rclpy.init()
    pipeline = NodePipeline('config_inferyolov8.txt', '/dev/video0', 'config_tracker.txt')
    pipeline.osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    pipeline.run()
    rclpy.shutdown()

def main5(video_file='sample_720p.h264'):
    rclpy.init()
    pipeline = NodeFilePipeline('config_inferyolov8.txt', video_file, 'config_tracker.txt')
    if pipeline.osdsinkpad:
        pipeline.osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, pipeline.osd_sink_pad_buffer_probe, 0)
    else:
        pipeline.get_logger().error("OSD sink pad not available, cannot add probe.")
    pipeline.run()
    rclpy.shutdown()

def main6(input_video='sample_720p.h264', output_video='output.mp4'):
    rclpy.init()
    pipeline_node = None
    pgie_config_path = 'config_inferyolov8.txt'
    tracker_config_path = 'config_tracker.txt'

    try:
        pipeline_node = NodeFileSinkPipeline(
            pgie_config=pgie_config_path,
            input_file_path=input_video,
            tracker_config_path=tracker_config_path,
            output_file_path=output_video
        )
        pipeline_node.get_logger().info(f"NodeFileSinkPipeline created for {input_video} -> {output_video}")

        pipeline_node.run()

    except Exception as e:
        if pipeline_node:
            pipeline_node.get_logger().error(f"Error in NodeFileSinkPipeline: {e}", exc_info=True)
        else:
            print(f"Error creating NodeFileSinkPipeline node: {e}")
    finally:
        if pipeline_node:
            pipeline_node.get_logger().info("Shutting down NodeFileSinkPipeline Node")
            pipeline_node.destroy_node()
        rclpy.shutdown()
        print("ROS shutdown complete.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run DeepStream pipelines.')
    parser.add_argument('--input', type=str, default='sample_720p.h264',
                        help='Path to the input video file.')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Path to the output video file (for file sink pipeline).')
    parser.add_argument('--pipeline', type=str, default='sink', choices=['display', 'sink'],
                        help='Choose pipeline type: "display" (main5) or "sink" (main6).')

    args = parser.parse_args()

    if args.pipeline == 'display':
        print(f"Running display pipeline (main5) with input: {args.input}")
        main5(args.input)
    elif args.pipeline == 'sink':
        print(f"Running file sink pipeline (main6) with input: {args.input}, output: {args.output}")
        main6(args.input, args.output)
    else:
        print("Invalid pipeline choice. Exiting.")
        sys.exit(1)

