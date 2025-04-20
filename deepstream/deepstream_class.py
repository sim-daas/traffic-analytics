#!/usr/bin/env python3
import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import configparser
import pyds
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True

class NodeFilePipeline(Node):
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
                    if obj_meta.class_id == 0: 
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


    def __init__(self, pgie_config, file_path, tracker_config_path):
        super().__init__('inference_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 0)
        Gst.init(None)
        
        self.pipeline = Gst.Pipeline()

        # Elements for MP4 support
        self.source = Gst.ElementFactory.make("filesrc", "file-source")
        self.demuxer = Gst.ElementFactory.make("qtdemux", "qt-demuxer")
        self.h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        self.decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        self.pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        self.sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

        self.source.set_property('location', file_path)
        self.streammux.set_property('width', 1920)
        self.streammux.set_property('height', 1080)
        self.streammux.set_property('batch-size', 1)
        self.streammux.set_property('batched-push-timeout', 4000000)
        self.pgie.set_property('config-file-path', pgie_config)

        config = configparser.ConfigParser()
        config.read(tracker_config_path)
        config.sections()

        for key in config['tracker']:
            if key == 'tracker-width' :
                tracker_width = config.getint('tracker', key)
                self.tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height' :
                tracker_height = config.getint('tracker', key)
                self.tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id' :
                tracker_gpu_id = config.getint('tracker', key)
                self.tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file' :
                tracker_ll_lib_file = config.get('tracker', key)
                self.tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file' :
                tracker_ll_config_file = config.get('tracker', key)
                self.tracker.set_property('ll-config-file', tracker_ll_config_file)

        # Adding elements to the pipeline
        self.pipeline.add(self.source)
        self.pipeline.add(self.demuxer)
        self.pipeline.add(self.h264parser)
        self.pipeline.add(self.decoder)
        self.pipeline.add(self.streammux)
        self.pipeline.add(self.pgie)
        self.pipeline.add(self.tracker)
        self.pipeline.add(self.nvvidconv)
        self.pipeline.add(self.nvosd)
        self.pipeline.add(self.sink)

        # Linking elements
        self.source.link(self.demuxer)
        self.demuxer.connect("pad-added", self.on_pad_added)
        self.h264parser.link(self.decoder)

        sinkpad = self.streammux.get_request_pad("sink_0")
        srcpad = self.decoder.get_static_pad("src")

        srcpad.link(sinkpad)
        self.streammux.link(self.pgie)
        self.pgie.link(self.tracker)
        self.tracker.link(self.nvvidconv)
        self.nvvidconv.link(self.nvosd)
        self.nvosd.link(self.sink)

    def on_pad_added(self, src, pad):
        caps = pad.query_caps(None)
        name = caps.to_string()
        if name.startswith("video/x-h264"):
            pad.link(self.h264parser.get_static_pad("sink"))
        self.osdsinkpad = self.nvosd.get_static_pad("sink")

        self.osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

    def run(self):

        self.pipeline.set_state(Gst.State.PLAYING)

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect ("message", bus_call, self.loop)

        try:
            self.loop.run()
        except:
            pass

        self.pipeline.set_state(Gst.State.NULL)

class Pipeline:
    def __init__(self, file_path, config_file):
        Gst.init(None)

        self.pipeline = Gst.Pipeline()
        self.source = Gst.ElementFactory.make("filesrc", "file-source")
        self.h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        self.decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        self.pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        self.sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        #self.transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

        self.source.set_property('location', file_path)
        self.streammux.set_property('width', 1920)
        self.streammux.set_property('height', 1080)
        self.streammux.set_property('batch-size', 1)
        self.streammux.set_property('batched-push-timeout', 4000000)
        self.pgie.set_property('config-file-path', config_file)

        self.pipeline.add(self.source)
        self.pipeline.add(self.h264parser)
        self.pipeline.add(self.decoder)
        self.pipeline.add(self.streammux)
        self.pipeline.add(self.pgie)
        self.pipeline.add(self.nvvidconv)
        self.pipeline.add(self.nvosd)
        self.pipeline.add(self.sink)
        #self.pipeline.add(self.transform)

        sinkpad = self.streammux.get_request_pad("sink_0")
        srcpad = self.decoder.get_static_pad("src")

        self.source.link(self.h264parser)
        self.h264parser.link(self.decoder)
        srcpad.link(sinkpad)
        self.streammux.link(self.pgie)
        self.pgie.link(self.nvvidconv)
        self.nvvidconv.link(self.nvosd)
        self.nvosd.link(self.sink)
        #self.nvosd.link(self.transform)
        #self.transform.link(self.sink)

        self.osdsinkpad = self.nvosd.get_static_pad("sink")
        #self.osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

    def osd_sink_pad_buffer_probe(self, pad,info,u_data):
    
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
                    print(obj_meta.confidence)
                    print("Hello world")
                except StopIteration:
                    break
                
                obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)
                try: 
                    l_obj=l_obj.next
                except StopIteration:
                    break

            try:
                l_frame=l_frame.next
            except StopIteration:
                break
			
        return Gst.PadProbeReturn.OK

    def run(self):

        self.pipeline.set_state(Gst.State.PLAYING)

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect ("message", bus_call, self.loop)

        try:
            self.loop.run()
        except:
            pass

        self.pipeline.set_state(Gst.State.NULL)

class VideoPipeline:
    def __init__(self, pgie_config, device, tracker_config_path):
        Gst.init(None)

        self.pipeline = Gst.Pipeline()

        self.source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
        self.caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
        self.vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
        self.nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
        self.caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        self.pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        self.sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")


        self.caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
        self.caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
        self.source.set_property('device', device)
        self.streammux.set_property('width', 1920)
        self.streammux.set_property('height', 1080)
        self.streammux.set_property('batch-size', 1)
        self.streammux.set_property('batched-push-timeout', 4000000)
        self.pgie.set_property('config-file-path', pgie_config)
        self.sink.set_property('sync', False)

        config = configparser.ConfigParser()
        config.read(tracker_config_path)
        config.sections()

        for key in config['tracker']:
            if key == 'tracker-width' :
                tracker_width = config.getint('tracker', key)
                self.tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height' :
                tracker_height = config.getint('tracker', key)
                self.tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id' :
                tracker_gpu_id = config.getint('tracker', key)
                self.tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file' :
                tracker_ll_lib_file = config.get('tracker', key)
                self.tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file' :
                tracker_ll_config_file = config.get('tracker', key)
                self.tracker.set_property('ll-config-file', tracker_ll_config_file)

        self.pipeline.add(self.source)
        self.pipeline.add(self.caps_v4l2src)
        self.pipeline.add(self.vidconvsrc)
        self.pipeline.add(self.nvvidconvsrc)
        self.pipeline.add(self.caps_vidconvsrc)
        self.pipeline.add(self.streammux)
        self.pipeline.add(self.pgie)
        self.pipeline.add(self.tracker)
        self.pipeline.add(self.nvvidconv)
        self.pipeline.add(self.nvosd)
        self.pipeline.add(self.sink)

        self.source.link(self.caps_v4l2src)
        self.caps_v4l2src.link(self.vidconvsrc)
        self.vidconvsrc.link(self.nvvidconvsrc)
        self.nvvidconvsrc.link(self.caps_vidconvsrc)
        sinkpad = self.streammux.get_request_pad("sink_0")
        srcpad = self.caps_vidconvsrc.get_static_pad("src")
        srcpad.link(sinkpad)
        self.streammux.link(self.pgie)
        self.pgie.link(self.tracker)
        self.tracker.link(self.nvvidconv)
        self.nvvidconv.link(self.nvosd)
        self.nvosd.link(self.sink)

        self.osdsinkpad = self.nvosd.get_static_pad("sink")


    def run(self):

        self.pipeline.set_state(Gst.State.PLAYING)

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect ("message", bus_call, self.loop)

        try:
            self.loop.run()
        except:
            pass

        self.pipeline.set_state(Gst.State.NULL)

class NodePipeline(Node):
    def osd_sink_pad_buffer_probe(self, pad,info,u_data):
        msg = String()
       # bounding_box = BoundingBox2D()
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
                    if obj_meta.class_id == 0: 
                        result = str(x1) + ", " + str(x2) + ", " + str(y1) + ", " + str(y2)
                        msg.data = result
                        self.publisher_.publish(msg)
          
                except StopIteration:
                    break
                obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)
                try: 
                    l_obj=l_obj.next
                except StopIteration:
                    break

            try:
                l_frame=l_frame.next
            except StopIteration:
                break
			
        return Gst.PadProbeReturn.OK


    def __init__(self, pgie_config, device, tracker_config_path):
        super().__init__('inference_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 0)
        Gst.init(None)

        self.pipeline = Gst.Pipeline()

        self.source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
        self.caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
        self.vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
        self.nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
        self.caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        self.pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        self.sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")


        self.caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
        self.caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
        self.source.set_property('device', device)
        self.streammux.set_property('width', 1920)
        self.streammux.set_property('height', 1080)
        self.streammux.set_property('batch-size', 1)
        self.streammux.set_property('batched-push-timeout', 4000000)
        self.pgie.set_property('config-file-path', pgie_config)
        self.sink.set_property('sync', False)

        config = configparser.ConfigParser()
        config.read(tracker_config_path)
        config.sections()

        for key in config['tracker']:
            if key == 'tracker-width' :
                tracker_width = config.getint('tracker', key)
                self.tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height' :
                tracker_height = config.getint('tracker', key)
                self.tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id' :
                tracker_gpu_id = config.getint('tracker', key)
                self.tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file' :
                tracker_ll_lib_file = config.get('tracker', key)
                self.tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file' :
                tracker_ll_config_file = config.get('tracker', key)
                self.tracker.set_property('ll-config-file', tracker_ll_config_file)

        self.pipeline.add(self.source)
        self.pipeline.add(self.caps_v4l2src)
        self.pipeline.add(self.vidconvsrc)
        self.pipeline.add(self.nvvidconvsrc)
        self.pipeline.add(self.caps_vidconvsrc)
        self.pipeline.add(self.streammux)
        self.pipeline.add(self.pgie)
        self.pipeline.add(self.tracker)
        self.pipeline.add(self.nvvidconv)
        self.pipeline.add(self.nvosd)
        self.pipeline.add(self.sink)

        self.source.link(self.caps_v4l2src)
        self.caps_v4l2src.link(self.vidconvsrc)
        self.vidconvsrc.link(self.nvvidconvsrc)
        self.nvvidconvsrc.link(self.caps_vidconvsrc)
        sinkpad = self.streammux.get_request_pad("sink_0")
        srcpad = self.caps_vidconvsrc.get_static_pad("src")
        srcpad.link(sinkpad)
        self.streammux.link(self.pgie)
        self.pgie.link(self.tracker)
        self.tracker.link(self.nvvidconv)
        self.nvvidconv.link(self.nvosd)
        self.nvosd.link(self.sink)

        self.osdsinkpad = self.nvosd.get_static_pad("sink")

        self.osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

    def run(self):

        self.pipeline.set_state(Gst.State.PLAYING)

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect ("message", bus_call, self.loop)

        try:
            self.loop.run()
        except:
            pass

        self.pipeline.set_state(Gst.State.NULL)





class Pipeline_tracker:
    def __init__(self, file_path, config_file, tracker_config_path):
        Gst.init(None)

        self.pipeline = Gst.Pipeline()
        self.source = Gst.ElementFactory.make("filesrc", "file-source")
        self.h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        self.decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        self.pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        self.sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        self.transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

        self.source.set_property('location', file_path)
        self.streammux.set_property('width', 1920)
        self.streammux.set_property('height', 1080)
        self.streammux.set_property('batch-size', 1)
        self.streammux.set_property('batched-push-timeout', 4000000)
        self.pgie.set_property('config-file-path', config_file)
 
        config = configparser.ConfigParser()
        config.read(tracker_config_path)
        config.sections()

        for key in config['tracker']:
            if key == 'tracker-width':
                tracker_width = config.getint('tracker', key)
                self.tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height':
                tracker_height = config.getint('tracker', key)
                self.tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id':
                tracker_gpu_id = config.getint('tracker', key)
                self.tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file':
                tracker_ll_lib_file = config.get('tracker', key)
                self.tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file':
                tracker_ll_config_file = config.get('tracker', key)
                self.tracker.set_property('ll-config-file', tracker_ll_config_file)
            if key == 'enable-batch-process':
                tracker_enable_batch_process = config.getint('tracker', key)
                self.tracker.set_property('enable_batch_process', tracker_enable_batch_process)
            if key == 'enable-past-frame':
                tracker_enable_past_frame = config.getint('tracker', key)
                self.tracker.set_property('enable_past_frame', tracker_enable_past_frame)

        self.pipeline.add(self.source)
        self.pipeline.add(self.h264parser)
        self.pipeline.add(self.decoder)
        self.pipeline.add(self.streammux)
        self.pipeline.add(self.pgie)
        self.pipeline.add(self.tracker)
        self.pipeline.add(self.nvvidconv)
        self.pipeline.add(self.nvosd)
        self.pipeline.add(self.sink)
        self.pipeline.add(self.transform)

        sinkpad = self.streammux.get_request_pad("sink_0")
        srcpad = self.decoder.get_static_pad("src")

        self.source.link(self.h264parser)
        self.h264parser.link(self.decoder)
        srcpad.link(sinkpad)
        self.streammux.link(self.pgie)
        self.pgie.link(self.tracker)
        self.tracker.link(self.nvvidconv)
        self.nvvidconv.link(self.nvosd)
        self.nvosd.link(self.transform)
        self.transform.link(self.sink)

        self.osdsinkpad = self.nvosd.get_static_pad("sink")
        #self.osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

    def run(self):

        self.pipeline.set_state(Gst.State.PLAYING)

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect ("message", bus_call, self.loop)

        try:
            self.loop.run()
        except:
            pass

        self.pipeline.set_state(Gst.State.NULL)


def main(args):
    pipeline = Pipeline(args[1], args[2])
    pipeline.run()

if __name__ == '__main__':
    sys.exit(main(sys.argv))