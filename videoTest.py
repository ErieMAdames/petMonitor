import gi
import io
from threading import Condition

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Define your class to handle streaming output
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            print('asdf')
            self.frame = buf
            self.condition.notify_all()

# Callback to handle appsink data
def on_new_sample(sink, data):
    sample = sink.emit('pull-sample')
    if sample:
        buf = sample.get_buffer()
        result, map_info = buf.map(Gst.MapFlags.READ)
        if result:
            data.write(map_info.data)
            buf.unmap(map_info)
    return Gst.FlowReturn.OK

# Initialize GStreamer
Gst.init(None)

# Create pipeline
pipeline = Gst.parse_launch(
    "libcamerasrc name=source ! video/x-raw, format=RGB, width=1536, height=864 ! "
    "videoconvert ! appsink name=custom_sink"
)

# Get the appsink element
appsink = pipeline.get_by_name('custom_sink')
appsink.set_property('emit-signals', True)
appsink.set_property('sync', False)

# Connect appsink signals
output_handler = StreamingOutput()
appsink.connect('new-sample', on_new_sample, output_handler)

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

# Main loop to keep the application running
try:
    loop = GLib.MainLoop()
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    pipeline.set_state(Gst.State.NULL)
