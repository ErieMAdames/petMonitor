import io
import logging
from http import server
from threading import Condition
import socketserver
from picamera2 import Picamera2, MappedArray

# HTML Page
PAGE = """\
<html>
<head>
<title>Raspberry Pi - Surveillance Camera</title>
</head>
<body>
<center><h1>Raspberry Pi - Surveillance Camera</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
</body>
</html>
"""

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def update_frame(self, frame):
        with self.condition:
            self.frame = frame
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (640, 480), "format": "SRGGB8"})
picam2.configure(camera_config)

output = StreamingOutput()

# Callback for capturing frames
def mjpeg_callback(request):
    with MappedArray(request, "main") as m:
        # Convert frame to JPEG and store in the output object
        output.update_frame(m.array.tobytes())

picam2.start_recording(mjpeg_callback)

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    print("Streaming on http://<your_pi_ip>:8000")
    server.serve_forever()
finally:
    picam2.stop_recording()
