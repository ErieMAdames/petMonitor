import io
import socketserver
from http import server
import cv2
import numpy as np
from threading import Condition
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from hailo_platform import HEF, VDevice  # Hailo imports
import logging

PAGE = """\
<html>
<head>
<title>PiCam2 Stream with Hailo AI</title>
</head>
<body>
<h1>PiCam2 Stream with Hailo AI Object Detection</h1>
<img src="stream.mjpg" width="640" height="480">
</body>
</html>
"""

# Hailo Initialization
vdevice = VDevice()  # Virtual Device
hef = HEF("/usr/share/hailo-models/yolov6n.hef")  # Replace with your .hef model path
network_group = vdevice.configure(hef)[0]

network_group.activate()

# Create VStreams for input and output
vstreams = vdevice.create_vstreams(network_group)
input_vstream = vstreams[0]
output_vstream = vstreams[1]

def preprocess_frame(frame, input_shape):
    """Resize and normalize the frame for Hailo model input."""
    print('error here', frame.shape, input_shape)
    resized = cv2.resize(frame, (input_shape[0], input_shape[1]))
    normalized = resized.astype(np.float32) / 255.0  # Normalize to 0-1 range
    return np.expand_dims(normalized, axis=0)  # Add batch dimension

def postprocess_detections(output, original_shape):
    """Parse Hailo output into human-readable detections."""
    detections = []
    for detection in output:
        x, y, w, h = detection[:4]
        label = int(detection[5])  # Class ID
        confidence = detection[4]
        # Scale back to original image size
        x *= original_shape[1]
        y *= original_shape[0]
        w *= original_shape[1]
        h *= original_shape[0]
        detections.append({"bbox": (int(x), int(y), int(w), int(h)), "confidence": confidence, "label": label})
    return detections

def overlay_detections(frame, detections):
    """Draw bounding boxes and labels on the frame."""
    for detection in detections:
        x, y, w, h = detection["bbox"]
        confidence = detection["confidence"]
        label = detection["label"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Label: {label} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

class StreamingOutput(io.BufferedIOBase):
    """Thread-safe buffer to store the latest frame."""
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    """Handles HTTP requests for the video stream."""
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(PAGE.encode('utf-8'))
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

                    # Decode frame for processing
                    np_frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)

                    # Preprocess frame for inference
                    input_tensor = preprocess_frame(np_frame, input_vstream.shape)

                    # Run inference using VStreams
                    input_vstream.write(input_tensor)
                    output_tensor = output_vstream.read()


                    # Post-process detections
                    detections = postprocess_detections(output_tensor, np_frame.shape)


                    # Overlay detections
                    processed_frame = overlay_detections(np_frame, detections)

                    # Encode frame as JPEG
                    _, jpeg = cv2.imencode('.jpg', processed_frame)

                    # Send frame to client
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(jpeg))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Streaming client error: %s', str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """HTTP server to serve the video stream."""
    allow_reuse_address = True
    daemon_threads = True

output = StreamingOutput()

# Camera Initialization
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start_recording(JpegEncoder(), FileOutput(output))

try:
    address = ('', 8765)
    server = StreamingServer(address, StreamingHandler)
    print("Starting server on port 8765...")
    server.serve_forever()
finally:
    picam2.stop_recording()
