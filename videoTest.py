import io
import socketserver
from http import server
import cv2
import numpy as np
from threading import Condition
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
from yolo import YoloPostProc
import logging
from pprint import pprint

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
hef = HEF("/usr/share/hailo-models/yolov6n.hef")  # Replace with your .hef model path

INPUT_RES_H = 640
INPUT_RES_W = 640

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
                devices = Device.scan()
                with VDevice(device_ids=devices) as target:
                    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
                    network_group = target.configure(hef, configure_params)[0]
                    network_group_params = network_group.create_params()
                    input_vstream_info = hef.get_input_vstream_infos()[0]
                    output_vstream_info = hef.get_output_vstream_infos()[0]
                    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
                    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
                    height, width, channels = hef.get_input_vstream_infos()[0].shape
                    while True:
                        with output.condition:
                            output.condition.wait()
                            frame = output.frame

                        # Decode frame for processing
                        np_frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                        resized_img = cv2.resize(np_frame, (INPUT_RES_H, INPUT_RES_H), interpolation = cv2.INTER_AREA)
                        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                            input_data = {input_vstream_info.name: np.expand_dims(np.asarray(resized_img), axis=0).astype(np.float32)}    
                            with network_group.activate(network_group_params):
                                infer_results = infer_pipeline.infer(input_data)
                        for ir in infer_results['yolov6n/yolox_nms_postprocess']:
                            try:
                                pprint(ir.shape)
                            except Exception:
                                continue
                        layer_from_shape: dict = {infer_results[key].shape:key for key in infer_results.keys()}
                        
                        pprint(layer_from_shape)
                        # postprocessing info for constructor as recommended in hailo_model_zoo/cfg/base/yolox.yaml
                        anchors = {"strides": [32, 16, 8], "sizes": [[1, 1], [1, 1], [1, 1]]}
                        yolox_post_proc = YoloPostProc(img_dims=(INPUT_RES_H,INPUT_RES_H), nms_iou_thresh=0.65, score_threshold=0.01, 
                                                            anchors=anchors, output_scheme=None, classes=80, labels_offset=1, 
                                                            meta_arch="yolox", device_pre_post_layers=[])                

                        # Order of insertion matters since we need the reorganized tensor to be in (BS,H,W,85) shape
                        endnodes = [infer_results[layer_from_shape[1, 80, 80, 4]],  # stride 8 
                                    infer_results[layer_from_shape[1, 80, 80, 1]],  # stride 8 
                                    infer_results[layer_from_shape[1, 80, 80, 80]], # stride 8 
                                    infer_results[layer_from_shape[1, 40, 40, 4]],  # stride 16
                                    infer_results[layer_from_shape[1, 40, 40, 1]],  # stride 16
                                    infer_results[layer_from_shape[1, 40, 40, 80]], # stride 16
                                    infer_results[layer_from_shape[1, 20, 20, 4]],  # stride 32
                                    infer_results[layer_from_shape[1, 20, 20, 1]],  # stride 32
                                    infer_results[layer_from_shape[1, 20, 20, 80]]  # stride 32
                                ]
                        hailo_preds = yolox_post_proc.yolo_postprocessing(endnodes)
                        num_detections = int(hailo_preds['num_detections'])
                        scores = hailo_preds["detection_scores"][0].numpy()
                        classes = hailo_preds["detection_classes"][0].numpy()
                        boxes = hailo_preds["detection_boxes"][0].numpy()
                        if scores[0] == 0:
                            num_detections = 0
                        preds_dict = {'scores': scores, 'classes': classes, 'boxes': boxes, 'num_detections': num_detections}
                        print(preds_dict)
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
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
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 960)}))
picam2.start_recording(JpegEncoder(), FileOutput(output))

try:
    address = ('', 8765)
    server = StreamingServer(address, StreamingHandler)
    print("Starting server on port 8765...")
    server.serve_forever()
finally:
    picam2.stop_recording()
