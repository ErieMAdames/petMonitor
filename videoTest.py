import io
import socketserver
from http import server
import cv2
import numpy as np
from picamera2 import Picamera2, MjpegEncoder
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
import random
import time
import os
from hailo_model_zoo.core.postprocessing.detection import yolo
import yolox_stream_report_detections as report

# Constants
INPUT_RES_H = 640
INPUT_RES_W = 640
MODEL_NAME = 'yolox_s_leaky'
HEF_PATH = "/usr/share/hailo-models/yolov6n.hef"

# Initialize Hailo HEF and network group
hef = HEF(HEF_PATH)
devices = Device.scan()

with VDevice(device_ids=devices) as target:
    # Configure the device and network
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

    # Camera setup using Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.encoder = MjpegEncoder()
    
    output = StreamingOutput()

    picam2.start_recording(output)

    # Initialize the HTTP server
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

    class StreamingOutput:
        """MJPEG streaming output."""
        def __init__(self):
            self.frame = None
            self.condition = Condition()

        def write(self, buf):
            with self.condition:
                self.frame = buf
                self.condition.notify_all()

        def flush(self):
            pass  # Required for compatibility

    class StreamingHandler(server.BaseHTTPRequestHandler):
        """HTTP handler for MJPEG stream."""
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

                        np_frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)

                        # Preprocess frame for inference
                        resized_img = cv2.resize(np_frame, (INPUT_RES_H, INPUT_RES_W), interpolation=cv2.INTER_AREA)
                        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                            input_data = {input_vstream_info.name: np.expand_dims(np.asarray(resized_img), axis=0).astype(np.float32)}
                            with network_group.activate(network_group_params):
                                infer_results = infer_pipeline.infer(input_data)

                        # Postprocessing: Detection
                        layer_from_shape = {infer_results[key].shape: key for key in infer_results.keys()}
                        anchors = {"strides": [32, 16, 8], "sizes": [[1, 1], [1, 1], [1, 1]]}
                        yolox_post_proc = yolo.YoloPostProc(img_dims=(INPUT_RES_H, INPUT_RES_W), nms_iou_thresh=0.65, 
                                                            score_threshold=0.01, anchors=anchors, output_scheme=None, 
                                                            classes=80, labels_offset=1, meta_arch="yolox", 
                                                            device_pre_post_layers=[])

                        endnodes = [infer_results[layer_from_shape[1, 80, 80, 4]],  
                                    infer_results[layer_from_shape[1, 80, 80, 1]],  
                                    infer_results[layer_from_shape[1, 80, 80, 80]], 
                                    infer_results[layer_from_shape[1, 40, 40, 4]],  
                                    infer_results[layer_from_shape[1, 40, 40, 1]],  
                                    infer_results[layer_from_shape[1, 40, 40, 80]],  
                                    infer_results[layer_from_shape[1, 20, 20, 4]],  
                                    infer_results[layer_from_shape[1, 20, 20, 1]],  
                                    infer_results[layer_from_shape[1, 20, 20, 80]]]
                        hailo_preds = yolox_post_proc.yolo_postprocessing(endnodes)
                        num_detections = int(hailo_preds['num_detections'])
                        scores = hailo_preds["detection_scores"][0].numpy()
                        classes = hailo_preds["detection_classes"][0].numpy()
                        boxes = hailo_preds["detection_boxes"][0].numpy()

                        if scores[0] == 0:
                            num_detections = 0

                        preds_dict = {'scores': scores, 'classes': classes, 'boxes': boxes, 'num_detections': num_detections}
                        frame = report.report_detections(preds_dict, np_frame, scale_factor_x=np_frame.shape[1], scale_factor_y=np_frame.shape[0])

                        # Encode the processed frame
                        _, jpeg = cv2.imencode('.jpg', frame)

                        # Send MJPEG frame
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(jpeg))
                        self.end_headers()
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b'\r\n')
                except Exception as e:
                    print(f"Streaming error: {str(e)}")

            else:
                self.send_error(404)
                self.end_headers()

    class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
        """HTTP server for MJPEG stream."""
        allow_reuse_address = True
        daemon_threads = True

    # Start MJPEG server
    address = ('', 8765)
    server = StreamingServer(address, StreamingHandler)
    print("Starting server on port 8765...")
    server.serve_forever()

finally:
    picam2.stop_recording()
    network_group.deactivate()  # Clean up Hailo resources
