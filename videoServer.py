#!/usr/bin/python3

import io
import logging
import socketserver
from http import server
from threading import Condition, Thread
import json
import cv2
import numpy as np
from servo import Servo
from pwm import PWM
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import asyncio
import websockets

PAGE = """\
<html>
<head>
<title>Video Stream</title>
<script>
let ws;

function initWebSocket() {
    ws = new WebSocket('ws://' + window.location.hostname + ':8765/ws');
    ws_images = new WebSocket('ws://' + window.location.hostname + ':8744/ws');
    ws_images.binaryType = "arraybuffer";
    ws.onopen = () => console.log("WebSocket connection established");
    ws.onclose = () => console.log("WebSocket connection closed");
    ws_images.onopen = () => console.log("WebSocket for Shadow & Habichuela connection established");
    ws_images.onclose = () => console.log("WebSocket for Shadow & Habichuela connection closed");
    ws_images.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
            // 4. Convert the ArrayBuffer to a Blob
            const blob = new Blob([event.data], { type: "image/jpeg" }); // Adjust type as needed

            // 5. Create an image URL from the Blob
            const imageUrl = URL.createObjectURL(blob);

            // 6. Display the image
            const img = document.createElement("img");
            img.src = imageUrl;
            document.body.appendChild(img);
        }
    };
}

function handleKey(event) {
    const key = event.key.toLowerCase();
    if (['w', 'a', 's', 'd'].includes(key)) {
        ws.send(JSON.stringify({ key: key, action: event.type }));
    }

    if (['p'].includes(key)) {
        ws_images.send(JSON.stringify({ dog: "dog" }));
    }
}

window.onload = () => {
    initWebSocket();
    document.addEventListener('keydown', handleKey);
    document.addEventListener('keyup', handleKey);
};

</script>
</head>
<body>
<h1>Picamera2 MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="1280" height="960" style="transform: rotate(180deg);"/>
</body>
</html>
"""
servo0_angle_offset = 0
servo1_angle_offset = -8
servo0_angle = 0
servo1_angle = 0 
servo0 = Servo(PWM("P0"))
servo1 = Servo(PWM("P1"))

servo0.set_angle(servo0_angle_offset)
servo1.set_angle(servo1_angle_offset)

def up():
    global servo1_angle_offset
    global servo1_angle
    servo1_angle = max(servo1_angle - 2, -90)
    servo1.set_angle(servo1_angle + servo1_angle_offset)
def down():
    global servo1_angle_offset
    global servo1_angle
    servo1_angle = min(servo1_angle + 2 , 90)
    servo1.set_angle(servo1_angle + servo1_angle_offset)
def right():
    global servo0_angle_offset
    global servo0_angle
    servo0_angle = max(servo0_angle - 2, -90)
    servo0.set_angle(servo0_angle + servo0_angle_offset)
def left():
    global servo0_angle_offset
    global servo0_angle
    servo0_angle = min(servo0_angle + 2, 90)
    servo0.set_angle(servo0_angle + servo0_angle_offset)

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
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

def draw_red_boxes_around_dark_areas(image_path, output_path="output.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found. Please check the path.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    for contour in sorted_contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(image, [contour], 0, (0,255,0), 4)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            masked_image = cv2.bitwise_and(gray, gray, mask=mask)
            gray[np.where(masked_image == 0)] = 255
            bright_area = 255 - gray[y:y+h, x:x+w]
            _, dark_thresholded = cv2.threshold(bright_area, 100, 255, cv2.THRESH_BINARY) 
            dark_contours, _ = cv2.findContours(dark_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dark_contours = sorted(dark_contours, key=cv2.contourArea, reverse=True) 
            for dark_contour in dark_contours:
                if cv2.contourArea(dark_contour) > 100:
                    dx, dy, dw, dh = cv2.boundingRect(dark_contour)
                    cv2.drawContours(image, [dark_contour], 0, (0,0,255), 4, offset=( x , y ))
                    offset = 10 
                    dx_min = max(dx - offset, 0)
                    dy_min = max(dy - offset, 0)
                    dx_max = min(dx + dw + offset, w)
                    dy_max = min(dy + dh + offset, h)
                    dx_min_full = x + dx_min
                    dy_min_full = y + dy_min
                    dx_max_full = x + dx_max
                    dy_max_full = y + dy_max
                    cv2.rectangle(image, (dx_min_full, dy_min_full), (dx_max_full, dy_max_full), (0, 0, 255), 2)
async def websocket_camera_movement_handler(websocket):
    async for message in websocket:
        data = json.loads(message)
        key = data.get("key")
        action = data.get("action")
        if key and action:
            if key == 'w':
                up()
            if key == 'a':
                left()
            if key == 'd':
                right()
            if key == 's':
                down()
async def websocket_poop_handler(websocket):
    async for message in websocket:
        data = json.loads(message)
        print('data')
        try:
            if data.get("dog", None) is not None:
                print(data)
                img = picam2_dog_monitor.capture_array()
                print('captured')
                _, jpeg = cv2.imencode('.jpg', img)
                print('jpeg')
                return jpeg.tobytes()
        except Exception as e:
            print(e)


async def start_websocket_server():
    async with websockets.serve(websocket_camera_movement_handler, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever
async def start_websocket_server_poop_monitor():
    async with websockets.serve(websocket_poop_handler, "0.0.0.0", 8744):
        await asyncio.Future()  # Run forever
def run_websocket_server_in_thread(coroutine):
    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coroutine)
    
    thread = Thread(target=run_loop)
    thread.daemon = True
    thread.start()
    return thread

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 960)}))
output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))
picam2_dog_monitor = Picamera2(1)
picam2_dog_monitor.start()
picam2_cat_monitor = Picamera2(2)
picam2_cat_monitor.start()
try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    http_server_thread = Thread(target=server.serve_forever)
    http_server_thread.daemon = True
    http_server_thread.start()

    run_websocket_server_in_thread(start_websocket_server())
    asyncio.run(start_websocket_server_poop_monitor())
finally:
    picam2.stop_recording()
