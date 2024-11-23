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
from adc import ADC
from ultrasonic import Ultrasonic
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from gpiozero import Button
from motor import Motor
import asyncio
import websockets
import base64
import time
from hailo_platform import HEF, VDevice
# Hailo Initialization

vdevice = VDevice()  # Virtual Device
hef = HEF("/usr/share/hailo-models/yolov6n.hef")  # Replace with your .hef model path
network_group = vdevice.configure(hef)[0]
input_vstream_info = network_group.get_input_vstream_infos()[0]
output_vstream_info = network_group.get_output_vstream_infos()[0]

button = Button(16)
counter = 0
last_pressed_time = 0
motor = Motor()

PAGE = ''
with open('index.html', 'r') as f:
    PAGE = f.read()

servo0_angle_offset = 0
servo1_angle_offset = -8
servo0_angle = 0
servo1_angle = 0 
servo0 = Servo(PWM("P0"))
servo1 = Servo(PWM("P1"))
water_monitor = ADC(0)

servo0.set_angle(servo0_angle_offset)
servo1.set_angle(servo1_angle_offset)
shadow_brightness = 50
habichuela_brightness = 50
ultrasonic = Ultrasonic()

def preprocess_frame(frame, input_shape):
    """Resize and normalize the frame for Hailo model input."""
    resized = cv2.resize(frame, (input_shape.width, input_shape.height))
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
def on_button_press():
    global counter, last_pressed_time
    current_time = time.time()
    if current_time - last_pressed_time >= 1:
        print("Button pressed " + str(counter) + ' times')
        motor.set_power(1)
        time.sleep(.25)
        motor.set_power(0)
        counter += 1
        last_pressed_time = current_time

# Attach the function to the button press event
button.when_pressed = on_button_press
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

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
def find_poop(image, brightness = 50):
    image = increase_brightness(image, brightness)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    detected = False
    for contour in sorted_contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
                    detected = True
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
                    cv2.rectangle(image, (dx_min_full, dy_min_full), (dx_max_full, dy_max_full), (0, 0, 255), 1)
    return image, detected
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
        global shadow_brightness
        global habichuela_brightness
        data = json.loads(message)
        if data.get("pet", None) == 'shadow':
            img = picam2_dog_monitor.capture_array()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, detected = find_poop(img, shadow_brightness)
            _, jpeg = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            response = json.dumps({"pet": "shadow", "image": img_base64, "detected": detected})
            await websocket.send(response)
        if data.get("pet", None) == 'habichuela':
            img = picam2_cat_monitor.capture_array()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, detected = find_poop(img, habichuela_brightness)
            _, jpeg = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            response = json.dumps({"pet": "habichuela", "image": img_base64, "detected": detected})
            await websocket.send(response)
        if data.get("slider", None) == 'shadow':
            shadow_brightness = int(data.get('value', 50))
        if data.get("slider", None) == 'habichuela':
            habichuela_brightness = int(data.get('value', 50))
        if data.get("water_level", None) == 'water_level':
            water_level = water_monitor.read()
            response = json.dumps({"water_level": water_level})
            await websocket.send(response)
        if data.get("food_level", None) == 'food_level':
            food_level = ultrasonic.get_distance()
            response = json.dumps({"food_level": food_level})
            await websocket.send(response)
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
