#!/usr/bin/python3
from threading import Thread
import json
import cv2
import numpy as np
from picamera2 import Picamera2
import asyncio
import websockets
import base64
import time
import requests

# Color range for brown (dog kibble) in HSV space
lower_brown = np.array([10, 100, 100])  # Lower bound of brown color in HSV
upper_brown = np.array([20, 255, 255])  # Upper bound of brown color in HSV

async def websocket_handler(websocket):
    async for message in websocket:
        data = json.loads(message)
        if data.get("food", None) == 'food':
            image = picam2.capture_array()
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_brown = np.array([10, 50, 30])  
            upper_brown = np.array([30, 255, 200])
            circle_center = (int(image.shape[1] / 2), int(image.shape[0] / 2))
            circle_radius = 125  # Adjust the radius as needed
            cv2.circle(image, circle_center, circle_radius, (0, 255, 0), 2)
            mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
            circle_mask = np.zeros_like(mask)
            cv2.circle(circle_mask, circle_center, circle_radius, 255, thickness=-1)
            masked_img = cv2.bitwise_and(mask, mask, mask=circle_mask)
            contours, _ = cv2.findContours(masked_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            output_image = image.copy()
            min_area = 30
            for i, contour in enumerate(sorted_contours):
                if cv2.contourArea(contour) > min_area:
                    cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)
            _, jpeg = cv2.imencode('.jpg', output_image)
            img_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            response = json.dumps({"image": img_base64})
            await websocket.send(response)

async def start_websocket_server_monitor():
    async with websockets.serve(websocket_handler, "0.0.0.0", 8500):
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
config = picam2.create_preview_configuration(main={"format":"XRGB8888"})
picam2.configure(config)
picam2.start()

try:
    asyncio.run(start_websocket_server_monitor())
finally:
    picam2.stop()
