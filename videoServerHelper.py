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

            # # Convert the image to HSV color space for better color detection
            # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Define the center and radius of the circle

            # # Find contours of the detected brown areas
            # contours, _ = cv2.findContours(masked_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # # Check if any contours were found (indicating brown kibble is detected)
            # brown_detected = False
            # for contour in contours:
            #     if cv2.contourArea(contour) > 500:  # Only consider large enough contours
            #         brown_detected = True
            #         # Draw a red outline around the detected brown area
            #         cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)  # Red outline

            # # Alert if brown kibble is detected or not
            # if brown_detected:
            #     print("Brown kibble detected inside the circle!")
            # else:
            #     print("No brown kibble detected inside the circle.")

            # # Convert the image back to BGR for sending to the websocket
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            circle_center = (int(image.shape[1] / 2), int(image.shape[0] / 2))
            circle_radius = 100  # Adjust the radius as needed

            # Draw a green circle on the image
            cv2.circle(image, circle_center, circle_radius, (0, 255, 0), 2)

            # Create a mask for the brown color inside the circle
            mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
            circle_mask = np.zeros_like(mask)
            cv2.circle(circle_mask, circle_center, circle_radius, 255, thickness=-1)
            masked_img = cv2.bitwise_and(mask, mask, mask=circle_mask)
            lower_brown = np.array([5, 50, 20])
            upper_brown = np.array([30, 255, 200])
            brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            clean_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            output_image = image.copy()
            for i, contour in enumerate(sorted_contours[:3]):
                cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 1)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
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
