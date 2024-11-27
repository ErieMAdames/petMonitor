#!/usr/bin/python3
import cv2
import numpy as np
from picamera2 import Picamera2
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format":"XRGB8888"})
picam2.configure(config)
picam2.start()

# Color range for brown (dog kibble) in HSV space
lower_brown = np.array([10, 30, 40])
upper_brown = np.array([30, 200, 200])

@app.route('/detect_food', methods=['POST'])
def detect_food():
    # Receive JSON data from the POST request
    data = request.get_json()
    if data.get("food", None) == 'food':
        image = picam2.capture_array()
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
        food_out = True
        for i, contour in enumerate(sorted_contours):
            if cv2.contourArea(contour) > min_area:
                food_out = False
                cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', output_image)
        img_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        response = {
            "image": img_base64,
            "food_out": food_out
        }
        return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8500, threaded=True)
