#!/usr/bin/python3

# How to do digital zoom using the "ScalerCrop" control.

import time

from picamera2 import Picamera2, Preview

picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)

preview_config = picam2.create_preview_configuration()
picam2.configure(preview_config)

picam2.start()
time.sleep(2)

size = picam2.capture_metadata()['ScalerCrop'][2:]

full_res = picam2.camera_properties['PixelArraySize']

for _ in range(20):
    # This syncs us to the arrival of a new camera frame:
    picam2.capture_metadata()

    size = [int(s * 0.95) for s in size]
    offset = [(r - s) // 2 for r, s in zip(full_res, size)]
    picam2.set_controls({"ScalerCrop": offset + size})

time.sleep(2)