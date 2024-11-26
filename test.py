#!/usr/bin/python3
#!/usr/bin/python3
import io
import logging
import socketserver
from http import server
from threading import Condition, Thread
import json
import cv2
import numpy as np
import sounddevice as sd
from servo import Servo
from pwm import PWM
from adc import ADC
from ultrasonic import Ultrasonic
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from gpiozero import Button
from motor import Motor
from playsound import playsound
import asyncio
import websockets
import base64
import time
import libcamera
import sqlite3
import requests
import schedule
from datetime import datetime

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 960)})
picam2.configure(config)
output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))
picam2.stop()
picam2_shadow_monitor = Picamera2(1)
picam2_shadow_monitor.start()
picam2_shadow_monitor.stop()
picam2_habichuela_monitor = Picamera2(3)
picam2_habichuela_monitor.start()
picam2_habichuela_monitor.stop()
picam2_shadow_food = Picamera2(2)
picam2_shadow_food.start()
picam2_shadow_food.stop()