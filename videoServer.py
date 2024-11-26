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


DB_PATH = "pet_monitor.db"
SAMPLERATE = 16000  # Sampling rate (Hz)
CHUNK_SIZE = 1024   # Number of audio frames per chunk
LOUDNESS_THRESHOLD = 0.5  # RMS value threshold for loud sounds
DEVICE_INDEX = 1  # Replace with your device index, or leave None for default
CHANNELS = 2  # Use 2 if your microphone supports only stereo
MUSIC_FILE = "relaxing.mp3"
bark_detected = False
rms = 0
shadow_pooped = False
shadow_poop_start_time = None
shadow_poop_clean_time = None
habichuela_pooped = False
habichuela_poop_start_time = None
habichuela_poop_clean_time = None
water_ran_out = False
water_refilled = True
food_ran_out = False
food_refilled = True
DETECTION_DURATION_THRESHOLD = 10  # 2 minutes in seconds
zoom_level_main = 1.0
zoom_level_shadow = 1.0
zoom_level_habichuela = 1.0
button = Button(16)
counter = 0
last_pressed_time = 0
motor = Motor()

PAGE = ''
with open('index.html', 'r') as f:
    PAGE = f.read()
pushover_user_key = ''
with open('pushoverUserKey.txt', 'r') as f:
    pushover_user_key = f.read().strip()
pushover_token = ''
with open('pushoverToken.txt', 'r') as f:
    pushover_token = f.read().strip()

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
def send_notification(title, message):
    global pushover_token
    global pushover_user_key
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": pushover_token,
            "user": pushover_user_key,
            "message": message,
            "title": title,
        }
    )

def create_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS food_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            event TEXT NOT NULL -- 'out' or 'refill'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS water_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            event TEXT NOT NULL -- 'out' or 'refill'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            type TEXT NOT NULL, -- 'poop_cat', 'poop_dog', 'bark'
            loudness REAL -- Optional, for barking events
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS camera_settings (
            camera_id TEXT PRIMARY KEY ,
            brightness REAL NOT NULL,
            zoom REAL NOT NULL,
            x REAL, -- Optional, for main cam position
            y REAL -- Optional, for main cam position
        )
    """)
    conn.commit()
    conn.close()
def log_food_event(event):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO food_logs (timestamp, event) VALUES (?, ?)", 
                   (datetime.now(), event))
    conn.commit()
    conn.close()

def log_water_event(event):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO water_logs (timestamp, event) VALUES (?, ?)", 
                   (datetime.now(), event))
    conn.commit()
    conn.close()

def log_activity(event_type, loudness=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO activity_logs (timestamp, type, loudness) VALUES (?, ?, ?)", 
                   (datetime.now(), event_type, loudness))
    conn.commit()
    conn.close()

def save_camera_settings(camera_id, brightness, zoom, x=0, y=0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO camera_settings (camera_id, brightness, zoom, x, y)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(camera_id) DO UPDATE SET brightness=?, zoom=?, x=?, y=?
    """, (camera_id, brightness, zoom, x, y, brightness, zoom, x, y))
    conn.commit()
    conn.close()

# Query data functions
def get_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM food_logs WHERE event='out' ORDER BY timestamp DESC LIMIT 1")
    food_out = cursor.fetchone()
    cursor.execute("SELECT timestamp FROM food_logs WHERE event='refill' ORDER BY timestamp DESC LIMIT 1")
    food_refilled = cursor.fetchone()
    cursor.execute("SELECT timestamp FROM water_logs WHERE event='out' ORDER BY timestamp DESC LIMIT 1")
    water_out = cursor.fetchone()
    cursor.execute("SELECT timestamp FROM water_logs WHERE event='refill' ORDER BY timestamp DESC LIMIT 1")
    water_refilled = cursor.fetchall()
    cursor.execute("SELECT timestamp FROM food_logs WHERE event='treat' ORDER BY timestamp DESC LIMIT 5")
    treats = cursor.fetchall()
    cursor.execute("SELECT timestamp FROM activity_logs WHERE type='habichuela pooped' ORDER BY timestamp DESC LIMIT 1")
    habichuela_pooped = cursor.fetchone()
    cursor.execute("SELECT timestamp FROM activity_logs WHERE type='habichuela poop cleaned' ORDER BY timestamp DESC LIMIT 1")
    habichuela_poop_cleaned = cursor.fetchone()
    cursor.execute("SELECT timestamp FROM activity_logs WHERE type='shadow pooped' ORDER BY timestamp DESC LIMIT 1")
    shadow_pooped = cursor.fetchone()
    cursor.execute("SELECT timestamp FROM activity_logs WHERE type='shadow poop cleaned' ORDER BY timestamp DESC LIMIT 1")
    shadow_poop_cleaned = cursor.fetchone()
    cursor.execute("SELECT timestamp FROM activity_logs WHERE type='bark detected' ORDER BY timestamp DESC LIMIT 5")
    barks = cursor.fetchall()
    conn.close()
    return {
        'food_out': food_out,
        'food_refilled': food_refilled,
        'water_out': water_out,
        'water_refilled': water_refilled,
        'treats': treats,
        'habichuela_pooped': habichuela_pooped,
        'habichuela_poop_cleaned': habichuela_poop_cleaned,
        'shadow_pooped': shadow_pooped,
        'shadow_poop_cleaned': shadow_poop_cleaned,
        'barks': barks,
    }

def get_habichuela_num_poops():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM activity_logs WHERE type='habichuela pooped'")
    habichuela_num_poops = cursor.fetchone()[0]
    conn.close()
    return habichuela_num_poops

def get_camera_settings():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM camera_settings")
    rows = cursor.fetchall()
    conn.close()
    return rows

def on_button_press():
    global counter, last_pressed_time
    current_time = time.time()
    if current_time - last_pressed_time >= 1:
        motor.set_power(30)
        time.sleep(.15)
        motor.set_power(0)
        counter += 1
        last_pressed_time = current_time
        log_food_event('treat')
        send_notification('Shadow took a treat', 'Shadow took a treat')
        

# Attach the function to the button press event
button.when_pressed = on_button_press
def up():
    global servo1_angle_offset
    global servo1_angle
    global servo0_angle
    global zoom_level_main
    servo1_angle = max(servo1_angle - 2, -90)
    servo1.set_angle(servo1_angle + servo1_angle_offset)
    save_camera_settings('main', 0, zoom_level_main, servo0_angle, servo1_angle)
def down():
    global servo1_angle_offset
    global servo1_angle
    global servo0_angle
    global zoom_level_main
    servo1_angle = min(servo1_angle + 2 , 90)
    servo1.set_angle(servo1_angle + servo1_angle_offset)
    save_camera_settings('main', 0, zoom_level_main, servo0_angle, servo1_angle)
def left():
    global servo0_angle_offset
    global servo0_angle
    global servo1_angle
    global zoom_level_main
    servo0_angle = max(servo0_angle - 2, -90)
    servo0.set_angle(servo0_angle + servo0_angle_offset)
    save_camera_settings('main', 0, zoom_level_main, servo0_angle, servo1_angle)
def right():
    global servo0_angle_offset
    global servo0_angle
    global servo0_angle_offset
    global servo0_angle
    global servo1_angle
    global zoom_level_main
    servo0_angle = min(servo0_angle + 2, 90)
    servo0.set_angle(servo0_angle + servo0_angle_offset)
    save_camera_settings('main', 0, zoom_level_main, servo0_angle, servo1_angle)

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
        global zoom_level_shadow
        global zoom_level_habichuela
        global zoom_level_main
        global size
        global full_res
        global shadow_pooped
        global shadow_poop_start_time
        global shadow_poop_clean_time
        global habichuela_pooped
        global habichuela_poop_start_time
        global habichuela_poop_clean_time
        global water_ran_out
        global water_refilled
        global food_ran_out
        global food_refilled
        data = json.loads(message)
        if data.get("pet", None) == 'shadow':
            img = picam2_shadow_monitor.capture_array()
            if zoom_level_shadow > 1:
                height, width = img.shape[:2]
                new_width = int(width / zoom_level_shadow)
                new_height = int(height / zoom_level_shadow)
                x1 = (width - new_width) // 2
                y1 = (height - new_height) // 2
                x2 = x1 + new_width
                y2 = y1 + new_height
                img = img[y1:y2, x1:x2]
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, detected = find_poop(img, shadow_brightness)
            if detected:
                if not shadow_pooped and shadow_poop_clean_time is None:
                    if shadow_poop_start_time is None:
                        shadow_poop_start_time = time.time()
                    elif time.time() - shadow_poop_start_time >= DETECTION_DURATION_THRESHOLD:
                        log_activity("shadow pooped")
                        shadow_pooped = True
                        send_notification('Shadow pooped', 'Shadow pooped, clean up the puppy pad')
            else:
                if shadow_pooped:
                    if shadow_poop_clean_time is None:
                        shadow_poop_clean_time = time.time()
                    elif time.time() - shadow_poop_clean_time >= DETECTION_DURATION_THRESHOLD:
                        log_activity("shadow poop cleaned")
                        send_notification('Shadow poop cleaned', 'Good job on cleaning up')
                        shadow_poop_start_time = None
                        shadow_poop_clean_time = None
                        shadow_pooped = False
            _, jpeg = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            response = json.dumps({"pet": "shadow", "image": img_base64, "detected": detected})
            await websocket.send(response)
        if data.get("pet", None) == 'habichuela':
            img = picam2_habichuela_monitor.capture_array()
            if zoom_level_habichuela > 1:
                height, width = img.shape[:2]
                new_width = int(width / zoom_level_habichuela)
                new_height = int(height / zoom_level_habichuela)
                x1 = (width - new_width) // 2
                y1 = (height - new_height) // 2
                x2 = x1 + new_width
                y2 = y1 + new_height
                # Crop and resize the image
                img = img[y1:y2, x1:x2]
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, detected = find_poop(img, habichuela_brightness)
            if detected:
                if not habichuela_pooped and habichuela_poop_clean_time is None:
                    if habichuela_poop_start_time is None:
                        habichuela_poop_start_time = time.time()
                    elif time.time() - habichuela_poop_start_time >= DETECTION_DURATION_THRESHOLD:
                        log_activity("habichuela pooped")
                        send_notification('Habichuela pooped', 'Habichuela pooped, add some baking soda to the litterbox')
                        habichuela_pooped = True
                        habichuela_num_poops = get_habichuela_num_poops()
                        if (habichuela_num_poops > 15):
                            send_notification('Go Change the litter box', 'Habichuela needs new litter')
            else:
                if habichuela_pooped:
                    if habichuela_poop_clean_time is None:
                        habichuela_poop_clean_time = time.time()
                    elif time.time() - habichuela_poop_clean_time >= DETECTION_DURATION_THRESHOLD:
                        log_activity("habichuela poop cleaned")
                        send_notification('Habichuela poop cleaned', "Good job on cleaning up, now it won't smell like cat turds")
                        habichuela_poop_start_time = None
                        habichuela_poop_clean_time = None
                        habichuela_pooped = False
            _, jpeg = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            response = json.dumps({"pet": "habichuela", "image": img_base64, "detected": detected})
            await websocket.send(response)
        if data.get("action", None) == 'brightness':
            if data.get("slider", None) == 'shadow':
                shadow_brightness = int(data.get('value', 50))
                save_camera_settings('shadow', shadow_brightness, zoom_level_shadow)
            if data.get("slider", None) == 'habichuela':
                habichuela_brightness = int(data.get('value', 50))
                save_camera_settings('habichuela', habichuela_brightness, zoom_level_habichuela)
        if data.get("action", None) == 'zoom':
            if data.get("slider", None) == 'shadow':
                zoom_level_shadow = 1 + (int(data.get('value', 0)) / 100)
                save_camera_settings('shadow', shadow_brightness, zoom_level_shadow)
            if data.get("slider", None) == 'habichuela':
                zoom_level_habichuela = 1 + (int(data.get('value', 0)) / 100)
                save_camera_settings('habichuela', habichuela_brightness, zoom_level_habichuela)
            if data.get("slider", None) == 'main':
                zoom_level_main = 1 - (int(data.get('value', 0)) / 100)
                picam2.capture_metadata()
                new_size = [int(s * zoom_level_main) for s in size]
                offset = [(r - s) // 2 for r, s in zip(full_res, new_size)]
                picam2.set_controls({"ScalerCrop": offset + new_size})
                save_camera_settings('main', 0, zoom_level_main, servo0_angle, servo1_angle)
        if data.get("water_level", None) == 'water_level':
            water_level = water_monitor.read()
            water_out = water_level < 1000
            if water_out and not water_ran_out:
                water_ran_out = True
                water_refilled = False
                log_water_event('out')
                send_notification('Out of Water', "Out of water, please refill their bowl")
            if not water_out and not water_refilled:
                log_water_event('refill')
                send_notification('Water refilled', "They won't get dehydrated any more")
                water_refilled = True
                water_ran_out = False
            response = json.dumps({"water_level": water_level, 'water_out': water_out})
            await websocket.send(response)
        if data.get("food_level", None) == 'food_level':
            food_level = ultrasonic.get_distance()
            food_out = food_level < 20
            if food_out and not food_ran_out:
                food_ran_out = True
                food_refilled = False
                log_food_event('out')
                send_notification('Out of Food', "Shadow is out of food, he needs to eat")
            if not food_out and not food_refilled:
                food_refilled = True
                food_ran_out = False
                log_food_event('refill')
                send_notification('Food refilled', "Shadow had food to eat now")
            response = json.dumps({"food_level": food_level, 'food_out': food_out})
            await websocket.send(response)
        if data.get("loudness", None) == 'loudness':
            global rms
            global bark_detected
            response = json.dumps({'bark_detected': bark_detected, "rms": rms})
            await websocket.send(response)
        if data.get('state', None) == 'state':
            cam_settings = get_camera_settings()
            state = {}
            for cam_setting in cam_settings:
                if cam_setting[0] == 'main':
                    state['mainZoom'] = (1 - cam_setting[2]) * 100
                if cam_setting[0] == 'shadow':
                    state['shadowZoom'] = (cam_setting[2] - 1) * 100
                    state['shadowBrightness'] = cam_setting[1]
                if cam_setting[0] == 'habichuela':
                    state['habichuelaZoom'] = (cam_setting[2] - 1) * 100
                    state['habichuelaBrightness'] = cam_setting[1]
            response = json.dumps({'state': state})
            await websocket.send(response)
        if data.get('logs', None) == 'logs':
            response = json.dumps({'logs': get_logs()})
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
def run_bark_detector_thread():
    def run_loop():
        with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, device=DEVICE_INDEX, callback=audio_callback):
            while True:
                sd.sleep(500)
    thread = Thread(target=run_loop)
    thread.daemon = True
    thread.start()
    return thread

def play_music():
    """Plays the relaxing music in a separate thread."""
    Thread(target=playsound, args=(MUSIC_FILE,), daemon=True).start()
last_bark_time = None
music_playing = None
def audio_callback(indata, frames, time_, status):
    """Callback to process audio input."""
    global bark_detected
    global rms
    global last_bark_time
    global music_playing
    rms = float(max(np.sqrt(np.mean(indata[:, 0]**2)), np.sqrt(np.mean(indata[:, 1]**2))))
    if rms > LOUDNESS_THRESHOLD:
        if last_bark_time is None:
            last_bark_time = time.time()
        elif time.time() - last_bark_time >= 60:
            bark_detected = True
            last_bark_time = time.time()
            log_activity('bark detected', rms)
            send_notification('Shadow is barking', "Shadow is barking at something, go check it out")
            if music_playing == None or time.time() - music_playing >= 240:
                music_playing = time.time()
                play_music()    
    else:
        bark_detected = False

def run_scheduler_thread():
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)
    thread = Thread(target=run_scheduler)
    thread.daemon = True
    thread.start()
    return thread
def walk_notification():
    send_notification('Time to take shadow for a walk', "Shadow needs to go for a walk")
def shadow_grooming_notification():
    send_notification('Its time to give Shadow a haircut', "Give shadow a haircut so his hair doesnt get matter")
def habichuela_grooming_notification():
    send_notification('Its time to give brush Habichuela', "Brush Habichuela so that you dont have to deal with so much fur")

schedule.every().day.at("18:00").do(walk_notification)
schedule.every().day.at("21:00").do(habichuela_grooming_notification)
schedule.every(30).days.at("20:00").do(shadow_grooming_notification)

create_tables()
cam_settings = get_camera_settings()
for cam_setting in cam_settings:
    if cam_setting[0] == 'main':
        zoom_level_main = cam_setting[2]
        servo0_angle = cam_setting[3]
        servo1_angle = cam_setting[4]
        servo0.set_angle(servo0_angle + servo0_angle_offset)
        servo1.set_angle(servo1_angle + servo1_angle_offset)
    if cam_setting[0] == 'shadow':
        zoom_level_shadow = cam_setting[2]
        shadow_brightness = int(cam_setting[1])
    if cam_setting[0] == 'habichuela':
        zoom_level_habichuela = cam_setting[2]
        habichuela_brightness = int(cam_setting[1])
    
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 960)})
config["transform"] = libcamera.Transform(vflip=1)
picam2.configure(config)
output = StreamingOutput()

picam2.start_recording(JpegEncoder(), FileOutput(output))
size = picam2.capture_metadata()['ScalerCrop'][2:]
full_res = picam2.camera_properties['PixelArraySize']
new_size = [int(s * zoom_level_main) for s in size]
offset = [(r - s) // 2 for r, s in zip(full_res, new_size)]
picam2.set_controls({"ScalerCrop": offset + new_size})

picam2_shadow_monitor = Picamera2(1)
picam2_shadow_monitor.start()

picam2_habichuela_monitor = Picamera2(2)
picam2_habichuela_monitor.start()
try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    http_server_thread = Thread(target=server.serve_forever)
    http_server_thread.daemon = True
    http_server_thread.start()
    run_websocket_server_in_thread(start_websocket_server())
    run_bark_detector_thread()
    run_scheduler_thread()
    asyncio.run(start_websocket_server_poop_monitor())
finally:
    picam2.stop_recording()
