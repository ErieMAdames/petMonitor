from picamera2 import Picamera2

cameras = Picamera2.global_camera_info()

for camera in cameras:
    print(camera)