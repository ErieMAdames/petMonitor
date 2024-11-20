import socket
import json
from servo import Servo
from pwm import PWM
import os

# Setup your servos and initial angles
servo0_angle_offset = 0
servo1_angle_offset = -8
servo0_angle = 0
servo1_angle = 0 
servo0 = Servo(PWM("P0"))
servo1 = Servo(PWM("P1"))

servo0.set_angle(servo0_angle_offset)
servo0.set_angle(servo0_angle_offset)\

# Set ulimit for file descriptors
os.system('ulimit -n 4096')

# # Start RTSP stream using ffmpeg
# ffmpeg_command = [
#     'ffmpeg', '-f', 'v4l2', '-i', '/dev/video0', '-preset', 'ultrafast',
#     '-f', 'rtsp', 'rtsp://192.168.86.46:8554/stream'
# ]
# ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

HOST = "192.168.86.32" # IP address of your Raspberry PI
PORT = 65432
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    try:
        while 1:
            client, clientInfo = s.accept()
            data = client.recv(1024)
            if data != b"":
                print(data)
                if (data == b"up"):
                    servo1_angle = max(servo1_angle - 2, -90)
                    servo1.set_angle(servo1_angle + servo1_angle_offset)
                elif (data == b"down"):
                    servo1_angle = min(servo1_angle + 2 , 90)
                    servo1.set_angle(servo1_angle + servo1_angle_offset)
                elif (data == b"left"):
                    servo0_angle = max(servo0_angle - 2, -90)
                    servo0.set_angle(servo0_angle + servo0_angle_offset)
                elif (data == b"right"):
                    servo0_angle = min(servo0_angle + 2, 90)
                    servo0.set_angle(servo0_angle + servo0_angle_offset)
                client.sendall(str.encode(json.dumps({
                    'data': data.decode("utf-8"),
                    'servo0_angle': servo0_angle,
                    'servo1_angle': servo1_angle
                }))) # Echo back to client
    except (KeyboardInterrupt, Exception) as e:
        print(e)
        print("Closing socket and terminating stream")
        # ffmpeg_process.terminate()  # Stop ffmpeg when done
        # ffmpeg_process.wait()  # Wait for ffmpeg to fully close
        client.close()
        s.close()
socket.socket.close()