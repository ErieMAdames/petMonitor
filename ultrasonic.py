# import time
# from pwm import PWM
# from pin import Pin

# class Ultrasonic():
#     def __init__(self, timeout=0.01):
#         self.timeout = timeout
#         self.trig = Pin("D8")
#         self.echo = Pin("D9")

#     def get_distance(self):
#         self.trig.low()
#         time.sleep(0.01)
#         self.trig.high()
#         time.sleep(0.000015)
#         self.trig.low()
#         pulse_end = 0
#         pulse_start = 0
#         timeout_start = time.time()
#         while self.echo.value()==0:
#             pulse_start = time.time()
#             if pulse_start - timeout_start > self.timeout:
#                 return -1
#         while self.echo.value()==1:
#             pulse_end = time.time()
#             if pulse_end - timeout_start > self.timeout:
#                 return -2
#         during = pulse_end - pulse_start
#         cm = round(during * 340 / 2 * 100, 2)
#         return cm
    

from gpiozero import DistanceSensor
from time import sleep

# Set up the ultrasonic sensor with trigger and echo pins
sensor = DistanceSensor(echo=6, trigger=5)

def get_distance():
    distance = sensor.distance * 100  # Convert from meters to centimeters
    return round(distance, 2)

try:
    while True:
        distance = get_distance()
        print(f"Distance: {distance} cm")
        sleep(1)  # Wait for 1 second before the next measurement

except KeyboardInterrupt:
    print("Measurement stopped by user")
