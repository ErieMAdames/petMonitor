from servo import Servo
from pwm import PWM

s = Servo(PWM(3), 10)
while True:
    angle = int(input())
    s.set_angle(angle)