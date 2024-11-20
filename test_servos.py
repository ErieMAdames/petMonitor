
from servo import Servo
from pwm import PWM
import time

# Setup your servos and initial angles
servo0_angle_offset = 0
servo1_angle_offset = -8
servo0_angle = 0
servo1_angle = 0 
servo0 = Servo(PWM("P0"))
servo1 = Servo(PWM("P1"))

# servo0.set_angle(servo0_angle_offset)
# servo0.set_angle(servo0_angle_offset)

servo1.set_angle(0)
servo0.set_angle(0)
time.sleep(1)

servo1.set_angle(90)
servo0.set_angle(90)