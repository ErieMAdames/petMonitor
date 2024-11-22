from pwm import PWM
from pin import Pin
import time
class Motor():
    STEP = 10
    DELAY = 0.1
    def __init__(self, pwm_pin, dir_pin, is_reversed=False):
        self.pwm_pin = pwm_pin
        self.dir_pin = dir_pin
        self._is_reversed = is_reversed
        self._power = 0
        self._except_power = 0
    def set_power(self, power):
        if power >= 0:
            direction = 0
        elif power < 0:
            direction = 1
        power = abs(power)
        if power != 0:
            power = int(power /2 ) + 50
        power = power
        direction = direction if not self._is_reversed else not direction  
        self.dir_pin.value(direction)
        self.pwm_pin.pulse_width_percent(power)

m = Motor(PWM("P13"), Pin("D4"))
while True:
    time_ = int(input())
    power = int(input())
    m.set_power(power)
    time.sleep(time_)
    m.set_power(0)