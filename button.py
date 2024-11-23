from gpiozero import Button
from motor import Motor
import time

button = Button(17)
counter = 0
last_pressed_time = 0
motor = Motor()

def on_button_press():
    global counter, last_pressed_time
    current_time = time.time()
    if current_time - last_pressed_time >= 2:
        print("Button pressed " + str(counter) + ' times')
        # motor.set_power(35)
        # time.sleep(5)
        motor.set_power(0)
        counter += 1
        last_pressed_time = current_time

# Attach the function to the button press event
button.when_pressed = on_button_press

try:
    while True:
        time.sleep(0.1)  # Keep the program running to listen for button presses

except KeyboardInterrupt:
    print("Program stopped by user")
