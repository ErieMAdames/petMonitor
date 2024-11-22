import pigpio
from time import sleep

# Initialize pigpio
pi = pigpio.pi()

# Define GPIO pin (e.g., GPIO 17)
BUTTON_PIN = 17

# Set the pin mode to input with pull-up resistor
pi.set_mode(BUTTON_PIN, pigpio.INPUT)
pi.set_pull_up_down(BUTTON_PIN, pigpio.PUD_UP)

# Variables to track button state
button_pressed = False
counter = 0

def on_button_press():
    global counter
    print("Button pressed " + str(counter) + " times")
    counter += 1

def on_button_release():
    print("Button released!")

# Callback for button state changes
def button_callback(gpio, level, tick):
    if level == 0:  # Button pressed
        on_button_press()
    elif level == 1:  # Button released
        on_button_release()

# Set up a callback on the BUTTON_PIN
pi.callback(BUTTON_PIN, pigpio.EITHER_EDGE, button_callback)

try:
    while True:
        sleep(0.1)  # Keep the program running to listen for button events

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    pi.stop()  # Clean up pigpio resources
