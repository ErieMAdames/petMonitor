from gpiozero import Button
from signal import pause

# Define the GPIO pin for the button (e.g., GPIO 17)
BUTTON_PIN = 17

# Initialize the button
button = Button(BUTTON_PIN, pull_up=True)

# Variables to track button presses
counter = 0

# Define button press actions
def on_button_press():
    global counter
    print(f"Button pressed {counter} times")
    counter += 1

def on_button_release():
    print("Button released!")

# Link the button actions
button.when_pressed = on_button_press
button.when_released = on_button_release

# Keep the program running
try:
    pause()
except KeyboardInterrupt:
    print("Program stopped by user")
