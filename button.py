from gpiozero import Button
from time import sleep

# Set up the button with GPIO pin (e.g., GPIO 17)
button = Button(17, bounce_time=0.2)

# Define a flag to track button state
button_pressed = False
counter = 0
# Define a function to be called when the button is pressed
def on_button_press():
    global counter
    print("Button pressed " + str(counter) + ' times')
    counter += 1
# Define a function to be called when the button is released
def on_button_release():
    print("Button released!")

# Attach the functions to the button press and release events
button.when_pressed = on_button_press
button.when_released = on_button_release

try:
    while True:
        sleep(0.1)  # Keep the program running to listen for button presses

except KeyboardInterrupt:
    print("Program stopped by user")
