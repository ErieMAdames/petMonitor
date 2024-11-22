from gpiozero import Button
from time import sleep

# Set up the button with GPIO pin (e.g., GPIO 17)
button = Button(17, pull_up=False)

# Define a flag to track button state
button_pressed = False

# Define a function to be called when the button is pressed
def on_button_press():
    global button_pressed
    if not button_pressed:  # Only trigger if the button hasn't been pressed yet
        print("Button pressed!")
        button_pressed = True  # Set flag to prevent multiple presses

# Define a function to be called when the button is released
def on_button_release():
    global button_pressed
    print("Button released!")
    button_pressed = False  # Reset flag when button is released

# Attach the functions to the button press and release events
button.when_pressed = on_button_press
button.when_released = on_button_release

try:
    while True:
        sleep(0.1)  # Keep the program running to listen for button presses

except KeyboardInterrupt:
    print("Program stopped by user")
