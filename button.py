from gpiozero import Button
from time import sleep

# Set up the button with GPIO pin (e.g., GPIO 17)
button = Button(17)

# Define a function to be called when the button is pressed
def on_button_press():
    print("Button pressed!")

# Attach the function to the button press event
button.when_pressed = on_button_press

try:
    while True:
        sleep(1)  # Keep the program running to listen for button presses

except KeyboardInterrupt:
    print("Program stopped by user")
