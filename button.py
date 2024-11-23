from gpiozero import Button
import time

# Set up the button with GPIO pin (e.g., GPIO 17)
button = Button(17)

# Define a flag to track button state
counter = 0

# Define a variable to track the last execution time
last_pressed_time = 0

# Define a function to be called when the button is pressed
def on_button_press():
    global counter, last_pressed_time
    current_time = time.time()
    if current_time - last_pressed_time >= 2:  # Check if 2 seconds have passed
        print("Button pressed " + str(counter) + ' times')
        counter += 1
        last_pressed_time = current_time

# Attach the function to the button press event
button.when_pressed = on_button_press

try:
    while True:
        time.sleep(0.1)  # Keep the program running to listen for button presses

except KeyboardInterrupt:
    print("Program stopped by user")
