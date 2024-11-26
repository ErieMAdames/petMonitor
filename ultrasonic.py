from gpiozero import DistanceSensor
from time import sleep
class Ultrasonic():
    def __init__(self):
        self.sensor = DistanceSensor(echo=6, trigger=5)

    def get_distance(self):
        distance = self.sensor.distance * 100  # Convert from meters to centimeters
        return round(distance, 2)


# Set up the ultrasonic sensor with trigger and echo pins


try:
    u = Ultrasonic()
    while True:
        distance = u.get_distance()
        print(f"Distance: {distance} cm")
        sleep(1)  # Wait for 1 second before the next measurement

except KeyboardInterrupt:
    print("Measurement stopped by user")
