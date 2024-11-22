from gpiozero import DistanceSensor
from time import sleep
class Ultrasonic():
    def __init__(self):
        self.sensor = DistanceSensor(echo=6, trigger=5)

    def get_distance(self):
        distance = self.sensor.distance * 100  # Convert from meters to centimeters
        return round(distance, 2)
