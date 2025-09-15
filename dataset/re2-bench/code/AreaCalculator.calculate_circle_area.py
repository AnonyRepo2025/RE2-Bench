
import math

class AreaCalculator():

    def __init__(self, radius):
        self.radius = radius

    def calculate_circle_area(self):
        return (math.pi * (self.radius ** 2))
