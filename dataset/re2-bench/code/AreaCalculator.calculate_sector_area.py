
import math

class AreaCalculator():

    def __init__(self, radius):
        self.radius = radius

    def calculate_sector_area(self, angle):
        return (((self.radius ** 2) * angle) / 2)
