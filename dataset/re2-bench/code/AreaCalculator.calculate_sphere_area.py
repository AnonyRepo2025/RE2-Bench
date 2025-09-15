
import math

class AreaCalculator():

    def __init__(self, radius):
        self.radius = radius

    def calculate_sphere_area(self):
        return ((4 * math.pi) * (self.radius ** 2))
