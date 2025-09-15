
from PIL import Image, ImageEnhance, ImageChops

class ImageProcessor():

    def __init__(self):
        self.image = None

    def adjust_brightness(self, factor):
        if self.image:
            enhancer = ImageEnhance.Brightness(self.image)
            self.image = enhancer.enhance(factor)
