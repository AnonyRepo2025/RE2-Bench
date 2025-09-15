
from PIL import Image, ImageEnhance, ImageChops

class ImageProcessor():

    def __init__(self):
        self.image = None

    def resize_image(self, width, height):
        if self.image:
            self.image = self.image.resize((width, height))
