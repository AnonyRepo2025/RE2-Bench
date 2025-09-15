
from PIL import Image, ImageEnhance, ImageChops

class ImageProcessor():

    def __init__(self):
        self.image = None

    def rotate_image(self, degrees):
        if self.image:
            self.image = self.image.rotate(degrees)
