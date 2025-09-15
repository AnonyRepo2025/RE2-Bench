
import xml.etree.ElementTree as ET

class XMLProcessor():

    def __init__(self, file_name):
        self.file_name = file_name
        self.root = None

    def find_element(self, element_name):
        elements = self.root.findall(element_name)
        return elements
