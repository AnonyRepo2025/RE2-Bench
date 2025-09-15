

class BinaryDataProcessor():

    def __init__(self, binary_string):
        self.binary_string = binary_string
        self.clean_non_binary_chars()

    def clean_non_binary_chars(self):
        self.binary_string = ''.join(filter((lambda x: (x in '01')), self.binary_string))
