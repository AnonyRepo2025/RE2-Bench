

class BinaryDataProcessor():

    def __init__(self, binary_string):
        self.binary_string = binary_string
        self.clean_non_binary_chars()

    def convert_to_utf8(self):
        byte_array = bytearray()
        for i in range(0, len(self.binary_string), 8):
            byte = self.binary_string[i:(i + 8)]
            decimal = int(byte, 2)
            byte_array.append(decimal)
        return byte_array.decode('utf-8')
