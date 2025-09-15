
from collections import Counter
import re

class NLPDataProcessor2():

    def process_data(self, string_list):
        words_list = []
        for string in string_list:
            processed_string = re.sub('[^a-zA-Z\\s]', '', string.lower())
            words = processed_string.split()
            words_list.append(words)
        return words_list
