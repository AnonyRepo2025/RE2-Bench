
import re

class RegexUtils():

    def split(self, pattern, text):
        return re.split(pattern, text)

    def generate_split_sentences_pattern(self):
        pattern = '[.!?][\\s]{1,2}(?=[A-Z])'
        return pattern

    def split_sentences(self, text):
        pattern = self.generate_split_sentences_pattern()
        return self.split(pattern, text)
