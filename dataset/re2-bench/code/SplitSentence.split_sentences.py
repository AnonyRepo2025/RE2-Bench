
import re

class SplitSentence():

    def split_sentences(self, sentences_string):
        sentences = re.split('(?<!\\w\\.\\w.)(?<![A-Z][a-z]\\.)(?<=\\.|\\?)\\s', sentences_string)
        return sentences
