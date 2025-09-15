

class BoyerMooreSearch():

    def __init__(self, text, pattern):
        (self.text, self.pattern) = (text, pattern)
        (self.textLen, self.patLen) = (len(text), len(pattern))

    def mismatch_in_text(self, currentPos):
        for i in range((self.patLen - 1), (- 1), (- 1)):
            if (self.pattern[i] != self.text[(currentPos + i)]):
                return (currentPos + i)
        return (- 1)
