

class Manacher():

    def __init__(self, input_string) -> None:
        self.input_string = input_string

    def palindromic_length(self, center, diff, string):
        if (((center - diff) == (- 1)) or ((center + diff) == len(string)) or (string[(center - diff)] != string[(center + diff)])):
            return 0
        return (1 + self.palindromic_length(center, (diff + 1), string))
