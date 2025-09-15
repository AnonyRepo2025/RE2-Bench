
import itertools

class ArrangementCalculator():

    def __init__(self, datas):
        self.datas = datas

    @staticmethod
    def count(n, m=None):
        if ((m is None) or (n == m)):
            return ArrangementCalculator.factorial(n)
        else:
            return (ArrangementCalculator.factorial(n) // ArrangementCalculator.factorial((n - m)))
