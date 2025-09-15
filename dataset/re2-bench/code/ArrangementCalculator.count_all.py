
import itertools

class ArrangementCalculator():

    def __init__(self, datas):
        self.datas = datas

    @staticmethod
    def count_all(n):
        total = 0
        for i in range(1, (n + 1)):
            total += ArrangementCalculator.count(n, i)
        return total
