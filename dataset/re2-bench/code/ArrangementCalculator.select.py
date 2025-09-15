
import itertools

class ArrangementCalculator():

    def __init__(self, datas):
        self.datas = datas

    def select(self, m=None):
        if (m is None):
            m = len(self.datas)
        result = []
        for permutation in itertools.permutations(self.datas, m):
            result.append(list(permutation))
        return result
