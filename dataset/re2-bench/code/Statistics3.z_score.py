
import math

class Statistics3():

    @staticmethod
    def z_score(data):
        mean = Statistics3.mean(data)
        std_deviation = Statistics3.standard_deviation(data)
        if ((std_deviation is None) or (std_deviation == 0)):
            return None
        return [((x - mean) / std_deviation) for x in data]
