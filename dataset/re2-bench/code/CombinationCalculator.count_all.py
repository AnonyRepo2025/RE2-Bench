
import math
from typing import List

class CombinationCalculator():

    def __init__(self, datas: List[str]):
        self.datas = datas

    @staticmethod
    def count_all(n: int) -> int:
        if ((n < 0) or (n > 63)):
            return False
        return (((1 << n) - 1) if (n != 63) else float('inf'))
