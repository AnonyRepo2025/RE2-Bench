

class AvgPartition():

    def __init__(self, lst, limit):
        self.lst = lst
        self.limit = limit

    def setNum(self):
        size = (len(self.lst) // self.limit)
        remainder = (len(self.lst) % self.limit)
        return (size, remainder)
