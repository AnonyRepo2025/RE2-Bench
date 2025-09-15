import copy
from collections.abc import Mapping



class OrderedSet:
    def __reversed__(self):
        return reversed(self.dict)