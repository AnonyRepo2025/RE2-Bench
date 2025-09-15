import string
from itertools import cycle
import numpy as np
from astropy.utils.data_info import ParentDtypeInfo
from .table import Column, Table
import warnings
from astropy.io.votable.table import parse
from astropy.utils.data import get_pkg_data_filename



class ArrayWrapper:
    info = ArrayWrapperInfo()
    def __init__(self, data, copy=True):
        self.data = np.array(data, copy=copy)
        if 'info' in getattr(data, '__dict__', ()):
            self.info = data.info
    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            out = self.data[item]
        else:
            out = self.__class__(self.data[item], copy=False)
            if 'info' in self.__dict__:
                out.info = self.info
        return out
    @property
    def shape(self):
        return self.data.shape