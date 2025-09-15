import copy
import itertools
import operator
import warnings
from functools import total_ordering, wraps
from django.utils.deprecation import RemovedInDjango50Warning

empty = object()

def new_method_proxy(func):
    def inner(self, *args):
        if self._wrapped is empty:
            self._setup()
        return func(self._wrapped, *args)

    inner._mask_wrapped = False
    return inner
