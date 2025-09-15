from __future__ import print_function, division
import functools
import itertools
from sympy.core.sympify import _sympify
from sympy import Basic, Tuple
from sympy.tensor.array.mutable_ndim_array import MutableNDimArray
from sympy.tensor.array.ndim_array import NDimArray, ImmutableNDimArray
from sympy.matrices import Matrix
from sympy.utilities.iterables import flatten
from sympy.utilities.iterables import flatten



class ImmutableDenseNDimArray(DenseNDimArray, ImmutableNDimArray):
    @classmethod
    def _new(cls, iterable, shape, **kwargs):
        from sympy.utilities.iterables import flatten

        shape, flat_list = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        shape = Tuple(*map(_sympify, shape))
        flat_list = flatten(flat_list)
        flat_list = Tuple(*flat_list)
        self = Basic.__new__(cls, flat_list, shape, **kwargs)
        self._shape = shape
        self._array = list(flat_list)
        self._rank = len(shape)
        self._loop_size = functools.reduce(lambda x,y: x*y, shape, 1)
        return self