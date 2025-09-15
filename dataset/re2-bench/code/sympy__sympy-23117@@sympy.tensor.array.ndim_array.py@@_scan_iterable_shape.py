from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.printing.defaults import Printable
import itertools
from collections.abc import Iterable
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.matrices.matrices import MatrixBase
from sympy.matrices.matrices import MatrixBase
from sympy.tensor.array import SparseNDimArray
from sympy.tensor.array.array_derivatives import ArrayDerivative
from sympy.tensor.array import SparseNDimArray
from sympy.tensor.array.arrayop import Flatten
from sympy.tensor.array.arrayop import Flatten
from sympy.tensor.array.arrayop import Flatten
from sympy.matrices.matrices import MatrixBase
from sympy.tensor.array import SparseNDimArray
from sympy.tensor.array.arrayop import Flatten
from sympy.matrices.matrices import MatrixBase
from sympy.tensor.array import SparseNDimArray
from sympy.tensor.array.arrayop import Flatten
from sympy.matrices.matrices import MatrixBase
from sympy.tensor.array import SparseNDimArray
from sympy.tensor.array.arrayop import Flatten
from sympy.tensor.array import SparseNDimArray
from sympy.tensor.array.arrayop import Flatten
from sympy.tensor.array import SparseNDimArray
from .arrayop import permutedims
from sympy.tensor.array.arrayop import Flatten
from sympy.tensor import Indexed



class NDimArray(Printable):
    _diff_wrt = True
    is_scalar = False
    @classmethod
    def _scan_iterable_shape(cls, iterable):
        def f(pointer):
            if not isinstance(pointer, Iterable):
                return [pointer], ()

            if len(pointer) == 0:
                return [], (0,)

            result = []
            elems, shapes = zip(*[f(i) for i in pointer])
            if len(set(shapes)) != 1:
                raise ValueError("could not determine shape unambiguously")
            for i in elems:
                result.extend(i)
            return result, (len(shapes),)+shapes[0]

        return f(iterable)