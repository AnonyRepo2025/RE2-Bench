from __future__ import print_function, division
from collections import defaultdict
import operator
import itertools
from sympy import Rational, prod, Integer
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
    bsgs_direct_product, canonicalize, riemann_bsgs
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.compatibility import string_types, reduce, range, SYMPY_INTS
from sympy.core.containers import Tuple, Dict
from sympy.core.decorators import deprecated
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.matrices import eye
from sympy.utilities.exceptions import SymPyDeprecationWarning
import warnings
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_new
from .array import NDimArray
from .array import tensorproduct, tensorcontraction, MutableDenseNDimArray
from .array import permutedims
from .array import tensorproduct, tensorcontraction
from .array import permutedims
from .array import MutableDenseNDimArray
from .array import MutableDenseNDimArray
from .array import tensorproduct, tensorcontraction
from .array import tensorproduct, tensorcontraction
from sympy import Matrix
from .array import tensorcontraction, tensorproduct, permutedims
from .array import Array
from sympy import Sum
from sympy.tensor.array import Array, permutedims
from .array import Array
from sympy import Indexed
from sympy import Sum
from .array import permutedims

_tensor_data_substitution_dict = _TensorDataLazyEvaluator()
TensorManager = _TensorManager()

class TensExpr(Expr):
    _op_priority = 12.0
    is_commutative = False
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    def replace_with_arrays(self, replacement_dict, indices=None):
        """
        Replace the tensorial expressions with arrays. The final array will
        correspond to the N-dimensional array with indices arranged according
        to ``indices``.

        Parameters
        ==========

        replacement_dict
            dictionary containing the replacement rules for tensors.
        indices
            the index order with respect to which the array is read. The
            original index order will be used if no value is passed.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices
        >>> from sympy.tensor.tensor import tensorhead
        >>> from sympy import symbols, diag

        >>> L = TensorIndexType("L")
        >>> i, j = tensor_indices("i j", L)
        >>> A = tensorhead("A", [L], [[1]])
        >>> A(i).replace_with_arrays({A(i): [1, 2]}, [i])
        [1, 2]

        Since 'indices' is optional, we can also call replace_with_arrays by
        this way if no specific index order is needed:

        >>> A(i).replace_with_arrays({A(i): [1, 2]})
        [1, 2]

        >>> expr = A(i)*A(j)
        >>> expr.replace_with_arrays({A(i): [1, 2]})
        [[1, 2], [2, 4]]

        For contractions, specify the metric of the ``TensorIndexType``, which
        in this case is ``L``, in its covariant form:

        >>> expr = A(i)*A(-i)
        >>> expr.replace_with_arrays({A(i): [1, 2], L: diag(1, -1)})
        -3

        Symmetrization of an array:

        >>> H = tensorhead("H", [L, L], [[1], [1]])
        >>> a, b, c, d = symbols("a b c d")
        >>> expr = H(i, j)/2 + H(j, i)/2
        >>> expr.replace_with_arrays({H(i, j): [[a, b], [c, d]]})
        [[a, b/2 + c/2], [b/2 + c/2, d]]

        Anti-symmetrization of an array:

        >>> expr = H(i, j)/2 - H(j, i)/2
        >>> repl = {H(i, j): [[a, b], [c, d]]}
        >>> expr.replace_with_arrays(repl)
        [[0, b/2 - c/2], [-b/2 + c/2, 0]]

        The same expression can be read as the transpose by inverting ``i`` and
        ``j``:

        >>> expr.replace_with_arrays(repl, [j, i])
        [[0, -b/2 + c/2], [b/2 - c/2, 0]]
        """
        from .array import Array

        indices = indices or []
        replacement_dict = {tensor: Array(array) for tensor, array in replacement_dict.items()}

        # Check dimensions of replaced arrays:
        for tensor, array in replacement_dict.items():
            if isinstance(tensor, TensorIndexType):
                expected_shape = [tensor.dim for i in range(2)]
            else:
                expected_shape = [index_type.dim for index_type in tensor.index_types]
            if len(expected_shape) != array.rank() or (not all([dim1 == dim2 if
                dim1 is not None else True for dim1, dim2 in zip(expected_shape,
                array.shape)])):
                raise ValueError("shapes for tensor %s expected to be %s, "\
                    "replacement array shape is %s" % (tensor, expected_shape,
                    array.shape))

        ret_indices, array = self._extract_data(replacement_dict)

        last_indices, array = self._match_indices_with_other_tensor(array, indices, ret_indices, replacement_dict)
        #permutation = self._get_indices_permutation(indices, ret_indices)
        #if not hasattr(array, "rank"):
            #return array
        #if array.rank() == 0:
            #array = array[()]
            #return array
        #array = permutedims(array, permutation)
        return array