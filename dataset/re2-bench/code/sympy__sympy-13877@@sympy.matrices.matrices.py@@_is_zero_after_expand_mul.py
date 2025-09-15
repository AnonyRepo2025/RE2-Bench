from __future__ import print_function, division
import collections
from mpmath.libmp.libmpf import prec_to_dps
from sympy.assumptions.refine import refine
from sympy.core.add import Add
from sympy.core.basic import Basic, Atom
from sympy.core.expr import Expr
from sympy.core.function import expand_mul
from sympy.core.power import Pow
from sympy.core.symbol import (Symbol, Dummy, symbols,
    _uniquely_named_symbol)
from sympy.core.numbers import Integer, ilcm, Float
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt, Max, Min
from sympy.functions import Abs, exp, factorial
from sympy.polys import PurePoly, roots, cancel, gcd
from sympy.printing import sstr
from sympy.simplify import simplify as _simplify, signsimp, nsimplify
from sympy.core.compatibility import reduce, as_int, string_types
from sympy.utilities.iterables import flatten, numbered_symbols
from sympy.core.decorators import call_highest_priority
from sympy.core.compatibility import (is_sequence, default_sort_key, range,
    NotIterable)
from types import FunctionType
from .common import (a2idx, classof, MatrixError, ShapeError,
        NonSquareMatrixError, MatrixCommon)
from sympy.matrices import eye
from sympy import Derivative
from sympy.matrices import zeros
from .dense import matrix2numpy
from sympy.matrices import diag, MutableMatrix
from sympy import binomial
from sympy.matrices.sparse import SparseMatrix
from .dense import Matrix
from sympy.physics.matrices import mgamma
from .dense import Matrix
from sympy import LeviCivita
from sympy.matrices import zeros
from sympy.matrices import diag
from sympy.matrices import Matrix, zeros
from sympy.ntheory import totient
from .dense import Matrix
from sympy.matrices import SparseMatrix
from sympy.matrices import eye
from sympy.matrices import zeros
import numpy
from sympy.printing.str import StrPrinter
from sympy import gcd
from sympy import eye



def _is_zero_after_expand_mul(x):
    """Tests by expand_mul only, suitable for polynomials and rational
    functions."""
    return expand_mul(x) == 0
