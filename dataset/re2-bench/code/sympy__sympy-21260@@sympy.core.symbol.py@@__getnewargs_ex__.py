from sympy.core.assumptions import StdFactKB, _assume_defined
from sympy.core.compatibility import is_sequence, ordered
from .basic import Basic, Atom
from .sympify import sympify
from .singleton import S
from .expr import Expr, AtomicExpr
from .cache import cacheit
from .function import FunctionClass
from .kind import NumberKind, UndefinedKind
from sympy.core.logic import fuzzy_bool
from sympy.logic.boolalg import Boolean
from sympy.utilities.iterables import cartes, sift
from sympy.core.containers import Tuple
import string
import re as _re
import random
from sympy.core.function import AppliedUndef
from inspect import currentframe
from sympy.core.power import Pow
from sympy import im, re
import sage.all as sage
from sympy.utilities.misc import filldedent

_uniquely_named_symbol = uniquely_named_symbol
_range = _re.compile('([0-9]*:[0-9]+|[a-zA-Z]?:[a-zA-Z])')

class Dummy(Symbol):
    _count = 0
    _prng = random.Random()
    _base_dummy_index = _prng.randint(10**6, 9*10**6)
    __slots__ = ('dummy_index',)
    is_Dummy = True
    def __getnewargs_ex__(self):
        return ((self.name, self.dummy_index), self.assumptions0)