from __future__ import print_function, division
import random
from collections import defaultdict
from sympy.core.basic import Atom
from sympy.core.compatibility import is_sequence, reduce, range, as_int
from sympy.core.sympify import _sympify
from sympy.logic.boolalg import as_Boolean
from sympy.matrices import zeros
from sympy.polys.polytools import lcm
from sympy.utilities.iterables import (flatten, has_variety, minlex,
    has_dups, runs)
from mpmath.libmp.libintmath import ifac
from sympy.printing.repr import srepr
from collections import deque

Perm = Permutation
_af_new = Perm._af_new

class Permutation(Atom):
    is_Permutation = True
    _array_form = None
    _cyclic_form = None
    _cycle_structure = None
    _size = None
    _rank = None
    print_cyclic = None
    def __repr__(self):
        from sympy.printing.repr import srepr
        return srepr(self)