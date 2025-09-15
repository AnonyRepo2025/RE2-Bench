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
    def __invert__(self):
        """
        Return the inverse of the permutation.

        A permutation multiplied by its inverse is the identity permutation.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.interactive import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([[2, 0], [3, 1]])
        >>> ~p
        Permutation([2, 3, 0, 1])
        >>> _ == p**-1
        True
        >>> p*~p == ~p*p == Permutation([0, 1, 2, 3])
        True
        """
        return self._af_new(_af_invert(self._array_form))