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
    def __pow__(self, n):
        """
        Routine for finding powers of a permutation.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.interactive import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([2, 0, 3, 1])
        >>> p.order()
        4
        >>> p**4
        Permutation([0, 1, 2, 3])
        """
        if isinstance(n, Permutation):
            raise NotImplementedError(
                'p**p is not defined; do you mean p^p (conjugate)?')
        n = int(n)
        return self._af_new(_af_pow(self.array_form, n))