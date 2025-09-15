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
    @classmethod
    def _af_new(cls, perm):
        """A method to produce a Permutation object from a list;
        the list is bound to the _array_form attribute, so it must
        not be modified; this method is meant for internal use only;
        the list ``a`` is supposed to be generated as a temporary value
        in a method, so p = Perm._af_new(a) is the only object
        to hold a reference to ``a``::

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Perm
        >>> from sympy.interactive import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> a = [2, 1, 3, 0]
        >>> p = Perm._af_new(a)
        >>> p
        Permutation([2, 1, 3, 0])

        """
        p = super(Permutation, cls).__new__(cls)
        p._array_form = perm
        p._size = len(perm)
        return p