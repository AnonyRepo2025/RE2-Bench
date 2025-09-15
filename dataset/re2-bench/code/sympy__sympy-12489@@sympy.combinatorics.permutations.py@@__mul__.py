from __future__ import print_function, division
import random
from collections import defaultdict
from sympy.core import Basic
from sympy.core.compatibility import is_sequence, reduce, range, as_int
from sympy.utilities.iterables import (flatten, has_variety, minlex,
    has_dups, runs)
from sympy.polys.polytools import lcm
from sympy.matrices import zeros
from mpmath.libmp.libintmath import ifac
from sympy.combinatorics.permutations import Permutation, Cycle
from collections import deque

Perm = Permutation
_af_new = Perm._af_new

class Permutation(Basic):
    is_Permutation = True
    _array_form = None
    _cyclic_form = None
    _cycle_structure = None
    _size = None
    _rank = None
    print_cyclic = True
    @property
    def array_form(self):
        """
        Return a copy of the attribute _array_form
        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> Permutation.print_cyclic = False
        >>> p = Permutation([[2, 0], [3, 1]])
        >>> p.array_form
        [2, 3, 0, 1]
        >>> Permutation([[2, 0, 3, 1]]).array_form
        [3, 2, 0, 1]
        >>> Permutation([2, 0, 3, 1]).array_form
        [2, 0, 3, 1]
        >>> Permutation([[1, 2], [4, 5]]).array_form
        [0, 2, 1, 3, 5, 4]
        """
        return self._array_form[:]
    def __mul__(self, other):
        """
        Return the product a*b as a Permutation; the ith value is b(a(i)).

        Examples
        ========

        >>> from sympy.combinatorics.permutations import _af_rmul, Permutation
        >>> Permutation.print_cyclic = False

        >>> a, b = [1, 0, 2], [0, 2, 1]
        >>> a = Permutation(a); b = Permutation(b)
        >>> list(a*b)
        [2, 0, 1]
        >>> [b(a(i)) for i in range(3)]
        [2, 0, 1]

        This handles operands in reverse order compared to _af_rmul and rmul:

        >>> al = list(a); bl = list(b)
        >>> _af_rmul(al, bl)
        [1, 2, 0]
        >>> [al[bl[i]] for i in range(3)]
        [1, 2, 0]

        It is acceptable for the arrays to have different lengths; the shorter
        one will be padded to match the longer one:

        >>> b*Permutation([1, 0])
        Permutation([1, 2, 0])
        >>> Permutation([1, 0])*b
        Permutation([2, 0, 1])

        It is also acceptable to allow coercion to handle conversion of a
        single list to the left of a Permutation:

        >>> [0, 1]*a # no change: 2-element identity
        Permutation([1, 0, 2])
        >>> [[0, 1]]*a # exchange first two elements
        Permutation([0, 1, 2])

        You cannot use more than 1 cycle notation in a product of cycles
        since coercion can only handle one argument to the left. To handle
        multiple cycles it is convenient to use Cycle instead of Permutation:

        >>> [[1, 2]]*[[2, 3]]*Permutation([]) # doctest: +SKIP
        >>> from sympy.combinatorics.permutations import Cycle
        >>> Cycle(1, 2)(2, 3)
        (1 3 2)

        """
        a = self.array_form
        # __rmul__ makes sure the other is a Permutation
        b = other.array_form
        if not b:
            perm = a
        else:
            b.extend(list(range(len(b), len(a))))
            perm = [b[i] for i in a] + b[len(a):]
        return self._af_new(perm)