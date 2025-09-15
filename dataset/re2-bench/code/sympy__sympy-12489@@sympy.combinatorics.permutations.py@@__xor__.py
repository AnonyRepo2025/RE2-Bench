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
    def size(self):
        """
        Returns the number of elements in the permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([[3, 2], [0, 1]]).size
        4

        See Also
        ========

        cardinality, length, order, rank
        """
        return self._size
    def __xor__(self, h):
        """Return the conjugate permutation ``~h*self*h` `.

        If ``a`` and ``b`` are conjugates, ``a = h*b*~h`` and
        ``b = ~h*a*h`` and both have the same cycle structure.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> Permutation.print_cyclic = True
        >>> p = Permutation(1, 2, 9)
        >>> q = Permutation(6, 9, 8)
        >>> p*q != q*p
        True

        Calculate and check properties of the conjugate:

        >>> c = p^q
        >>> c == ~q*p*q and p == q*c*~q
        True

        The expression q^p^r is equivalent to q^(p*r):

        >>> r = Permutation(9)(4, 6, 8)
        >>> q^p^r == q^(p*r)
        True

        If the term to the left of the conjugate operator, i, is an integer
        then this is interpreted as selecting the ith element from the
        permutation to the right:

        >>> all(i^p == p(i) for i in range(p.size))
        True

        Note that the * operator as higher precedence than the ^ operator:

        >>> q^r*p^r == q^(r*p)^r == Permutation(9)(1, 6, 4)
        True

        Notes
        =====

        In Python the precedence rule is p^q^r = (p^q)^r which differs
        in general from p^(q^r)

        >>> q^p^r
        (9)(1 4 8)
        >>> q^(p^r)
        (9)(1 8 6)

        For a given r and p, both of the following are conjugates of p:
        ~r*p*r and r*p*~r. But these are not necessarily the same:

        >>> ~r*p*r == r*p*~r
        True

        >>> p = Permutation(1, 2, 9)(5, 6)
        >>> ~r*p*r == r*p*~r
        False

        The conjugate ~r*p*r was chosen so that ``p^q^r`` would be equivalent
        to ``p^(q*r)`` rather than ``p^(r*q)``. To obtain r*p*~r, pass ~r to
        this method:

        >>> p^~r == r*p*~r
        True
        """

        if self.size != h.size:
            raise ValueError("The permutations must be of equal size.")
        a = [None]*self.size
        h = h._array_form
        p = self._array_form
        for i in range(self.size):
            a[h[i]] = h[p[i]]
        return self._af_new(a)