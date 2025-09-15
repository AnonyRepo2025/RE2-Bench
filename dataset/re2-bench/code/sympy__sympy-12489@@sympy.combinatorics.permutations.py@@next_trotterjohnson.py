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
    def next_trotterjohnson(self):
        """
        Returns the next permutation in Trotter-Johnson order.
        If self is the last permutation it returns None.
        See [4] section 2.4. If it is desired to generate all such
        permutations, they can be generated in order more quickly
        with the ``generate_bell`` function.

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Permutation
        >>> Permutation.print_cyclic = False
        >>> p = Permutation([3, 0, 2, 1])
        >>> p.rank_trotterjohnson()
        4
        >>> p = p.next_trotterjohnson(); p
        Permutation([0, 3, 2, 1])
        >>> p.rank_trotterjohnson()
        5

        See Also
        ========

        rank_trotterjohnson, unrank_trotterjohnson, sympy.utilities.iterables.generate_bell
        """
        pi = self.array_form[:]
        n = len(pi)
        st = 0
        rho = pi[:]
        done = False
        m = n-1
        while m > 0 and not done:
            d = rho.index(m)
            for i in range(d, m):
                rho[i] = rho[i + 1]
            par = _af_parity(rho[:m])
            if par == 1:
                if d == m:
                    m -= 1
                else:
                    pi[st + d], pi[st + d + 1] = pi[st + d + 1], pi[st + d]
                    done = True
            else:
                if d == 0:
                    m -= 1
                    st += 1
                else:
                    pi[st + d], pi[st + d - 1] = pi[st + d - 1], pi[st + d]
                    done = True
        if m == 0:
            return None
        return self._af_new(pi)