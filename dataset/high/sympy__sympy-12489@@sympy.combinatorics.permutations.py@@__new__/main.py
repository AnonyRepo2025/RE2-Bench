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
    def __new__(cls, *args, **kwargs):
        size = kwargs.pop('size', None)
        if size is not None:
            size = int(size)

        ok = True
        if not args:  # a
            return cls._af_new(list(range(size or 0)))
        elif len(args) > 1:  # c
            return cls._af_new(Cycle(*args).list(size))
        if len(args) == 1:
            a = args[0]
            if isinstance(a, cls):  # g
                if size is None or size == a.size:
                    return a
                return cls(a.array_form, size=size)
            if isinstance(a, Cycle):  # f
                return cls._af_new(a.list(size))
            if not is_sequence(a):  # b
                return cls._af_new(list(range(a + 1)))
            if has_variety(is_sequence(ai) for ai in a):
                ok = False
        else:
            ok = False
        if not ok:
            raise ValueError("Permutation argument must be a list of ints, "
                             "a list of lists, Permutation or Cycle.")

        args = list(args[0])

        is_cycle = args and is_sequence(args[0])
        if is_cycle:  # e
            args = [[int(i) for i in c] for c in args]
        else:  # d
            args = [int(i) for i in args]

        temp = flatten(args)
        if has_dups(temp) and not is_cycle:
            raise ValueError('there were repeated elements.')
        temp = set(temp)

        if not is_cycle and \
                any(i not in temp for i in range(len(temp))):
            raise ValueError("Integers 0 through %s must be present." %
                             max(temp))

        if is_cycle:
            # it's not necessarily canonical so we won't store
            # it -- use the array form instead
            c = Cycle()
            for ci in args:
                c = c(*ci)
            aform = c.list()
        else:
            aform = list(args)
        if size and size > len(aform):
            aform.extend(list(range(len(aform), size)))

        return cls._af_new(aform)
    @property
    def size(self):
        return self._size

class Cycle(dict):
    def __missing__(self, arg):
        """Enter arg into dictionary and return arg."""
        arg = as_int(arg)
        self[arg] = arg
        return arg

    def __iter__(self):
        for i in self.list():
            yield i

    def __call__(self, *other):
        rv = Cycle(*other)
        for k, v in zip(list(self.keys()), [rv[self[k]] for k in self.keys()]):
            rv[k] = v
        return rv
    def __init__(self, *args):
        if not args:
            return
        if len(args) == 1:
            if isinstance(args[0], Permutation):
                for c in args[0].cyclic_form:
                    self.update(self(*c))
                return
            elif isinstance(args[0], Cycle):
                for k, v in args[0].items():
                    self[k] = v
                return
        args = [as_int(a) for a in args]
        if any(i < 0 for i in args):
            raise ValueError('negative integers are not allowed in a cycle.')
        if has_dups(args):
            raise ValueError('All elements must be unique in a cycle.')
        for i in range(-len(args), 0):
            self[args[i]] = args[i + 1]
    def list(self, size=None):
        if not self and size is None:
            raise ValueError('must give size for empty Cycle')
        if size is not None:
            big = max([i for i in self.keys() if self[i] != i] + [0])
            size = max(size, big + 1)
        else:
            size = self.size
        return [self[i] for i in range(size)]
    def __missing__(self, arg):
        """Enter arg into dictionary and return arg."""
        arg = as_int(arg)
        self[arg] = arg
        return arg
    @property
    def size(self):
        if not self:
            return 0
        return max(self.keys()) + 1