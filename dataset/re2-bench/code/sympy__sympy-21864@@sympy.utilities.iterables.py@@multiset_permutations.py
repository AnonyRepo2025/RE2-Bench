from collections import defaultdict, OrderedDict
from itertools import (
    combinations, combinations_with_replacement, permutations,
    product, product as cartes
)
import random
from operator import gt
from sympy.core import Basic
from sympy.core.compatibility import (as_int, is_sequence, iterable, ordered)
from sympy.core.compatibility import default_sort_key  # noqa: F401
import sympy
from sympy.utilities.enumerative import (
    multiset_partitions_taocp, list_visitor, MultisetPartitionTraverser)
from sympy.tensor.array import NDimArray
from sympy.printing import pprint
from io import StringIO
import sys
from math import ceil, log
from sympy.core.containers import Dict
from sympy.sets.sets import Set
import itertools
from sympy import Symbol



def multiset_permutations(m, size=None, g=None):
    """
    Return the unique permutations of multiset ``m``.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_permutations
    >>> from sympy import factorial
    >>> [''.join(i) for i in multiset_permutations('aab')]
    ['aab', 'aba', 'baa']
    >>> factorial(len('banana'))
    720
    >>> len(list(multiset_permutations('banana')))
    60
    """
    if g is None:
        if type(m) is dict:
            g = [[k, m[k]] for k in ordered(m)]
        else:
            m = list(ordered(m))
            g = [list(i) for i in group(m, multiple=False)]
        del m
    do = [gi for gi in g if gi[1] > 0]
    SUM = sum([gi[1] for gi in do])
    if not do or size is not None and (size > SUM or size < 1):
        if not do and size is None or size == 0:
            yield []
        return
    elif size == 1:
        for k, v in do:
            yield [k]
    elif len(do) == 1:
        k, v = do[0]
        v = v if size is None else (size if size <= v else 0)
        yield [k for i in range(v)]
    elif all(v == 1 for k, v in do):
        for p in permutations([k for k, v in do], size):
            yield list(p)
    else:
        size = size if size is not None else SUM
        for i, (k, v) in enumerate(do):
            do[i][1] -= 1
            for j in multiset_permutations(None, size - 1, do):
                if j:
                    yield [k] + j
            do[i][1] += 1
