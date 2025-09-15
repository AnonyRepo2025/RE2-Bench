from __future__ import print_function, division
from collections import defaultdict, OrderedDict
from itertools import (
    combinations, combinations_with_replacement, permutations,
    product, product as cartes
)
import random
from operator import gt
from sympy.core import Basic
from sympy.core.compatibility import (
    as_int, default_sort_key, is_sequence, iterable, ordered
)
from sympy.utilities.enumerative import (
    multiset_partitions_taocp, list_visitor, MultisetPartitionTraverser)
from sympy.tensor.array import NDimArray
from sympy.printing import pprint
from sympy.core.compatibility import StringIO
import sys
from math import ceil, log
from sympy.core.containers import Dict
from sympy.sets.sets import Set
import itertools
from sympy import Symbol



def generate_derangements(perm):
    """
    Routine to generate unique derangements.

    TODO: This will be rewritten to use the
    ECO operator approach once the permutations
    branch is in master.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_derangements
    >>> list(generate_derangements([0, 1, 2]))
    [[1, 2, 0], [2, 0, 1]]
    >>> list(generate_derangements([0, 1, 2, 3]))
    [[1, 0, 3, 2], [1, 2, 3, 0], [1, 3, 0, 2], [2, 0, 3, 1], \
    [2, 3, 0, 1], [2, 3, 1, 0], [3, 0, 1, 2], [3, 2, 0, 1], \
    [3, 2, 1, 0]]
    >>> list(generate_derangements([0, 1, 1]))
    []

    See Also
    ========
    sympy.functions.combinatorial.factorials.subfactorial
    """
    for p in multiset_permutations(perm):
        if not any(i == j for i, j in zip(perm, p)):
            yield p
