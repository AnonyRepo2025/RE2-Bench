from sympy.core import Basic, Dict, sympify
from sympy.core.compatibility import as_int, default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.numbers import bell
from sympy.matrices import zeros
from sympy.sets.sets import FiniteSet, Union
from sympy.utilities.iterables import flatten, group
from collections import defaultdict
from sympy.testing.randtest import _randint



class Partition(FiniteSet):
    _rank = None
    _partition = None
    def __new__(cls, *partition):
        """
        Generates a new partition object.

        This method also verifies if the arguments passed are
        valid and raises a ValueError if they are not.

        Examples
        ========

        Creating Partition from Python lists:

        >>> from sympy.combinatorics.partitions import Partition
        >>> a = Partition([1, 2], [3])
        >>> a
        Partition({3}, {1, 2})
        >>> a.partition
        [[1, 2], [3]]
        >>> len(a)
        2
        >>> a.members
        (1, 2, 3)

        Creating Partition from Python sets:

        >>> Partition({1, 2, 3}, {4, 5})
        Partition({4, 5}, {1, 2, 3})

        Creating Partition from SymPy finite sets:

        >>> from sympy.sets.sets import FiniteSet
        >>> a = FiniteSet(1, 2, 3)
        >>> b = FiniteSet(4, 5)
        >>> Partition(a, b)
        Partition({4, 5}, {1, 2, 3})
        """
        args = []
        dups = False
        for arg in partition:
            if isinstance(arg, list):
                as_set = set(arg)
                if len(as_set) < len(arg):
                    dups = True
                    break  # error below
                arg = as_set
            args.append(_sympify(arg))

        if not all(isinstance(part, FiniteSet) for part in args):
            raise ValueError(
                "Each argument to Partition should be " \
                "a list, set, or a FiniteSet")

        # sort so we have a canonical reference for RGS
        U = Union(*args)
        if dups or len(U) < sum(len(arg) for arg in args):
            raise ValueError("Partition contained duplicate elements.")

        obj = FiniteSet.__new__(cls, *args)
        obj.members = tuple(U)
        obj.size = len(U)
        return obj