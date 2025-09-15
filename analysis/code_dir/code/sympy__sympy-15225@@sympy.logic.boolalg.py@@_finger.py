from __future__ import print_function, division
from collections import defaultdict
from itertools import combinations, product
from sympy.core.basic import Basic, as_Basic
from sympy.core.cache import cacheit
from sympy.core.numbers import Number, oo
from sympy.core.operations import LatticeOp
from sympy.core.function import Application, Derivative
from sympy.core.compatibility import ordered, range, with_metaclass, as_int, reduce
from sympy.core.sympify import converter, _sympify, sympify
from sympy.core.singleton import Singleton, S
from sympy.utilities.misc import filldedent
from sympy.core.symbol import Symbol
from sympy.logic.inference import satisfiable
from sympy.core.relational import Relational
from sympy.calculus.util import periodicity
from sympy.core.relational import Relational
from sympy.core.relational import Eq, Ne
from sympy.utilities.misc import filldedent
from sympy.utilities.misc import filldedent
from sympy.core.relational import Relational, Eq, Ne
from sympy.core.relational import Eq, Relational
from sympy.functions.elementary.piecewise import Piecewise
from sympy.sets.sets import Intersection
from sympy.sets.sets import Union
from sympy import Equality, GreaterThan, LessThan, StrictGreaterThan, StrictLessThan, Unequality
from sympy.core.relational import Relational
from sympy.core.relational import Eq, Ne
from sympy.core.relational import Eq, Ne
from sympy.functions import Piecewise
from sympy.simplify.simplify import simplify
true = BooleanTrue()
false = BooleanFalse()
S.true = true
S.false = false
converter[bool] = lambda x: S.true if x else S.false

def _finger(eq):
    f = eq.free_symbols
    d = dict(list(zip(f, [[0] * 5 for fi in f])))
    for a in eq.args:
        if a.is_Symbol:
            d[a][0] += 1
        elif a.is_Not:
            d[a.args[0]][1] += 1
        else:
            o = len(a.args) + sum((isinstance(ai, Not) for ai in a.args))
            for ai in a.args:
                if ai.is_Symbol:
                    d[ai][2] += 1
                    d[ai][-1] += o
                elif ai.is_Not:
                    d[ai.args[0]][3] += 1
                else:
                    raise NotImplementedError('unexpected level of nesting')
    inv = defaultdict(list)
    for k, v in ordered(iter(d.items())):
        inv[tuple(v)].append(k)
    return inv