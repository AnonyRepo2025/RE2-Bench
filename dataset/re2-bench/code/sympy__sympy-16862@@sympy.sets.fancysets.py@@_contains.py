from __future__ import print_function, division
from sympy.core.basic import Basic
from sympy.core.compatibility import as_int, with_metaclass, range, PY3
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols
from sympy.core.sympify import _sympify, sympify, converter
from sympy.logic.boolalg import And
from sympy.sets.sets import Set, Interval, Union, FiniteSet
from sympy.utilities.misc import filldedent
from sympy.functions.elementary.trigonometric import _pi_coeff as coeff
from sympy.matrices import Matrix
from sympy.solvers.solveset import solveset, linsolve
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import is_sequence, iterable, cartes
from sympy.sets.setexpr import SetExpr
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.integers import ceiling
from sympy import sin, cos
from sympy.functions import arg, Abs
from sympy.core.containers import Tuple
from sympy.sets import ProductSet



class ImageSet(Set):
    lamda = property(lambda self: self.args[0])
    base_set = property(lambda self: self.args[1])
    def _contains(self, other):
        from sympy.matrices import Matrix
        from sympy.solvers.solveset import solveset, linsolve
        from sympy.solvers.solvers import solve
        from sympy.utilities.iterables import is_sequence, iterable, cartes
        L = self.lamda
        if is_sequence(other):
            if not is_sequence(L.expr):
                return S.false
            if len(L.expr) != len(other):
                raise ValueError(filldedent('''
    Dimensions of other and output of Lambda are different.'''))
        elif iterable(other):
                raise ValueError(filldedent('''
    `other` should be an ordered object like a Tuple.'''))

        solns = None
        if self._is_multivariate():
            if not is_sequence(L.expr):
                # exprs -> (numer, denom) and check again
                # XXX this is a bad idea -- make the user
                # remap self to desired form
                return other.as_numer_denom() in self.func(
                    Lambda(L.variables, L.expr.as_numer_denom()), self.base_set)
            eqs = [expr - val for val, expr in zip(other, L.expr)]
            variables = L.variables
            free = set(variables)
            if all(i.is_number for i in list(Matrix(eqs).jacobian(variables))):
                solns = list(linsolve([e - val for e, val in
                zip(L.expr, other)], variables))
            else:
                try:
                    syms = [e.free_symbols & free for e in eqs]
                    solns = {}
                    for i, (e, s, v) in enumerate(zip(eqs, syms, other)):
                        if not s:
                            if e != v:
                                return S.false
                            solns[vars[i]] = [v]
                            continue
                        elif len(s) == 1:
                            sy = s.pop()
                            sol = solveset(e, sy)
                            if sol is S.EmptySet:
                                return S.false
                            elif isinstance(sol, FiniteSet):
                                solns[sy] = list(sol)
                            else:
                                raise NotImplementedError
                        else:
                            # if there is more than 1 symbol from
                            # variables in expr than this is a
                            # coupled system
                            raise NotImplementedError
                    solns = cartes(*[solns[s] for s in variables])
                except NotImplementedError:
                    solns = solve([e - val for e, val in
                        zip(L.expr, other)], variables, set=True)
                    if solns:
                        _v, solns = solns
                        # watch for infinite solutions like solving
                        # for x, y and getting (x, 0), (0, y), (0, 0)
                        solns = [i for i in solns if not any(
                            s in i for s in variables)]
        else:
            x = L.variables[0]
            if isinstance(L.expr, Expr):
                # scalar -> scalar mapping
                solnsSet = solveset(L.expr - other, x)
                if solnsSet.is_FiniteSet:
                    solns = list(solnsSet)
                else:
                    msgset = solnsSet
            else:
                # scalar -> vector
                for e, o in zip(L.expr, other):
                    solns = solveset(e - o, x)
                    if solns is S.EmptySet:
                        return S.false
                    for soln in solns:
                        try:
                            if soln in self.base_set:
                                break  # check next pair
                        except TypeError:
                            if self.base_set.contains(soln.evalf()):
                                break
                    else:
                        return S.false  # never broke so there was no True
                return S.true

        if solns is None:
            raise NotImplementedError(filldedent('''
            Determining whether %s contains %s has not
            been implemented.''' % (msgset, other)))
        for soln in solns:
            try:
                if soln in self.base_set:
                    return S.true
            except TypeError:
                return self.base_set.contains(soln.evalf())
        return S.false