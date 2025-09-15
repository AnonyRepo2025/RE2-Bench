from __future__ import print_function, division
from sympy.core import S, I, pi, oo, zoo, ilcm, Mod
from sympy.core.function import Function, Derivative, ArgumentIndexError
from sympy.core.compatibility import reduce
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Dummy
from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
        sinh, cosh, asinh, acosh, atanh, acoth, Abs)
from sympy.utilities.iterables import default_sort_key
from sympy import unpolarify
from sympy.series.limits import limit
from sympy import unpolarify
from sympy import gamma, hyperexpand
from sympy.functions import factorial, RisingFactorial, Piecewise
from sympy import Sum
from sympy.functions import factorial, RisingFactorial
from sympy import Order, Add
from sympy import And, Or, re, Ne, oo
from sympy.simplify.hyperexpand import hyperexpand
import sage.all as sage
from sympy import hyperexpand
from sympy.functions import exp_polar, ceiling
from sympy import Expr
import mpmath
from sympy import gamma
from sympy import unpolarify
from sympy import Piecewise



class hyper(TupleParametersBase):
    def _eval_nseries(self, x, n, logx):

        from sympy.functions import factorial, RisingFactorial
        from sympy import Order, Add

        arg = self.args[2]
        x0 = arg.limit(x, 0)
        ap = self.args[0]
        bq = self.args[1]

        if x0 != 0:
            return super(hyper, self)._eval_nseries(x, n, logx)

        terms = []

        for i in range(n):
            num = 1
            den = 1
            for a in ap:
                num *= RisingFactorial(a, i)

            for b in bq:
                den *= RisingFactorial(b, i)

            terms.append(((num/den) * (arg**i)) / factorial(i))

        return (Add(*terms) + Order(x**n,x))