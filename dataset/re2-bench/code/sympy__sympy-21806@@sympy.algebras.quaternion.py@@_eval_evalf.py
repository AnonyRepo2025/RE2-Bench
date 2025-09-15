from sympy import S, Rational
from sympy import re, im, conjugate, sign
from sympy import sqrt, sin, cos, acos, exp, ln
from sympy import trigsimp
from sympy import integrate
from sympy import Matrix
from sympy import sympify
from sympy.core.evalf import prec_to_dps
from sympy.core.expr import Expr



class Quaternion(Expr):
    _op_priority = 11.0
    is_commutative = False
    def _eval_evalf(self, prec):
        """Returns the floating point approximations (decimal numbers) of the quaternion.

        Returns
        =======

        Quaternion
            Floating point approximations of quaternion(self)

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import sqrt
        >>> q = Quaternion(1/sqrt(1), 1/sqrt(2), 1/sqrt(3), 1/sqrt(4))
        >>> q.evalf()
        1.00000000000000
        + 0.707106781186547*i
        + 0.577350269189626*j
        + 0.500000000000000*k

        """

        return Quaternion(*[arg.evalf(n=prec_to_dps(prec)) for arg in self.args])