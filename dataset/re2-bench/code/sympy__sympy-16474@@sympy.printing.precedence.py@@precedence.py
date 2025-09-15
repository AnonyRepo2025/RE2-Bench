from __future__ import print_function, division
from sympy.core.function import _coeff_isneg
from sympy import Integral, Sum, Product, Limit, Derivative, Transpose, Adjoint
from sympy.core.expr import UnevaluatedExpr
from sympy.tensor.functions import TensorProduct

PRECEDENCE = {
    "Lambda": 1,
    "Xor": 10,
    "Or": 20,
    "And": 30,
    "Relational": 35,
    "Add": 40,
    "Mul": 50,
    "Pow": 60,
    "Func": 70,
    "Not": 100,
    "Atom": 1000,
    "BitwiseOr": 36,
    "BitwiseAnd": 38
}
PRECEDENCE_VALUES = {
    "Equivalent": PRECEDENCE["Xor"],
    "Xor": PRECEDENCE["Xor"],
    "Implies": PRECEDENCE["Xor"],
    "Or": PRECEDENCE["Or"],
    "And": PRECEDENCE["And"],
    "Add": PRECEDENCE["Add"],
    "Pow": PRECEDENCE["Pow"],
    "Relational": PRECEDENCE["Relational"],
    "Sub": PRECEDENCE["Add"],
    "Not": PRECEDENCE["Not"],
    "Function" : PRECEDENCE["Func"],
    "NegativeInfinity": PRECEDENCE["Add"],
    "MatAdd": PRECEDENCE["Add"],
    "MatPow": PRECEDENCE["Pow"],
    "TensAdd": PRECEDENCE["Add"],
    # As soon as `TensMul` is a subclass of `Mul`, remove this:
    "TensMul": PRECEDENCE["Mul"],
    "HadamardProduct": PRECEDENCE["Mul"],
    "HadamardPower": PRECEDENCE["Pow"],
    "KroneckerProduct": PRECEDENCE["Mul"],
    "Equality": PRECEDENCE["Mul"],
    "Unequality": PRECEDENCE["Mul"],
}
PRECEDENCE_FUNCTIONS = {
    "Integer": precedence_Integer,
    "Mul": precedence_Mul,
    "Rational": precedence_Rational,
    "Float": precedence_Float,
    "PolyElement": precedence_PolyElement,
    "FracElement": precedence_FracElement,
    "UnevaluatedExpr": precedence_UnevaluatedExpr,
}

def precedence(item):
    """Returns the precedence of a given object.

    This is the precedence for StrPrinter.
    """
    if hasattr(item, "precedence"):
        return item.precedence
    try:
        mro = item.__class__.__mro__
    except AttributeError:
        return PRECEDENCE["Atom"]
    for i in mro:
        n = i.__name__
        if n in PRECEDENCE_FUNCTIONS:
            return PRECEDENCE_FUNCTIONS[n](item)
        elif n in PRECEDENCE_VALUES:
            return PRECEDENCE_VALUES[n]
    return PRECEDENCE["Atom"]
