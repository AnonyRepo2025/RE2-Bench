from sympy import Symbol, I, Mul, Pow, Add
from sympy.physics.quantum import TensorProduct

__all__ = ['evaluate_pauli_product']

class Pauli(Symbol):
    __slots__ = ("i", "label")
    def __getnewargs_ex__(self):
        return (self.i, self.label), {}