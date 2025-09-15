from sympy import Symbol, Number, sympify
from sympy import MutableDenseNDimArray, S
from sympy.tensor.tensor import (Tensor, TensExpr, TensAdd, TensMul,
                                 TensorIndex)
from .array import derive_by_array, tensorcontraction



class PartialDerivative(TensExpr):
    def __new__(cls, expr, *variables):

        # Flatten:
        if isinstance(expr, PartialDerivative):
            variables = expr.variables + variables
            expr = expr.expr

        args, indices, free, dum = cls._contract_indices_for_derivative(
            S(expr), variables)

        obj = TensExpr.__new__(cls, *args)

        obj._indices = indices
        obj._free = free
        obj._dum = dum
        return obj
    @classmethod
    def _contract_indices_for_derivative(cls, expr, variables):
        variables_opposite_valence = []

        for i in variables:
            if isinstance(i, Tensor):
                i_free_indices = i.get_free_indices()
                variables_opposite_valence.append(
                        i.xreplace({k: -k for k in i_free_indices}))
            elif isinstance(i, Symbol):
                variables_opposite_valence.append(i)

        args, indices, free, dum = TensMul._tensMul_contract_indices(
            [expr] + variables_opposite_valence, replace_indices=True)

        for i in range(1, len(args)):
            args_i = args[i]
            if isinstance(args_i, Tensor):
                i_indices = args[i].get_free_indices()
                args[i] = args[i].xreplace({k: -k for k in i_indices})

        return args, indices, free, dum
    @property
    def expr(self):
        return self.args[0]
    @property
    def variables(self):
        return self.args[1:]