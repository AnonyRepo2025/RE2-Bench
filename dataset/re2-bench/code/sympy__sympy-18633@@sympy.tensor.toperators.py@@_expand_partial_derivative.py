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
    def _expand_partial_derivative(self):
        args, indices, free, dum = self._contract_indices_for_derivative(self.expr, self.variables)

        obj = self.func(*args)
        obj._indices = indices
        obj._free = free
        obj._dum = dum

        result = obj

        if not args[0].free_symbols:
            return S.Zero
        elif isinstance(obj.expr, TensAdd):
            # take care of sums of multi PDs
            result = obj.expr.func(*[
                    self.func(a, *obj.variables)._expand_partial_derivative()
                    for a in result.expr.args])
        elif isinstance(obj.expr, TensMul):
            # take care of products of multi PDs
            if len(obj.variables) == 1:
                # derivative with respect to single variable
                terms = []
                mulargs = list(obj.expr.args)
                for ind in range(len(mulargs)):
                    if not isinstance(sympify(mulargs[ind]), Number):
                        # a number coefficient is not considered for
                        # expansion of PartialDerivative
                        d = self.func(mulargs[ind], *obj.variables)._expand_partial_derivative()
                        terms.append(TensMul(*(mulargs[:ind]
                                               + [d]
                                               + mulargs[(ind + 1):])))
                result = TensAdd.fromiter(terms)
            else:
                # derivative with respect to multiple variables
                # decompose:
                # partial(expr, (u, v))
                # = partial(partial(expr, u).doit(), v).doit()
                result = obj.expr  # init with expr
                for v in obj.variables:
                    result = self.func(result, v)._expand_partial_derivative()
                    # then throw PD on it

        return result
    @property
    def expr(self):
        return self.args[0]
    @property
    def variables(self):
        return self.args[1:]