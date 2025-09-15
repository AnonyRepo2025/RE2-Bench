def iterable(i, exclude=(string_types, dict, NotIterable)):
    if hasattr(i, '_iterable'):
        return i._iterable
    try:
        iter(i)
    except TypeError:
        return False
    if exclude:
        return not isinstance(i, exclude)
    return True

def as_content_primitive(self, radical=False, clear=True):
    coef = S.One
    args = []
    for i, a in enumerate(self.args):
        c, p = a.as_content_primitive(radical=radical, clear=clear)
        coef *= c
        if p is not S.One:
            args.append(p)
    return (coef, self.func(*args))

def args(self):
    return self._args

def as_content_primitive(self, radical=False, clear=True):
    if self:
        if self.is_positive:
            return (self, S.One)
        return (-self, S.NegativeOne)
    return (S.One, self)

def getit(self):
    try:
        return self._assumptions[fact]
    except KeyError:
        if self._assumptions is self.default_assumptions:
            self._assumptions = self.default_assumptions.copy()
        return _ask(fact, self)

def _ask(fact, obj):
    assumptions = obj._assumptions
    handler_map = obj._prop_handler
    assumptions._tell(fact, None)
    try:
        evaluate = handler_map[fact]
    except KeyError:
        pass
    else:
        a = evaluate(obj)
        if a is not None:
            assumptions.deduce_all_facts(((fact, a),))
            return a
    prereq = list(_assume_rules.prereq[fact])
    shuffle(prereq)
    for pk in prereq:
        if pk in assumptions:
            continue
        if pk in handler_map:
            _ask(pk, obj)
            ret_val = assumptions.get(fact)
            if ret_val is not None:
                return ret_val
    return None

def _tell(self, k, v):
    if k in self and self[k] is not None:
        if self[k] == v:
            return False
        else:
            raise InconsistentAssumptions(self, k, v)
    else:
        self[k] = v
        return True

def _eval_is_positive(self):
    return self.p > 0

def deduce_all_facts(self, facts):
    full_implications = self.rules.full_implications
    beta_triggers = self.rules.beta_triggers
    beta_rules = self.rules.beta_rules
    if isinstance(facts, dict):
        facts = facts.items()
    while facts:
        beta_maytrigger = set()
        for k, v in facts:
            if not self._tell(k, v) or v is None:
                continue
            for key, value in full_implications[k, v]:
                self._tell(key, value)
            beta_maytrigger.update(beta_triggers[k, v])
        facts = []
        for bidx in beta_maytrigger:
            bcond, bimpl = beta_rules[bidx]
            if all((self.get(k) is v for k, v in bcond)):
                facts.append(bimpl)

def __mul__(self, other):
    if global_evaluate[0]:
        if isinstance(other, integer_types):
            return Integer(self.p * other)
        elif isinstance(other, Integer):
            return Integer(self.p * other.p)
        elif isinstance(other, Rational):
            return Rational(self.p * other.p, other.q, igcd(self.p, other.q))
        return Rational.__mul__(self, other)
    return Rational.__mul__(self, other)

def __new__(cls, i):
    if isinstance(i, string_types):
        i = i.replace(' ', '')
    try:
        ival = int(i)
    except TypeError:
        raise TypeError('Argument of Integer should be of numeric type, got %s.' % i)
    if ival == 1:
        return S.One
    if ival == -1:
        return S.NegativeOne
    if ival == 0:
        return S.Zero
    obj = Expr.__new__(cls)
    obj.p = ival
    return obj

def __new__(cls, *args):
    obj = object.__new__(cls)
    obj._assumptions = cls.default_assumptions
    obj._mhash = None
    obj._args = args
    return obj

def as_content_primitive(self, radical=False, clear=True):
    return (S.One, self)

def func(self):
    return self.__class__

def __new__(cls, *args, **options):
    from sympy import Order
    args = list(map(_sympify, args))
    args = [a for a in args if a is not cls.identity]
    evaluate = options.get('evaluate')
    if evaluate is None:
        evaluate = global_evaluate[0]
    if not evaluate:
        obj = cls._from_args(args)
        obj = cls._exec_constructor_postprocessors(obj)
        return obj
    if len(args) == 0:
        return cls.identity
    if len(args) == 1:
        return args[0]
    c_part, nc_part, order_symbols = cls.flatten(args)
    is_commutative = not nc_part
    obj = cls._from_args(c_part + nc_part, is_commutative)
    obj = cls._exec_constructor_postprocessors(obj)
    if order_symbols is not None:
        return Order(obj, *order_symbols)
    return obj



from __future__ import print_function, division
from sympy.core.add import Add
from sympy.core.compatibility import iterable, is_sequence, SYMPY_INTS, range
from sympy.core.mul import Mul, _keep_coeff
from sympy.core.power import Pow
from sympy.core.basic import Basic, preorder_traversal
from sympy.core.expr import Expr
from sympy.core.sympify import sympify
from sympy.core.numbers import Rational, Integer, Number, I
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.coreerrors import NonCommutativeExpression
from sympy.core.containers import Tuple, Dict
from sympy.utilities import default_sort_key
from sympy.utilities.iterables import common_prefix, common_suffix, variations, ordered
from collections import defaultdict
from sympy.simplify.simplify import powsimp
from sympy.polys import gcd, factor
from sympy.concrete.summations import Sum
from sympy.integrals.integrals import Integral
from sympy import Dummy
from sympy.polys.polytools import real_roots
from sympy.polys.polyroots import roots
from sympy.polys.polyerrors import PolynomialError
_eps = Dummy(positive=True)

def factor_terms(expr, radical=False, clear=False, fraction=False, sign=True):

    def do(expr):
        from sympy.concrete.summations import Sum
        from sympy.integrals.integrals import Integral
        is_iterable = iterable(expr)
        if not isinstance(expr, Basic) or expr.is_Atom:
            if is_iterable:
                return type(expr)([do(i) for i in expr])
            return expr
        if expr.is_Pow or expr.is_Function or is_iterable or (not hasattr(expr, 'args_cnc')):
            args = expr.args
            newargs = tuple([do(i) for i in args])
            if newargs == args:
                return expr
            return expr.func(*newargs)
        if isinstance(expr, (Sum, Integral)):
            return _factor_sum_int(expr, radical=radical, clear=clear, fraction=fraction, sign=sign)
        cont, p = expr.as_content_primitive(radical=radical, clear=clear)
        if p.is_Add:
            list_args = [do(a) for a in Add.make_args(p)]
            if all((a.as_coeff_Mul()[0].extract_multiplicatively(-1) is not None for a in list_args)):
                cont = -cont
                list_args = [-a for a in list_args]
            special = {}
            for i, a in enumerate(list_args):
                b, e = a.as_base_exp()
                if e.is_Mul and e != Mul(*e.args):
                    list_args[i] = Dummy()
                    special[list_args[i]] = a
            p = Add._from_args(list_args)
            p = gcd_terms(p, isprimitive=True, clear=clear, fraction=fraction).xreplace(special)
        elif p.args:
            p = p.func(*[do(a) for a in p.args])
        rv = _keep_coeff(cont, p, clear=clear, sign=sign)
        return rv
    expr = sympify(expr)
    return do(expr)