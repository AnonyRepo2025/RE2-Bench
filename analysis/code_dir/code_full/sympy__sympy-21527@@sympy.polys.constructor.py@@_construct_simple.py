def getter(self):
    try:
        return self[cls.option]
    except KeyError:
        return cls.default()

def from_sympy(self, a):
    if a.is_Rational:
        return MPQ(a.p, a.q)
    elif a.is_Float:
        from sympy.polys.domains import RR
        return MPQ(*map(int, RR.to_rational(a)))
    else:
        raise CoercionFailed('expected `Rational` object, got %s' % a)

def __new__(cls, numerator, denominator=None):
    if denominator is not None:
        if isinstance(numerator, int) and isinstance(denominator, int):
            divisor = gcd(numerator, denominator)
            numerator //= divisor
            denominator //= divisor
            return cls._new_check(numerator, denominator)
    else:
        if isinstance(numerator, int):
            return cls._new(numerator, 1)
        elif isinstance(numerator, PythonMPQ):
            return cls._new(numerator.numerator, numerator.denominator)
        if isinstance(numerator, (Decimal, float, str)):
            numerator = Fraction(numerator)
        if isinstance(numerator, Fraction):
            return cls._new(numerator.numerator, numerator.denominator)
    raise TypeError('PythonMPQ() requires numeric or string argument')

def _new_check(cls, numerator, denominator):
    if not denominator:
        raise ZeroDivisionError(f'Zero divisor {numerator}/{denominator}')
    elif denominator < 0:
        numerator = -numerator
        denominator = -denominator
    return cls._new(numerator, denominator)

def _new(cls, numerator, denominator):
    obj = super().__new__(cls)
    obj.numerator = numerator
    obj.denominator = denominator
    return obj

def pure_complex(v, or_real=False):
    h, t = v.as_coeff_Add()
    if not t:
        if or_real:
            return (h, t)
        return
    c, i = t.as_coeff_Mul()
    if i is S.ImaginaryUnit:
        return (h, c)

def as_coeff_Add(self, rational=False):
    return (S.Zero, self)

def as_coeff_Mul(self, rational=False):
    coeff, args = (self.args[0], self.args[1:])
    if coeff.is_Number:
        if not rational or coeff.is_Rational:
            if len(args) == 1:
                return (coeff, args[0])
            else:
                return (coeff, self._new_rawargs(*args))
        elif coeff.is_extended_negative:
            return (S.NegativeOne, self._new_rawargs(*(-coeff,) + args))
    return (S.One, self)

def args(self):
    return self._args

def from_sympy(self, a):
    r, b = a.as_coeff_Add()
    x = self.dom.from_sympy(r)
    if not b:
        return self.new(x, 0)
    r, b = b.as_coeff_Mul()
    y = self.dom.from_sympy(r)
    if b is I:
        return self.new(x, y)
    else:
        raise CoercionFailed('{} is not Gaussian'.format(a))

def new(self, *args):
    return self.dtype(*args)

def __new__(cls, x, y=0):
    conv = cls.base.convert
    return cls.new(conv(x), conv(y))

def convert(self, element, base=None):
    if _not_a_coeff(element):
        raise CoercionFailed('%s is not in any domain' % element)
    if base is not None:
        return self.convert_from(element, base)
    if self.of_type(element):
        return element
    from sympy.polys.domains import ZZ, QQ, RealField, ComplexField
    if ZZ.of_type(element):
        return self.convert_from(element, ZZ)
    if isinstance(element, int):
        return self.convert_from(ZZ(element), ZZ)
    if HAS_GMPY:
        integers = ZZ
        if isinstance(element, integers.tp):
            return self.convert_from(element, integers)
        rationals = QQ
        if isinstance(element, rationals.tp):
            return self.convert_from(element, rationals)
    if isinstance(element, float):
        parent = RealField(tol=False)
        return self.convert_from(parent(element), parent)
    if isinstance(element, complex):
        parent = ComplexField(tol=False)
        return self.convert_from(parent(element), parent)
    if isinstance(element, DomainElement):
        return self.convert_from(element, element.parent())
    if self.is_Numerical and getattr(element, 'is_ground', False):
        return self.convert(element.LC())
    if isinstance(element, Basic):
        try:
            return self.from_sympy(element)
        except (TypeError, ValueError):
            pass
    elif not is_sequence(element):
        try:
            element = sympify(element, strict=True)
            if isinstance(element, Basic):
                return self.from_sympy(element)
        except (TypeError, ValueError):
            pass
    raise CoercionFailed("can't convert %s of type %s to %s" % (element, type(element), self))

def _not_a_coeff(expr):
    if type(expr) in illegal_types or expr in finf:
        return True
    if type(expr) is float and float(expr) != expr:
        return True
    return

def __eq__(self, other):
    if isinstance(other, PythonMPQ):
        return self.numerator == other.numerator and self.denominator == other.denominator
    elif isinstance(other, self._compatible_types):
        return self.__eq__(PythonMPQ(other))
    else:
        return NotImplemented



from sympy.core import sympify
from sympy.core.compatibility import ordered
from sympy.core.evalf import pure_complex
from sympy.polys.domains import ZZ, QQ, ZZ_I, QQ_I, EX
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.domains.realfield import RealField
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import parallel_dict_from_basic
from sympy.utilities import public
from sympy.polys.numberfields import primitive_element

def _construct_simple(coeffs, opt):
    rationals = floats = complexes = algebraics = False
    float_numbers = []
    if opt.extension is True:
        is_algebraic = lambda coeff: coeff.is_number and coeff.is_algebraic
    else:
        is_algebraic = lambda coeff: False
    for coeff in coeffs:
        if coeff.is_Rational:
            if not coeff.is_Integer:
                rationals = True
        elif coeff.is_Float:
            if algebraics:
                return False
            else:
                floats = True
                float_numbers.append(coeff)
        else:
            is_complex = pure_complex(coeff)
            if is_complex:
                complexes = True
                x, y = is_complex
                if x.is_Rational and y.is_Rational:
                    if not (x.is_Integer and y.is_Integer):
                        rationals = True
                    continue
                else:
                    floats = True
                    if x.is_Float:
                        float_numbers.append(x)
                    if y.is_Float:
                        float_numbers.append(y)
            elif is_algebraic(coeff):
                if floats:
                    return False
                algebraics = True
            else:
                return None
    max_prec = max((c._prec for c in float_numbers)) if float_numbers else 53
    if algebraics:
        domain, result = _construct_algebraic(coeffs, opt)
    else:
        if floats and complexes:
            domain = ComplexField(prec=max_prec)
        elif floats:
            domain = RealField(prec=max_prec)
        elif rationals or opt.field:
            domain = QQ_I if complexes else QQ
        else:
            domain = ZZ_I if complexes else ZZ
        result = [domain.from_sympy(coeff) for coeff in coeffs]
    return (domain, result)