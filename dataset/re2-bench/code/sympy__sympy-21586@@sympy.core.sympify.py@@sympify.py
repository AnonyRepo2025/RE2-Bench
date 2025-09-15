import typing
from inspect import getmro
from .compatibility import iterable
from .parameters import global_parameters
from .basic import Basic
from typing import Any, Callable, Dict, Type
import numpy as np
from sympy.parsing.sympy_parser import (parse_expr, TokenError,
                                            standard_transformations)
from sympy.parsing.sympy_parser import convert_xor as t_convert_xor
from sympy.parsing.sympy_parser import rationalize as t_rationalize
import string
from random import choice
from sympy.core.symbol import Symbol
import numpy as np
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.numbers import Float
import numpy as np
from ..tensor.array import Array

converter = {}  # type: Dict[Type[Any], Callable[[Any], Basic]]

def sympify(a, locals=None, convert_xor=True, strict=False, rational=False,
        evaluate=None):
    """
    Converts an arbitrary expression to a type that can be used inside SymPy.

    Explanation
    ===========

    It will convert Python ints into instances of sympy.Integer,
    floats into instances of sympy.Float, etc. It is also able to coerce symbolic
    expressions which inherit from Basic. This can be useful in cooperation
    with SAGE.

    .. warning::
        Note that this function uses ``eval``, and thus shouldn't be used on
        unsanitized input.

    If the argument is already a type that SymPy understands, it will do
    nothing but return that value. This can be used at the beginning of a
    function to ensure you are working with the correct type.

    Examples
    ========

    >>> from sympy import sympify

    >>> sympify(2).is_integer
    True
    >>> sympify(2).is_real
    True

    >>> sympify(2.0).is_real
    True
    >>> sympify("2.0").is_real
    True
    >>> sympify("2e-45").is_real
    True

    If the expression could not be converted, a SympifyError is raised.

    >>> sympify("x***2")
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: "could not parse 'x***2'"

    Locals
    ------

    The sympification happens with access to everything that is loaded
    by ``from sympy import *``; anything used in a string that is not
    defined by that import will be converted to a symbol. In the following,
    the ``bitcount`` function is treated as a symbol and the ``O`` is
    interpreted as the Order object (used with series) and it raises
    an error when used improperly:

    >>> s = 'bitcount(42)'
    >>> sympify(s)
    bitcount(42)
    >>> sympify("O(x)")
    O(x)
    >>> sympify("O + 1")
    Traceback (most recent call last):
    ...
    TypeError: unbound method...

    In order to have ``bitcount`` be recognized it can be imported into a
    namespace dictionary and passed as locals:

    >>> ns = {}
    >>> exec('from sympy.core.evalf import bitcount', ns)
    >>> sympify(s, locals=ns)
    6

    In order to have the ``O`` interpreted as a Symbol, identify it as such
    in the namespace dictionary. This can be done in a variety of ways; all
    three of the following are possibilities:

    >>> from sympy import Symbol
    >>> ns["O"] = Symbol("O")  # method 1
    >>> exec('from sympy.abc import O', ns)  # method 2
    >>> ns.update(dict(O=Symbol("O")))  # method 3
    >>> sympify("O + 1", locals=ns)
    O + 1

    If you want *all* single-letter and Greek-letter variables to be symbols
    then you can use the clashing-symbols dictionaries that have been defined
    there as private variables: _clash1 (single-letter variables), _clash2
    (the multi-letter Greek names) or _clash (both single and multi-letter
    names that are defined in abc).

    >>> from sympy.abc import _clash1
    >>> set(_clash1)
    {'E', 'I', 'N', 'O', 'Q', 'S'}
    >>> sympify('I & Q', _clash1)
    I & Q

    Strict
    ------

    If the option ``strict`` is set to ``True``, only the types for which an
    explicit conversion has been defined are converted. In the other
    cases, a SympifyError is raised.

    >>> print(sympify(None))
    None
    >>> sympify(None, strict=True)
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: None

    Evaluation
    ----------

    If the option ``evaluate`` is set to ``False``, then arithmetic and
    operators will be converted into their SymPy equivalents and the
    ``evaluate=False`` option will be added. Nested ``Add`` or ``Mul`` will
    be denested first. This is done via an AST transformation that replaces
    operators with their SymPy equivalents, so if an operand redefines any
    of those operations, the redefined operators will not be used. If
    argument a is not a string, the mathematical expression is evaluated
    before being passed to sympify, so adding evaluate=False will still
    return the evaluated result of expression.

    >>> sympify('2**2 / 3 + 5')
    19/3
    >>> sympify('2**2 / 3 + 5', evaluate=False)
    2**2/3 + 5
    >>> sympify('4/2+7', evaluate=True)
    9
    >>> sympify('4/2+7', evaluate=False)
    4/2 + 7
    >>> sympify(4/2+7, evaluate=False)
    9.00000000000000

    Extending
    ---------

    To extend ``sympify`` to convert custom objects (not derived from ``Basic``),
    just define a ``_sympy_`` method to your class. You can do that even to
    classes that you do not own by subclassing or adding the method at runtime.

    >>> from sympy import Matrix
    >>> class MyList1(object):
    ...     def __iter__(self):
    ...         yield 1
    ...         yield 2
    ...         return
    ...     def __getitem__(self, i): return list(self)[i]
    ...     def _sympy_(self): return Matrix(self)
    >>> sympify(MyList1())
    Matrix([
    [1],
    [2]])

    If you do not have control over the class definition you could also use the
    ``converter`` global dictionary. The key is the class and the value is a
    function that takes a single argument and returns the desired SymPy
    object, e.g. ``converter[MyList] = lambda x: Matrix(x)``.

    >>> class MyList2(object):   # XXX Do not do this if you control the class!
    ...     def __iter__(self):  #     Use _sympy_!
    ...         yield 1
    ...         yield 2
    ...         return
    ...     def __getitem__(self, i): return list(self)[i]
    >>> from sympy.core.sympify import converter
    >>> converter[MyList2] = lambda x: Matrix(x)
    >>> sympify(MyList2())
    Matrix([
    [1],
    [2]])

    Notes
    =====

    The keywords ``rational`` and ``convert_xor`` are only used
    when the input is a string.

    convert_xor
    -----------

    >>> sympify('x^y',convert_xor=True)
    x**y
    >>> sympify('x^y',convert_xor=False)
    x ^ y

    rational
    --------

    >>> sympify('0.1',rational=False)
    0.1
    >>> sympify('0.1',rational=True)
    1/10

    Sometimes autosimplification during sympification results in expressions
    that are very different in structure than what was entered. Until such
    autosimplification is no longer done, the ``kernS`` function might be of
    some use. In the example below you can see how an expression reduces to
    -1 by autosimplification, but does not do so when ``kernS`` is used.

    >>> from sympy.core.sympify import kernS
    >>> from sympy.abc import x
    >>> -2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1
    -1
    >>> s = '-2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1'
    >>> sympify(s)
    -1
    >>> kernS(s)
    -2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1

    Parameters
    ==========

    a :
        - any object defined in SymPy
        - standard numeric python types: int, long, float, Decimal
        - strings (like "0.09", "2e-19" or 'sin(x)')
        - booleans, including ``None`` (will leave ``None`` unchanged)
        - dict, lists, sets or tuples containing any of the above

    convert_xor : boolean, optional
        If true, treats XOR as exponentiation.
        If False, treats XOR as XOR itself.
        Used only when input is a string.

    locals : any object defined in SymPy, optional
        In order to have strings be recognized it can be imported
        into a namespace dictionary and passed as locals.

    strict : boolean, optional
        If the option strict is set to True, only the types for which
        an explicit conversion has been defined are converted. In the
        other cases, a SympifyError is raised.

    rational : boolean, optional
        If true, converts floats into Rational.
        If false, it lets floats remain as it is.
        Used only when input is a string.

    evaluate : boolean, optional
        If False, then arithmetic and operators will be converted into
        their SymPy equivalents. If True the expression will be evaluated
        and the result will be returned.

    """
    # XXX: If a is a Basic subclass rather than instance (e.g. sin rather than
    # sin(x)) then a.__sympy__ will be the property. Only on the instance will
    # a.__sympy__ give the *value* of the property (True). Since sympify(sin)
    # was used for a long time we allow it to pass. However if strict=True as
    # is the case in internal calls to _sympify then we only allow
    # is_sympy=True.
    #
    # https://github.com/sympy/sympy/issues/20124
    is_sympy = getattr(a, '__sympy__', None)
    if is_sympy is True:
        return a
    elif is_sympy is not None:
        if not strict:
            return a
        else:
            raise SympifyError(a)

    if isinstance(a, CantSympify):
        raise SympifyError(a)
    cls = getattr(a, "__class__", None)
    if cls is None:
        cls = type(a)  # Probably an old-style class
    conv = converter.get(cls, None)
    if conv is not None:
        return conv(a)

    for superclass in getmro(cls):
        try:
            return converter[superclass](a)
        except KeyError:
            continue

    if cls is type(None):
        if strict:
            raise SympifyError(a)
        else:
            return a

    if evaluate is None:
        evaluate = global_parameters.evaluate

    # Support for basic numpy datatypes
    if _is_numpy_instance(a):
        import numpy as np
        if np.isscalar(a):
            return _convert_numpy_types(a, locals=locals,
                convert_xor=convert_xor, strict=strict, rational=rational,
                evaluate=evaluate)

    _sympy_ = getattr(a, "_sympy_", None)
    if _sympy_ is not None:
        try:
            return a._sympy_()
        # XXX: Catches AttributeError: 'SympyConverter' object has no
        # attribute 'tuple'
        # This is probably a bug somewhere but for now we catch it here.
        except AttributeError:
            pass

    if not strict:
        # Put numpy array conversion _before_ float/int, see
        # <https://github.com/sympy/sympy/issues/13924>.
        flat = getattr(a, "flat", None)
        if flat is not None:
            shape = getattr(a, "shape", None)
            if shape is not None:
                from ..tensor.array import Array
                return Array(a.flat, a.shape)  # works with e.g. NumPy arrays

    if not isinstance(a, str):
        if _is_numpy_instance(a):
            import numpy as np
            assert not isinstance(a, np.number)
            if isinstance(a, np.ndarray):
                # Scalar arrays (those with zero dimensions) have sympify
                # called on the scalar element.
                if a.ndim == 0:
                    try:
                        return sympify(a.item(),
                                       locals=locals,
                                       convert_xor=convert_xor,
                                       strict=strict,
                                       rational=rational,
                                       evaluate=evaluate)
                    except SympifyError:
                        pass
        else:
            # float and int can coerce size-one numpy arrays to their lone
            # element.  See issue https://github.com/numpy/numpy/issues/10404.
            for coerce in (float, int):
                try:
                    return sympify(coerce(a))
                except (TypeError, ValueError, AttributeError, SympifyError):
                    continue

    if strict:
        raise SympifyError(a)

    if iterable(a):
        try:
            return type(a)([sympify(x, locals=locals, convert_xor=convert_xor,
                rational=rational, evaluate=evaluate) for x in a])
        except TypeError:
            # Not all iterables are rebuildable with their type.
            pass

    if not isinstance(a, str):
        try:
            a = str(a)
        except Exception as exc:
            raise SympifyError(a, exc)
        from sympy.utilities.exceptions import SymPyDeprecationWarning
        SymPyDeprecationWarning(
            feature="String fallback in sympify",
            useinstead= \
                'sympify(str(obj)) or ' + \
                'sympy.core.sympify.converter or obj._sympy_',
            issue=18066,
            deprecated_since_version='1.6'
        ).warn()

    from sympy.parsing.sympy_parser import (parse_expr, TokenError,
                                            standard_transformations)
    from sympy.parsing.sympy_parser import convert_xor as t_convert_xor
    from sympy.parsing.sympy_parser import rationalize as t_rationalize

    transformations = standard_transformations

    if rational:
        transformations += (t_rationalize,)
    if convert_xor:
        transformations += (t_convert_xor,)

    try:
        a = a.replace('\n', '')
        expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
    except (TokenError, SyntaxError) as exc:
        raise SympifyError('could not parse %r' % a, exc)

    return expr
