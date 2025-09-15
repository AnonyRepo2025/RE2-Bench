import math
import pprint
from collections.abc import Collection
from collections.abc import Sized
from decimal import Decimal
from numbers import Complex
from types import TracebackType
from typing import Any
from typing import Callable
from typing import cast
from typing import Generic
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import _pytest._code
from _pytest.compat import final
from _pytest.compat import STRING_TYPES
from _pytest.outcomes import fail
from numpy import ndarray
import sys
import itertools
import math
import numpy as np
import numpy as np
import math
import math

E = TypeVar("E", bound=BaseException)
raises.Exception = fail.Exception  # type: ignore

def raises(
    expected_exception: Union[Type[E], Tuple[Type[E], ...]], *args: Any, **kwargs: Any
) -> Union["RaisesContext[E]", _pytest._code.ExceptionInfo[E]]:
    r"""Assert that a code block/function call raises ``expected_exception``
    or raise a failure exception otherwise.

    :kwparam match:
        If specified, a string containing a regular expression,
        or a regular expression object, that is tested against the string
        representation of the exception using :py:func:`re.search`. To match a literal
        string that may contain :std:ref:`special characters <re-syntax>`, the pattern can
        first be escaped with :py:func:`re.escape`.

        (This is only used when :py:func:`pytest.raises` is used as a context manager,
        and passed through to the function otherwise.
        When using :py:func:`pytest.raises` as a function, you can use:
        ``pytest.raises(Exc, func, match="passed on").match("my pattern")``.)

    .. currentmodule:: _pytest._code

    Use ``pytest.raises`` as a context manager, which will capture the exception of the given
    type::

        >>> import pytest
        >>> with pytest.raises(ZeroDivisionError):
        ...    1/0

    If the code block does not raise the expected exception (``ZeroDivisionError`` in the example
    above), or no exception at all, the check will fail instead.

    You can also use the keyword argument ``match`` to assert that the
    exception matches a text or regex::

        >>> with pytest.raises(ValueError, match='must be 0 or None'):
        ...     raise ValueError("value must be 0 or None")

        >>> with pytest.raises(ValueError, match=r'must be \d+$'):
        ...     raise ValueError("value must be 42")

    The context manager produces an :class:`ExceptionInfo` object which can be used to inspect the
    details of the captured exception::

        >>> with pytest.raises(ValueError) as exc_info:
        ...     raise ValueError("value must be 42")
        >>> assert exc_info.type is ValueError
        >>> assert exc_info.value.args[0] == "value must be 42"

    .. note::

       When using ``pytest.raises`` as a context manager, it's worthwhile to
       note that normal context manager rules apply and that the exception
       raised *must* be the final line in the scope of the context manager.
       Lines of code after that, within the scope of the context manager will
       not be executed. For example::

           >>> value = 15
           >>> with pytest.raises(ValueError) as exc_info:
           ...     if value > 10:
           ...         raise ValueError("value must be <= 10")
           ...     assert exc_info.type is ValueError  # this will not execute

       Instead, the following approach must be taken (note the difference in
       scope)::

           >>> with pytest.raises(ValueError) as exc_info:
           ...     if value > 10:
           ...         raise ValueError("value must be <= 10")
           ...
           >>> assert exc_info.type is ValueError

    **Using with** ``pytest.mark.parametrize``

    When using :ref:`pytest.mark.parametrize ref`
    it is possible to parametrize tests such that
    some runs raise an exception and others do not.

    See :ref:`parametrizing_conditional_raising` for an example.

    **Legacy form**

    It is possible to specify a callable by passing a to-be-called lambda::

        >>> raises(ZeroDivisionError, lambda: 1/0)
        <ExceptionInfo ...>

    or you can specify an arbitrary callable with arguments::

        >>> def f(x): return 1/x
        ...
        >>> raises(ZeroDivisionError, f, 0)
        <ExceptionInfo ...>
        >>> raises(ZeroDivisionError, f, x=0)
        <ExceptionInfo ...>

    The form above is fully supported but discouraged for new code because the
    context manager form is regarded as more readable and less error-prone.

    .. note::
        Similar to caught exception objects in Python, explicitly clearing
        local references to returned ``ExceptionInfo`` objects can
        help the Python interpreter speed up its garbage collection.

        Clearing those references breaks a reference cycle
        (``ExceptionInfo`` --> caught exception --> frame stack raising
        the exception --> current frame stack --> local variables -->
        ``ExceptionInfo``) which makes Python keep all objects referenced
        from that cycle (including all local variables in the current
        frame) alive until the next cyclic garbage collection run.
        More detailed information can be found in the official Python
        documentation for :ref:`the try statement <python:try>`.
    """
    __tracebackhide__ = True

    if not expected_exception:
        raise ValueError(
            f"Expected an exception type or a tuple of exception types, but got `{expected_exception!r}`. "
            f"Raising exceptions is already understood as failing the test, so you don't need "
            f"any special code to say 'this should never raise an exception'."
        )
    if isinstance(expected_exception, type):
        excepted_exceptions: Tuple[Type[E], ...] = (expected_exception,)
    else:
        excepted_exceptions = expected_exception
    for exc in excepted_exceptions:
        if not isinstance(exc, type) or not issubclass(exc, BaseException):
            msg = "expected exception must be a BaseException type, not {}"  # type: ignore[unreachable]
            not_a = exc.__name__ if isinstance(exc, type) else type(exc).__name__
            raise TypeError(msg.format(not_a))

    message = f"DID NOT RAISE {expected_exception}"

    if not args:
        match: Optional[Union[str, Pattern[str]]] = kwargs.pop("match", None)
        if kwargs:
            msg = "Unexpected keyword arguments passed to pytest.raises: "
            msg += ", ".join(sorted(kwargs))
            msg += "\nUse context-manager form instead?"
            raise TypeError(msg)
        return RaisesContext(expected_exception, message, match)
    else:
        func = args[0]
        if not callable(func):
            raise TypeError(f"{func!r} object (type: {type(func)}) must be callable")
        try:
            func(*args[1:], **kwargs)
        except expected_exception as e:
            # We just caught the exception - there is a traceback.
            assert e.__traceback__ is not None
            return _pytest._code.ExceptionInfo.from_exc_info(
                (type(e), e, e.__traceback__)
            )
    fail(message)
