import inspect
import re
import sys
import traceback
from inspect import CO_VARARGS
from inspect import CO_VARKEYWORDS
from io import StringIO
from traceback import format_exception_only
from types import CodeType
from types import FrameType
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union
from weakref import ref
import attr
import pluggy
import py
import _pytest
from _pytest._code.source import findsource
from _pytest._code.source import getrawcode
from _pytest._code.source import getstatementrange_ast
from _pytest._code.source import Source
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import safeformat
from _pytest._io.saferepr import saferepr
from _pytest.compat import ATTRS_EQ_FIELD
from _pytest.compat import get_real_func
from _pytest.compat import overload
from _pytest.compat import TYPE_CHECKING
from typing import Type
from typing_extensions import Literal
from weakref import ReferenceType

co_equal = compile(
    "__recursioncache_locals_1 == __recursioncache_locals_2", "?", "eval"
)
_E = TypeVar("_E", bound=BaseException, covariant=True)
_PLUGGY_DIR = py.path.local(pluggy.__file__.rstrip("oc"))
_PYTEST_DIR = py.path.local(_pytest.__file__).dirpath()
_PY_DIR = py.path.local(py.__file__).dirpath()

class ExceptionInfo(Generic[_E]
):
    _assert_start_repr = "AssertionError('assert "
    _excinfo = attr.ib(type=Optional[Tuple["Type[_E]", "_E", TracebackType]])
    _striptext = attr.ib(type=str, default="")
    _traceback = attr.ib(type=Optional[Traceback], default=None)
    @property
    def value(self) -> _E:
        """The exception value."""
        assert (
            self._excinfo is not None
        ), ".value can only be used after the context manager exits"
        return self._excinfo[1]
    def match(self, regexp: "Union[str, Pattern]") -> "Literal[True]":
        """Check whether the regular expression `regexp` matches the string
        representation of the exception using :func:`python:re.search`.

        If it matches `True` is returned, otherwise an `AssertionError` is raised.
        """
        __tracebackhide__ = True
        msg = "Regex pattern {!r} does not match {!r}."
        if regexp == str(self.value):
            msg += " Did you mean to `re.escape()` the regex?"
        assert re.search(regexp, str(self.value)), msg.format(regexp, str(self.value))
        # Return True to allow for "assert excinfo.match()".
        return True