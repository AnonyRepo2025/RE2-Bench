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

class TracebackEntry:
    _repr_style = None  # type: Optional[Literal["short", "long"]]
    exprinfo = None
    source = property(getsource)
    def __str__(self) -> str:
        name = self.frame.code.name
        try:
            line = str(self.statement).lstrip()
        except KeyboardInterrupt:
            raise
        except BaseException:
            line = "???"
        # This output does not quite match Python's repr for traceback entries,
        # but changing it to do so would break certain plugins.  See
        # https://github.com/pytest-dev/pytest/pull/7535/ for details.
        return "  File %r:%d in %s\n  %s\n" % (
            str(self.path),
            self.lineno + 1,
            name,
            line,
        )