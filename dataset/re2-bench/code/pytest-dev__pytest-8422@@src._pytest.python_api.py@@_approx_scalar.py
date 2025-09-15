import math
import pprint
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sized
from decimal import Decimal
from numbers import Complex
from types import TracebackType
from typing import Any
from typing import Callable
from typing import cast
from typing import Generic
from typing import Optional
from typing import overload
from typing import Pattern
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
import numpy as np
import numpy as np

_E = TypeVar("_E", bound=BaseException)
raises.Exception = fail.Exception  # type: ignore

class ApproxBase:
    __array_ufunc__ = None
    __array_priority__ = 100
    __hash__ = None  # type: ignore
    def _approx_scalar(self, x) -> "ApproxScalar":
        if isinstance(x, Decimal):
            return ApproxDecimal(x, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
        return ApproxScalar(x, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)