import ast
import enum
import re
import types
from typing import Callable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Sequence
import attr
from _pytest.compat import TYPE_CHECKING
from typing import NoReturn

__all__ = [
    "Expression",
    "ParseError",
]
IDENT_PREFIX = "$"

class MatcherAdapter(Mapping[(str, bool)]
):
    def __getitem__(self, key: str) -> bool:
        return self.matcher(key[len(IDENT_PREFIX) :])