import contextlib
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any, Generator, Iterator, List, Optional, Sequence, Tuple, Union
from sphinx.util import logging
from sphinx.util.inspect import safe_getattr

logger = logging.getLogger(__name__)

def _make_subclass(name: str, module: str, superclass: Any = _MockObject,
                   attributes: Any = None, decorator_args: Tuple = ()) -> Any:
    attrs = {'__module__': module,
             '__display_name__': module + '.' + name,
             '__name__': name,
             '__sphinx_decorator_args__': decorator_args}
    attrs.update(attributes or {})

    return type(name, (superclass,), attrs)
