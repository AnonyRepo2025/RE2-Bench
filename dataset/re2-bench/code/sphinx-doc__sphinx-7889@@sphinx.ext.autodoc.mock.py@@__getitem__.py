import contextlib
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import FunctionType, MethodType, ModuleType
from typing import Any, Generator, Iterator, List, Sequence, Tuple, Union
from sphinx.util import logging

logger = logging.getLogger(__name__)

class _MockObject:
    __display_name__ = '_MockObject'
    __sphinx_mock__ = True
    def __getitem__(self, key: Any) -> "_MockObject":
        return _make_subclass(str(key), self.__display_name__, self.__class__)()