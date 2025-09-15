import atexit
import contextlib
import fnmatch
import importlib.util
import itertools
import os
import shutil
import sys
import uuid
import warnings
from enum import Enum
from functools import partial
from os.path import expanduser
from os.path import expandvars
from os.path import isabs
from os.path import sep
from pathlib import Path
from pathlib import PurePath
from posixpath import sep as posix_sep
from types import ModuleType
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import Set
from typing import TypeVar
from typing import Union
import py
from _pytest.compat import assert_never
from _pytest.outcomes import skip
from _pytest.warning_types import PytestWarning
import stat

LOCK_TIMEOUT = 60 * 60 * 24 * 3
_AnyPurePath = TypeVar("_AnyPurePath", bound=PurePath)

def visit(
    path: str, recurse: Callable[["os.DirEntry[str]"], bool]
) -> Iterator["os.DirEntry[str]"]:
    """Walk a directory recursively, in breadth-first order.

    Entries at each directory level are sorted.
    """
    entries = sorted(os.scandir(path), key=lambda entry: entry.name)
    yield from entries
    for entry in entries:
        if entry.is_dir() and recurse(entry):
            yield from visit(entry.path, recurse)
