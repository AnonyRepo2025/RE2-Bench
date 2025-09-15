import atexit
import contextlib
import fnmatch
import itertools
import os
import shutil
import sys
import uuid
import warnings
from functools import partial
from os.path import expanduser
from os.path import expandvars
from os.path import isabs
from os.path import sep
from posixpath import sep as posix_sep
from typing import Iterable
from typing import Iterator
from typing import Set
from typing import TypeVar
from typing import Union
from _pytest.outcomes import skip
from _pytest.warning_types import PytestWarning
from pathlib import Path, PurePath
from pathlib2 import Path, PurePath
import stat

__all__ = ["Path", "PurePath"]
LOCK_TIMEOUT = 60 * 60 * 3
_AnyPurePath = TypeVar("_AnyPurePath", bound=PurePath)

def ensure_deletable(path: Path, consider_lock_dead_if_created_before: float) -> bool:
    """checks if a lock exists and breaks it if its considered dead"""
    if path.is_symlink():
        return False
    lock = get_lock_path(path)
    if not lock.exists():
        return True
    try:
        lock_time = lock.stat().st_mtime
    except Exception:
        return False
    else:
        if lock_time < consider_lock_dead_if_created_before:
            # wa want to ignore any errors while trying to remove the lock such as:
            # - PermissionDenied, like the file permissions have changed since the lock creation
            # - FileNotFoundError, in case another pytest process got here first.
            # and any other cause of failure.
            with contextlib.suppress(OSError):
                lock.unlink()
                return True
        return False
