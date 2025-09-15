import atexit
import contextlib
import fnmatch
import importlib.util
import itertools
import os
import shutil
import sys
import types
import uuid
import warnings
from enum import Enum
from errno import EBADF
from errno import ELOOP
from errno import ENOENT
from errno import ENOTDIR
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
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from _pytest.compat import assert_never
from _pytest.outcomes import skip
from _pytest.warning_types import PytestWarning
import stat

LOCK_TIMEOUT = 60 * 60 * 24 * 3
_AnyPurePath = TypeVar("_AnyPurePath", bound=PurePath)
_IGNORED_ERRORS = (ENOENT, ENOTDIR, EBADF, ELOOP)
_IGNORED_WINERRORS = (
    21,  # ERROR_NOT_READY - drive exists but is not accessible
    1921,  # ERROR_CANT_RESOLVE_FILENAME - fix for broken symlink pointing to itself
)

def on_rm_rf_error(
    func,
    path: str,
    excinfo: Union[
        BaseException,
        Tuple[Type[BaseException], BaseException, Optional[types.TracebackType]],
    ],
    *,
    start_path: Path,
) -> bool:
    """Handle known read-only errors during rmtree.

    The returned value is used only by our own tests.
    """
    if isinstance(excinfo, BaseException):
        exc = excinfo
    else:
        exc = excinfo[1]

    # Another process removed the file in the middle of the "rm_rf" (xdist for example).
    # More context: https://github.com/pytest-dev/pytest/issues/5974#issuecomment-543799018
    if isinstance(exc, FileNotFoundError):
        return False

    if not isinstance(exc, PermissionError):
        warnings.warn(
            PytestWarning(f"(rm_rf) error removing {path}\n{type(exc)}: {exc}")
        )
        return False

    if func not in (os.rmdir, os.remove, os.unlink):
        if func not in (os.open,):
            warnings.warn(
                PytestWarning(
                    "(rm_rf) unknown function {} when removing {}:\n{}: {}".format(
                        func, path, type(exc), exc
                    )
                )
            )
        return False

    # Chmod + retry.
    import stat

    def chmod_rw(p: str) -> None:
        mode = os.stat(p).st_mode
        os.chmod(p, mode | stat.S_IRUSR | stat.S_IWUSR)

    # For files, we need to recursively go upwards in the directories to
    # ensure they all are also writable.
    p = Path(path)
    if p.is_file():
        for parent in p.parents:
            chmod_rw(str(parent))
            # Stop when we reach the original path passed to rm_rf.
            if parent == start_path:
                break
    chmod_rw(str(path))

    func(path)
    return True
