import functools
import inspect
import io
import re
import sys
from contextlib import contextmanager
from inspect import Parameter
from inspect import signature
import attr
import py
import _pytest
from _pytest._io.saferepr import saferepr
from _pytest.outcomes import fail
from _pytest.outcomes import TEST_OUTCOME
from contextlib import nullcontext  # noqa
from types import ModuleType
import pytest
import warnings
from _pytest.deprecated import FUNCARGNAMES

NOTSET = object()
MODULE_NOT_FOUND_ERROR = (
    "ModuleNotFoundError" if sys.version_info[:2] >= (3, 6) else "ImportError"
)
REGEX_TYPE = type(re.compile(""))
_non_printable_ascii_translate_table = {
    i: "\\x{:02x}".format(i) for i in range(128) if i not in range(32, 127)
}
STRING_TYPES = bytes, str
COLLECT_FAKEMODULE_ATTRIBUTES = (
    "Collector",
    "Module",
    "Function",
    "Instance",
    "Session",
    "Item",
    "Class",
    "File",
    "_fillfuncargs",
)

def num_mock_patch_args(function):
    """ return number of arguments used up by mock arguments (if any) """
    patchings = getattr(function, "patchings", None)
    if not patchings:
        return 0

    mock_sentinel = getattr(sys.modules.get("mock"), "DEFAULT", object())
    ut_mock_sentinel = getattr(sys.modules.get("unittest.mock"), "DEFAULT", object())

    return len(
        [
            p
            for p in patchings
            if not p.attribute_name
            and (p.new is mock_sentinel or p.new is ut_mock_sentinel)
        ]
    )
