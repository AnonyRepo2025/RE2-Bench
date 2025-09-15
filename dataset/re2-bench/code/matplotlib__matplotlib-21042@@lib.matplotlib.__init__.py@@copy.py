import atexit
from collections import namedtuple
from collections.abc import MutableMapping
import contextlib
import functools
import importlib
import inspect
from inspect import Parameter
import locale
import logging
import os
from pathlib import Path
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import numpy
from packaging.version import parse as parse_version
from . import _api, _version, cbook, docstring, rcsetup
from matplotlib.cbook import MatplotlibDeprecationWarning, sanitize_sequence
from matplotlib.cbook import mplDeprecation  # deprecated
from matplotlib.rcsetup import validate_backend, cycler
from matplotlib.cm import _colormaps as colormaps
from . import ft2font
import ssl
import matplotlib as mpl
from matplotlib import ft2font
import setuptools_scm
import certifi
import urllib.request
from .style.core import STYLE_BLACKLIST
from .style.core import STYLE_BLACKLIST
from .style.core import STYLE_BLACKLIST
import pytest
import winreg
from matplotlib import pyplot as plt

_log = logging.getLogger(__name__)
__bibtex__ = r"""@Article{Hunter:2007,
  Author    = {Hunter, J. D.},
  Title     = {Matplotlib: A 2D graphics environment},
  Journal   = {Computing in Science \& Engineering},
  Volume    = {9},
  Number    = {3},
  Pages     = {90--95},
  abstract  = {Matplotlib is a 2D graphics package used for Python
  for application development, interactive scripting, and
  publication-quality image generation across user
  interfaces and operating systems.},
  publisher = {IEEE COMPUTER SOC},
  year      = 2007
}"""
_VersionInfo = namedtuple('_VersionInfo',
                          'major, minor, micro, releaselevel, serial')
_ExecInfo = namedtuple("_ExecInfo", "executable version")
_deprecated_map = {}
_deprecated_ignore_map = {}
_deprecated_remain_as_none = {}
rcParamsDefault = _rc_params_in_file(
    cbook._get_data_path("matplotlibrc"),
    # Strip leading comment.
    transform=lambda line: line[1:] if line.startswith("#") else line,
    fail_on_error=True)
rcParams = RcParams()  # The global instance.
rcParamsOrig = rcParams.copy()
default_test_modules = [
    'matplotlib.tests',
    'mpl_toolkits.tests',
]
test.__test__ = False  # pytest: this function is not a test

class RcParams(MutableMapping, dict):
    validate = rcsetup._validators
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
    def __iter__(self):
        """Yield sorted list of keys."""
        with _api.suppress_matplotlib_deprecation_warning():
            yield from sorted(dict.__iter__(self))
    def copy(self):
        rccopy = RcParams()
        for k in self:  # Skip deprecations and revalidation.
            dict.__setitem__(rccopy, k, dict.__getitem__(self, k))
        return rccopy