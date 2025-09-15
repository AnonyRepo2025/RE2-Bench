import os
import re
import tempfile
from typing import Optional
import attr
import py
import pytest
from .pathlib import ensure_reset_dir
from .pathlib import LOCK_TIMEOUT
from .pathlib import make_numbered_dir
from .pathlib import make_numbered_dir_with_cleanup
from .pathlib import Path
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
import getpass



class TempdirFactory:
    _tmppath_factory = attr.ib(type=TempPathFactory)
    def mktemp(self, basename: str, numbered: bool = True) -> py.path.local:
        """
        Same as :meth:`TempPathFactory.mkdir`, but returns a ``py.path.local`` object.
        """
        return py.path.local(self._tmppath_factory.mktemp(basename, numbered).resolve())