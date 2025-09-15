import os
import re
import tempfile
from pathlib import Path
from typing import Optional
import attr
import py
from .pathlib import ensure_reset_dir
from .pathlib import LOCK_TIMEOUT
from .pathlib import make_numbered_dir
from .pathlib import make_numbered_dir_with_cleanup
from _pytest.compat import final
from _pytest.config import Config
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
import getpass



class TempPathFactory:
    _given_basetemp = attr.ib(type=Optional[Path])
    _trace = attr.ib()
    _basetemp = attr.ib(type=Optional[Path])
    def getbasetemp(self) -> Path:
        """Return base temporary directory."""
        if self._basetemp is not None:
            return self._basetemp

        if self._given_basetemp is not None:
            basetemp = self._given_basetemp
            ensure_reset_dir(basetemp)
            basetemp = basetemp.resolve()
        else:
            from_env = os.environ.get("PYTEST_DEBUG_TEMPROOT")
            temproot = Path(from_env or tempfile.gettempdir()).resolve()
            user = get_user() or "unknown"
            # use a sub-directory in the temproot to speed-up
            # make_numbered_dir() call
            rootdir = temproot.joinpath(f"pytest-of-{user}")
            try:
                rootdir.mkdir(exist_ok=True)
            except OSError:
                # getuser() likely returned illegal characters for the platform, use unknown back off mechanism
                rootdir = temproot.joinpath("pytest-of-unknown")
                rootdir.mkdir(exist_ok=True)
            basetemp = make_numbered_dir_with_cleanup(
                prefix="pytest-", root=rootdir, keep=3, lock_timeout=LOCK_TIMEOUT
            )
        assert basetemp is not None, basetemp
        self._basetemp = t = basetemp
        self._trace("new basetemp", t)
        return t