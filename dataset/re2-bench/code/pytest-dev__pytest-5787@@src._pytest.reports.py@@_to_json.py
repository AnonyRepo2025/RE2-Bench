from pprint import pprint
from typing import Optional
import py
from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ReprEntry
from _pytest._code.code import ReprEntryNative
from _pytest._code.code import ReprExceptionInfo
from _pytest._code.code import ReprFileLocation
from _pytest._code.code import ReprFuncArgs
from _pytest._code.code import ReprLocals
from _pytest._code.code import ReprTraceback
from _pytest._code.code import TerminalRepr
from _pytest.outcomes import skip
from _pytest.pathlib import Path



class BaseReport:
    when = None  # type: Optional[str]
    location = None
    passed = property(lambda x: x.outcome == "passed")
    failed = property(lambda x: x.outcome == "failed")
    skipped = property(lambda x: x.outcome == "skipped")
    def _to_json(self):
        """
        This was originally the serialize_report() function from xdist (ca03269).

        Returns the contents of this report as a dict of builtin entries, suitable for
        serialization.

        Experimental method.
        """
        return _report_to_json(self)