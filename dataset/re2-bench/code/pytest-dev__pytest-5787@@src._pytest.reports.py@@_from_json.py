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
    @classmethod
    def _from_json(cls, reportdict):
        """
        This was originally the serialize_report() function from xdist (ca03269).

        Factory method that returns either a TestReport or CollectReport, depending on the calling
        class. It's the callers responsibility to know which class to pass here.

        Experimental method.
        """
        kwargs = _report_kwargs_from_json(reportdict)
        return cls(**kwargs)