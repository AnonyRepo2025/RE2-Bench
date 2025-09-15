import codecs
import re
import textwrap
import numpy as np
from .computation import apply_ufunc

_cpython_optimized_encoders = (
    "utf-8",
    "utf8",
    "latin-1",
    "latin1",
    "iso-8859-1",
    "mbcs",
    "ascii",
)
_cpython_optimized_decoders = _cpython_optimized_encoders + ("utf-16", "utf-32")

class StringAccessor:
    __slots__ = ("_obj",)
    def match(self, pat, case=True, flags=0):
        """
        Determine if each string in the array matches a regular expression.

        Parameters
        ----------
        pat : string
            Character sequence or regular expression
        case : boolean, default True
            If True, case sensitive
        flags : int, default 0 (no flags)
            re module flags, e.g. re.IGNORECASE

        Returns
        -------
        matched : array of bool
        """
        if not case:
            flags |= re.IGNORECASE

        pat = self._obj.dtype.type(pat)
        regex = re.compile(pat, flags=flags)
        f = lambda x: bool(regex.match(x))
        return self._apply(f, dtype=bool)