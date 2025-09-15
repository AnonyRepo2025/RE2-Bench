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
    def startswith(self, pat):
        """
        Test if the start of each string in the array matches a pattern.

        Parameters
        ----------
        pat : str
            Character sequence. Regular expressions are not accepted.

        Returns
        -------
        startswith : array of bool
            An array of booleans indicating whether the given pattern matches
            the start of each string element.
        """
        pat = self._obj.dtype.type(pat)
        f = lambda x: x.startswith(pat)
        return self._apply(f, dtype=bool)