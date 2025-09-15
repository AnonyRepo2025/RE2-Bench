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
    def endswith(self, pat):
        """
        Test if the end of each string in the array matches a pattern.

        Parameters
        ----------
        pat : str
            Character sequence. Regular expressions are not accepted.

        Returns
        -------
        endswith : array of bool
            A Series of booleans indicating whether the given pattern matches
            the end of each string element.
        """
        pat = self._obj.dtype.type(pat)
        f = lambda x: x.endswith(pat)
        return self._apply(f, dtype=bool)