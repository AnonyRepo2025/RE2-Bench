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
    def len(self):
        """
        Compute the length of each string in the array.

        Returns
        -------
        lengths array : array of int
        """
        return self._apply(len, dtype=int)