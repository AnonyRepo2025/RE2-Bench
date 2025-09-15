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
    def __init__(self, obj):
        self._obj = obj
    def _apply(self, f, dtype=None):
        # TODO handling of na values ?
        if dtype is None:
            dtype = self._obj.dtype

        g = np.vectorize(f, otypes=[dtype])
        return apply_ufunc(g, self._obj, dask="parallelized", output_dtypes=[dtype])
    def upper(self):
        """
        Convert strings in the array to uppercase.

        Returns
        -------
        uppered : same type as values
        """
        return self._apply(lambda x: x.upper())
    def contains(self, pat, case=True, flags=0, regex=True):
        """
        Test if pattern or regex is contained within each string of the array.

        Return boolean array based on whether a given pattern or regex is
        contained within a string of the array.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Flags to pass through to the re module, e.g. re.IGNORECASE.
        regex : bool, default True
            If True, assumes the pat is a regular expression.
            If False, treats the pat as a literal string.

        Returns
        -------
        contains : array of bool
            An array of boolean values indicating whether the
            given pattern is contained within the string of each element
            of the array.
        """
        pat = self._obj.dtype.type(pat)
        if regex:
            if not case:
                flags |= re.IGNORECASE

            regex = re.compile(pat, flags=flags)

            if regex.groups > 0:  # pragma: no cover
                raise ValueError("This pattern has match groups.")

            f = lambda x: bool(regex.search(x))
        else:
            if case:
                f = lambda x: pat in x
            else:
                uppered = self._obj.str.upper()
                return uppered.str.contains(pat.upper(), regex=False)

        return self._apply(f, dtype=bool)