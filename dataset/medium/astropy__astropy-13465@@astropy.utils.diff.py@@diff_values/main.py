import difflib
import functools
import sys
import numbers
import numpy as np
from .misc import indent

__all__ = ['fixed_width_indent', 'diff_values', 'report_diff_values',
           'where_not_allclose']
fixed_width_indent = functools.partial(indent, width=2)

def diff_values(a, b, rtol=0.0, atol=0.0):
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return False
        return not np.allclose(a, b, rtol=rtol, atol=atol)
    else:
        return a != b
