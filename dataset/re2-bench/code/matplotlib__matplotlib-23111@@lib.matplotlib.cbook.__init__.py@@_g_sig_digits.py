import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import warnings
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
import gc
from matplotlib.artist import Artist
from gi.repository import GLib
import bz2

ls_mapper = {'-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'}
ls_mapper_r = {v: k for k, v in ls_mapper.items()}
STEP_LOOKUP_MAP = {'default': lambda x, y: (x, y),
                   'steps': pts_to_prestep,
                   'steps-pre': pts_to_prestep,
                   'steps-post': pts_to_poststep,
                   'steps-mid': pts_to_midstep}

def _g_sig_digits(value, delta):
    """
    Return the number of significant digits to %g-format *value*, assuming that
    it is known with an error of *delta*.
    """
    if delta == 0:
        # delta = 0 may occur when trying to format values over a tiny range;
        # in that case, replace it by the distance to the closest float.
        delta = abs(np.spacing(value))
    # If e.g. value = 45.67 and delta = 0.02, then we want to round to 2 digits
    # after the decimal point (floor(log10(0.02)) = -2); 45.67 contributes 2
    # digits before the decimal point (floor(log10(45.67)) + 1 = 2): the total
    # is 4 significant digits.  A value of 0 contributes 1 "digit" before the
    # decimal point.
    # For inf or nan, the precision doesn't matter.
    return max(
        0,
        (math.floor(math.log10(abs(value))) + 1 if value else 1)
        - math.floor(math.log10(delta))) if math.isfinite(value) else 0
