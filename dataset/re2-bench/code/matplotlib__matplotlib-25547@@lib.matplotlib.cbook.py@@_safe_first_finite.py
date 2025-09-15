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

def _safe_first_finite(obj, *, skip_nonfinite=True):
    """
    Return the first finite element in *obj* if one is available and skip_nonfinite is
    True. Otherwise return the first element.

    This is a method for internal use.

    This is a type-independent way of obtaining the first finite element, supporting
    both index access and the iterator protocol.
    """
    def safe_isfinite(val):
        if val is None:
            return False
        try:
            return np.isfinite(val) if np.isscalar(val) else True
        except TypeError:
            # This is something that numpy can not make heads or tails
            # of, assume "finite"
            return True
    if skip_nonfinite is False:
        if isinstance(obj, collections.abc.Iterator):
            # needed to accept `array.flat` as input.
            # np.flatiter reports as an instance of collections.Iterator
            # but can still be indexed via [].
            # This has the side effect of re-setting the iterator, but
            # that is acceptable.
            try:
                return obj[0]
            except TypeError:
                pass
            raise RuntimeError("matplotlib does not support generators "
                               "as input")
        return next(iter(obj))
    elif isinstance(obj, np.flatiter):
        # TODO do the finite filtering on this
        return obj[0]
    elif isinstance(obj, collections.abc.Iterator):
        raise RuntimeError("matplotlib does not "
                           "support generators as input")
    else:
        return next((val for val in obj if safe_isfinite(val)), safe_first_element(obj))
