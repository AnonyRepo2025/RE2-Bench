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

class Grouper:
    def __getstate__(self):
        return {
            **vars(self),
            # Convert weak refs to strong ones.
            "_mapping": {k(): [v() for v in vs] for k, vs in self._mapping.items()},
        }