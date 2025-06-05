from contextlib import ExitStack
from enum import Enum
import functools
import importlib
import inspect
import logging
from numbers import Number
import re
import sys
import threading
import time
from cycler import cycler
import matplotlib
import matplotlib.colorbar
import matplotlib.image
from matplotlib import _api
from matplotlib import rcsetup, style
from matplotlib import _pylab_helpers, interactive
from matplotlib import cbook
from matplotlib import _docstring
from matplotlib.backend_bases import FigureCanvasBase, MouseButton
from matplotlib.figure import Figure, FigureBase, figaspect
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib import rcParams, rcParamsDefault, get_backend, rcParamsOrig
from matplotlib.rcsetup import interactive_bk as _interactive_bk
from matplotlib.artist import Artist
from matplotlib.axes import Axes, Subplot
from matplotlib.projections import PolarAxes
from matplotlib import mlab  # for detrend_none, window_hanning
from matplotlib.scale import get_scale_names
from matplotlib import cm
from matplotlib.cm import _colormaps as colormaps, get_cmap, register_cmap
from matplotlib.colors import _color_sequences as color_sequences
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import Button, Slider, Widget
from .ticker import (
    TickHelper, Formatter, FixedFormatter, NullFormatter, FuncFormatter,
    FormatStrFormatter, ScalarFormatter, LogFormatter, LogFormatterExponent,
    LogFormatterMathtext, Locator, IndexLocator, FixedLocator, NullLocator,
    LinearLocator, LogLocator, AutoLocator, MultipleLocator, MaxNLocator)
from IPython.core.pylabtools import backend2gui
import matplotlib.backends
from matplotlib import patheffects
from IPython import get_ipython

_log = logging.getLogger(__name__)
_ReplDisplayHook = Enum("_ReplDisplayHook", ["NONE", "PLAIN", "IPYTHON"])
_REPL_DISPLAYHOOK = _ReplDisplayHook.NONE
draw_all = _pylab_helpers.Gcf.draw_all
_backend_mod = None
_NON_PLOT_COMMANDS = {
    'connect', 'disconnect', 'get_current_fig_manager', 'ginput',
    'new_figure_manager', 'waitforbuttonpress'}

def xticks(ticks=None, labels=None, *, minor=False, **kwargs):
    ax = gca()

    if ticks is None:
        locs = ax.get_xticks(minor=minor)
        if labels is not None:
            raise TypeError("xticks(): Parameter 'labels' can't be set "
                            "without setting 'ticks'")
    else:
        locs = ax.set_xticks(ticks, minor=minor)

    if labels is None:
        labels = ax.get_xticklabels(minor=minor)
        for l in labels:
            l._internal_update(kwargs)
    else:
        labels = ax.set_xticklabels(labels, minor=minor, **kwargs)

    return locs, labels
