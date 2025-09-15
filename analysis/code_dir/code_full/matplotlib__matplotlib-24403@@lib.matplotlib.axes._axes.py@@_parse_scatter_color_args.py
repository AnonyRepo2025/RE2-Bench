def __getitem__(self, key):
    if key in _deprecated_map:
        version, alt_key, alt_val, inverse_alt = _deprecated_map[key]
        _api.warn_deprecated(version, name=key, obj_type='rcparam', alternative=alt_key)
        return inverse_alt(dict.__getitem__(self, alt_key))
    elif key in _deprecated_ignore_map:
        version, alt_key = _deprecated_ignore_map[key]
        _api.warn_deprecated(version, name=key, obj_type='rcparam', alternative=alt_key)
        return dict.__getitem__(self, alt_key) if alt_key else None
    elif key == 'backend' and self is globals().get('rcParams'):
        val = dict.__getitem__(self, key)
        if val is rcsetup._auto_backend_sentinel:
            from matplotlib import pyplot as plt
            plt.switch_backend(rcsetup._auto_backend_sentinel)
    return dict.__getitem__(self, key)

def _safe_first_finite(obj, *, skip_nonfinite=True):

    def safe_isfinite(val):
        if val is None:
            return False
        try:
            return np.isfinite(val) if np.isscalar(val) else True
        except TypeError:
            return True
    if skip_nonfinite is False:
        if isinstance(obj, collections.abc.Iterator):
            try:
                return obj[0]
            except TypeError:
                pass
            raise RuntimeError('matplotlib does not support generators as input')
        return next(iter(obj))
    elif isinstance(obj, np.flatiter):
        return obj[0]
    elif isinstance(obj, collections.abc.Iterator):
        raise RuntimeError('matplotlib does not support generators as input')
    else:
        return next((val for val in obj if safe_isfinite(val)))

def safe_isfinite(val):
    if val is None:
        return False
    try:
        return np.isfinite(val) if np.isscalar(val) else True
    except TypeError:
        return True



import functools
import itertools
import logging
import math
from numbers import Integral, Number
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import _AxesBase, _TransformedBoundsLocator, _process_plot_format
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
from builtins import range
_log = logging.getLogger(__name__)

class Axes(_AxesBase):
    annotate.__doc__ = mtext.Annotation.__init__.__doc__
    fill_between = _preprocess_data(_docstring.dedent_interpd(fill_between), replace_names=['x', 'y1', 'y2', 'where'])
    fill_betweenx = _preprocess_data(_docstring.dedent_interpd(fill_betweenx), replace_names=['y', 'x1', 'x2', 'where'])
    table = mtable.table
    stackplot = _preprocess_data()(mstack.stackplot)
    streamplot = _preprocess_data(replace_names=['x', 'y', 'u', 'v', 'start_points'])(mstream.streamplot)
    tricontour = mtri.tricontour
    tricontourf = mtri.tricontourf
    tripcolor = mtri.tripcolor
    triplot = mtri.triplot

    @staticmethod
    def _parse_scatter_color_args(c, edgecolors, kwargs, xsize, get_next_color_func):
        facecolors = kwargs.pop('facecolors', None)
        facecolors = kwargs.pop('facecolor', facecolors)
        edgecolors = kwargs.pop('edgecolor', edgecolors)
        kwcolor = kwargs.pop('color', None)
        if kwcolor is not None and c is not None:
            raise ValueError("Supply a 'c' argument or a 'color' kwarg but not both; they differ but their functionalities overlap.")
        if kwcolor is not None:
            try:
                mcolors.to_rgba_array(kwcolor)
            except ValueError as err:
                raise ValueError("'color' kwarg must be a color or sequence of color specs.  For a sequence of values to be color-mapped, use the 'c' argument instead.") from err
            if edgecolors is None:
                edgecolors = kwcolor
            if facecolors is None:
                facecolors = kwcolor
        if edgecolors is None and (not mpl.rcParams['_internal.classic_mode']):
            edgecolors = mpl.rcParams['scatter.edgecolors']
        c_was_none = c is None
        if c is None:
            c = facecolors if facecolors is not None else 'b' if mpl.rcParams['_internal.classic_mode'] else get_next_color_func()
        c_is_string_or_strings = isinstance(c, str) or (np.iterable(c) and len(c) > 0 and isinstance(cbook._safe_first_finite(c), str))

        def invalid_shape_exception(csize, xsize):
            return ValueError(f"'c' argument has {csize} elements, which is inconsistent with 'x' and 'y' with size {xsize}.")
        c_is_mapped = False
        valid_shape = True
        if not c_was_none and kwcolor is None and (not c_is_string_or_strings):
            try:
                c = np.asanyarray(c, dtype=float)
            except ValueError:
                pass
            else:
                if c.shape == (1, 4) or c.shape == (1, 3):
                    c_is_mapped = False
                    if c.size != xsize:
                        valid_shape = False
                elif c.size == xsize:
                    c = c.ravel()
                    c_is_mapped = True
                else:
                    if c.shape in ((3,), (4,)):
                        _log.warning('*c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2D array with a single row if you intend to specify the same RGB or RGBA value for all points.')
                    valid_shape = False
        if not c_is_mapped:
            try:
                colors = mcolors.to_rgba_array(c)
            except (TypeError, ValueError) as err:
                if 'RGBA values should be within 0-1 range' in str(err):
                    raise
                else:
                    if not valid_shape:
                        raise invalid_shape_exception(c.size, xsize) from err
                    raise ValueError(f"'c' argument must be a color, a sequence of colors, or a sequence of numbers, not {c!r}") from err
            else:
                if len(colors) not in (0, 1, xsize):
                    raise invalid_shape_exception(len(colors), xsize)
        else:
            colors = None
        return (c, colors, edgecolors)