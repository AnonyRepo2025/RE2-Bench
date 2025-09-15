import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
from matplotlib import font_manager

_log = logging.getLogger(__name__)
__all__ = ('TickHelper', 'Formatter', 'FixedFormatter',
           'NullFormatter', 'FuncFormatter', 'FormatStrFormatter',
           'StrMethodFormatter', 'ScalarFormatter', 'LogFormatter',
           'LogFormatterExponent', 'LogFormatterMathtext',
           'LogFormatterSciNotation',
           'LogitFormatter', 'EngFormatter', 'PercentFormatter',
           'Locator', 'IndexLocator', 'FixedLocator', 'NullLocator',
           'LinearLocator', 'LogLocator', 'AutoLocator',
           'MultipleLocator', 'MaxNLocator', 'AutoMinorLocator',
           'SymmetricalLogLocator', 'AsinhLocator', 'LogitLocator')

class LogLocator(Locator):
    def tick_values(self, vmin, vmax):
        if self.numticks == 'auto':
            if self.axis is not None:
                numticks = np.clip(self.axis.get_tick_space(), 2, 9)
            else:
                numticks = 9
        else:
            numticks = self.numticks

        b = self._base
        if vmin <= 0.0:
            if self.axis is not None:
                vmin = self.axis.get_minpos()

            if vmin <= 0.0 or not np.isfinite(vmin):
                raise ValueError(
                    "Data has no positive values, and therefore can not be "
                    "log-scaled.")

        _log.debug('vmin %s vmax %s', vmin, vmax)

        if vmax < vmin:
            vmin, vmax = vmax, vmin
        log_vmin = math.log(vmin) / math.log(b)
        log_vmax = math.log(vmax) / math.log(b)

        numdec = math.floor(log_vmax) - math.ceil(log_vmin)

        if isinstance(self._subs, str):
            _first = 2.0 if self._subs == 'auto' else 1.0
            if numdec > 10 or b < 3:
                if self._subs == 'auto':
                    return np.array([])  # no minor or major ticks
                else:
                    subs = np.array([1.0])  # major ticks
            else:
                subs = np.arange(_first, b)
        else:
            subs = self._subs

        # Get decades between major ticks.
        stride = (max(math.ceil(numdec / (numticks - 1)), 1)
                  if mpl.rcParams['_internal.classic_mode'] else
                  numdec // numticks + 1)

        # if we have decided that the stride is as big or bigger than
        # the range, clip the stride back to the available range - 1
        # with a floor of 1.  This prevents getting axis with only 1 tick
        # visible.
        if stride >= numdec:
            stride = max(1, numdec - 1)

        # Does subs include anything other than 1?  Essentially a hack to know
        # whether we're a major or a minor locator.
        have_subs = len(subs) > 1 or (len(subs) == 1 and subs[0] != 1.0)

        decades = np.arange(math.floor(log_vmin) - stride,
                            math.ceil(log_vmax) + 2 * stride, stride)

        if hasattr(self, '_transform'):
            ticklocs = self._transform.inverted().transform(decades)
            if have_subs:
                if stride == 1:
                    ticklocs = np.ravel(np.outer(subs, ticklocs))
                else:
                    # No ticklocs if we have >1 decade between major ticks.
                    ticklocs = np.array([])
        else:
            if have_subs:
                if stride == 1:
                    ticklocs = np.concatenate(
                        [subs * decade_start for decade_start in b ** decades])
                else:
                    ticklocs = np.array([])
            else:
                ticklocs = b ** decades

        _log.debug('ticklocs %r', ticklocs)
        if (len(subs) > 1
                and stride == 1
                and ((vmin <= ticklocs) & (ticklocs <= vmax)).sum() <= 1):
            # If we're a minor locator *that expects at least two ticks per
            # decade* and the major locator stride is 1 and there's no more
            # than one minor tick, switch to AutoLocator.
            return AutoLocator().tick_values(vmin, vmax)
        else:
            return self.raise_if_exceeds(ticklocs)