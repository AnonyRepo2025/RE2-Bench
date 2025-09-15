from __future__ import annotations
import re
from copy import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple, Optional, ClassVar
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import Locator, Formatter, AutoLocator, AutoMinorLocator, FixedLocator, LinearLocator, LogLocator, SymmetricalLogLocator, MaxNLocator, MultipleLocator, EngFormatter, FuncFormatter, LogFormatterSciNotation, ScalarFormatter, StrMethodFormatter
from matplotlib.dates import AutoDateLocator, AutoDateFormatter, ConciseDateFormatter
from matplotlib.axis import Axis
from matplotlib.scale import ScaleBase
from pandas import Series
from seaborn._core.rules import categorical_order
from seaborn._core.typing import Default, default
from typing import TYPE_CHECKING
from seaborn._core.properties import Property
from numpy.typing import ArrayLike, NDArray

class Nominal(Scale):

    def _setup(self, data: Series, prop: Property, axis: Axis | None=None) -> Scale:
        new = copy(self)
        if new._tick_params is None:
            new = new.tick()
        if new._label_params is None:
            new = new.label()
        stringify = np.vectorize(format, otypes=['object'])
        units_seed = categorical_order(data, new.order)

        class CatScale(mpl.scale.LinearScale):
            name = None

            def set_default_locators_and_formatters(self, axis):
                ...
        mpl_scale = CatScale(data.name)
        if axis is None:
            axis = PseudoAxis(mpl_scale)
            axis.set_view_interval(0, len(units_seed) - 1)
        new._matplotlib_scale = mpl_scale
        axis.update_units(stringify(np.array(units_seed)))

        def convert_units(x):
            keep = np.array([x_ in units_seed for x_ in x], bool)
            out = np.full(len(x), np.nan)
            out[keep] = axis.convert_units(stringify(x[keep]))
            return out
        new._pipeline = [convert_units, prop.get_mapping(new, data)]

        def spacer(x):
            return 1
        new._spacer = spacer
        if prop.legend:
            new._legend = (units_seed, list(stringify(units_seed)))
        return new