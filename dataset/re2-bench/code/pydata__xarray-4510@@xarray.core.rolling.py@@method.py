import functools
import warnings
from typing import Any, Callable, Dict
import numpy as np
from . import dtypes, duck_array_ops, utils
from .dask_array_ops import dask_rolling_wrapper
from .ops import inject_reduce_methods
from .options import _get_keep_attrs
from .pycompat import is_duck_dask_array
import bottleneck
from .dataarray import DataArray
from .dataarray import DataArray
from .dataset import Dataset
from .dataset import Dataset
from .dataarray import DataArray
from .dataset import Dataset

_ROLLING_REDUCE_DOCSTRING_TEMPLATE = """\
Reduce this object's data windows by applying `{name}` along its dimension.

Parameters
----------
keep_attrs : bool, default: None
    If True, the attributes (``attrs``) will be copied from the original
    object to the new one. If False, the new object will be returned
    without attributes. If None uses the global default.
**kwargs : dict
    Additional keyword arguments passed on to `{name}`.

Returns
-------
reduced : same type as caller
    New object with `{name}` applied along its rolling dimnension.
"""

class Rolling:
    """A object that implements the moving window pattern.

    See Also
    --------
    Dataset.groupby
    DataArray.groupby
    Dataset.rolling
    DataArray.rolling
    """

    __slots__ = ("obj", "window", "min_periods", "center", "dim", "keep_attrs")
    _attributes = ("window", "min_periods", "center", "dim", "keep_attrs")

    def __init__(self, obj, windows, min_periods=None, center=False, keep_attrs=None):
        """
        Moving window object.

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            window along (e.g. `time`) to the size of the moving window.
        min_periods : int, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool, default: False
            Set the labels at the center of the window.

        Returns
        -------
        rolling : type of input argument
        """
        self.dim, self.window = [], []
        for d, w in windows.items():
            self.dim.append(d)
            if w <= 0:
                raise ValueError("window must be > 0")
            self.window.append(w)

        self.center = self._mapping_to_list(center, default=False)
        self.obj = obj

        # attributes
        if min_periods is not None and min_periods <= 0:
            raise ValueError("min_periods must be greater than zero or None")

        self.min_periods = np.prod(self.window) if min_periods is None else min_periods

        if keep_attrs is not None:
            warnings.warn(
                "Passing ``keep_attrs`` to ``rolling`` is deprecated and will raise an"
                " error in xarray 0.18. Please pass ``keep_attrs`` directly to the"
                " applied function. Note that keep_attrs is now True per default.",
                FutureWarning,
            )
        self.keep_attrs = keep_attrs

    def __repr__(self):
        """provide a nice str repr of our rolling object"""

        attrs = [
            "{k}->{v}{c}".format(k=k, v=w, c="(center)" if c else "")
            for k, w, c in zip(self.dim, self.window, self.center)
        ]
        return "{klass} [{attrs}]".format(
            klass=self.__class__.__name__, attrs=",".join(attrs)
        )

    def __len__(self):
        return self.obj.sizes[self.dim]

    def _reduce_method(name: str) -> Callable:  # type: ignore
        array_agg_func = getattr(duck_array_ops, name)
        bottleneck_move_func = getattr(bottleneck, "move_" + name, None)

        def method(self, keep_attrs=None, **kwargs):

            keep_attrs = self._get_keep_attrs(keep_attrs)

            return self._numpy_or_bottleneck_reduce(
                array_agg_func, bottleneck_move_func, keep_attrs=keep_attrs, **kwargs
            )

        method.__name__ = name
        method.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name=name)
        return method

    argmax = _reduce_method("argmax")
    argmin = _reduce_method("argmin")
    max = _reduce_method("max")
    min = _reduce_method("min")
    mean = _reduce_method("mean")
    prod = _reduce_method("prod")
    sum = _reduce_method("sum")
    std = _reduce_method("std")
    var = _reduce_method("var")
    median = _reduce_method("median")

    def count(self, keep_attrs=None):
        keep_attrs = self._get_keep_attrs(keep_attrs)
        rolling_count = self._counts(keep_attrs=keep_attrs)
        enough_periods = rolling_count >= self.min_periods
        return rolling_count.where(enough_periods)

    count.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name="count")

    def _mapping_to_list(
        self, arg, default=None, allow_default=True, allow_allsame=True
    ):
        if utils.is_dict_like(arg):
            if allow_default:
                return [arg.get(d, default) for d in self.dim]
            else:
                for d in self.dim:
                    if d not in arg:
                        raise KeyError(f"argument has no key {d}.")
                return [arg[d] for d in self.dim]
        elif allow_allsame:  # for single argument
            return [arg] * len(self.dim)
        elif len(self.dim) == 1:
            return [arg]
        else:
            raise ValueError(
                "Mapping argument is necessary for {}d-rolling.".format(len(self.dim))
            )

    def _get_keep_attrs(self, keep_attrs):

        if keep_attrs is None:
            # TODO: uncomment the next line and remove the others after the deprecation
            # keep_attrs = _get_keep_attrs(default=True)

            if self.keep_attrs is None:
                keep_attrs = _get_keep_attrs(default=True)
            else:
                keep_attrs = self.keep_attrs

        return keep_attrs
