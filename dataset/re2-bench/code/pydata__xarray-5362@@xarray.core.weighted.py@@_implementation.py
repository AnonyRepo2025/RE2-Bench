from typing import TYPE_CHECKING, Generic, Hashable, Iterable, Optional, TypeVar, Union
from . import duck_array_ops
from .computation import dot
from .pycompat import is_duck_dask_array
from .common import DataWithCoords  # noqa: F401
from .dataarray import DataArray, Dataset
from .dataarray import DataArray

T_DataWithCoords = TypeVar("T_DataWithCoords", bound="DataWithCoords")
_WEIGHTED_REDUCE_DOCSTRING_TEMPLATE = """
    Reduce this {cls}'s data by a weighted ``{fcn}`` along some dimension(s).

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to apply the weighted ``{fcn}``.
    skipna : bool, optional
        If True, skip missing values (as marked by NaN). By default, only
        skips missing values for float dtypes; other dtypes either do not
        have a sentinel missing value (int) or skipna=True has not been
        implemented (object, datetime64 or timedelta64).
    keep_attrs : bool, optional
        If True, the attributes (``attrs``) will be copied from the original
        object to the new one.  If False (default), the new object will be
        returned without attributes.

    Returns
    -------
    reduced : {cls}
        New {cls} object with weighted ``{fcn}`` applied to its data and
        the indicated dimension(s) removed.

    Notes
    -----
        Returns {on_zero} if the ``weights`` sum to 0.0 along the reduced
        dimension(s).
    """
_SUM_OF_WEIGHTS_DOCSTRING = """
    Calculate the sum of weights, accounting for missing values in the data

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to sum the weights.
    keep_attrs : bool, optional
        If True, the attributes (``attrs``) will be copied from the original
        object to the new one.  If False (default), the new object will be
        returned without attributes.

    Returns
    -------
    reduced : {cls}
        New {cls} object with the sum of the weights over the given dimension.
    """

class DataArrayWeighted(Weighted['DataArray']
):
    def _implementation(self, func, dim, **kwargs) -> "DataArray":

        self._check_dim(dim)

        dataset = self.obj._to_temp_dataset()
        dataset = dataset.map(func, dim=dim, **kwargs)
        return self.obj._from_temp_dataset(dataset)