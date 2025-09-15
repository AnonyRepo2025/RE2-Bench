from typing import TYPE_CHECKING, Hashable, Iterable, Optional, Union, overload
from .computation import dot
from .options import _get_keep_attrs
from .dataarray import DataArray, Dataset
from .dataarray import DataArray

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

class Weighted:
    __slots__ = ("obj", "weights")
    @staticmethod
    def _reduce(
        da: "DataArray",
        weights: "DataArray",
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
    ) -> "DataArray":
        """reduce using dot; equivalent to (da * weights).sum(dim, skipna)

            for internal use only
        """

        # need to infer dims as we use `dot`
        if dim is None:
            dim = ...

        # need to mask invalid values in da, as `dot` does not implement skipna
        if skipna or (skipna is None and da.dtype.kind in "cfO"):
            da = da.fillna(0.0)

        # `dot` does not broadcast arrays, so this avoids creating a large
        # DataArray (if `weights` has additional dimensions)
        # maybe add fasttrack (`(da * weights).sum(dims=dim, skipna=skipna)`)
        return dot(da, weights, dims=dim)
    def _sum_of_weights(
        self, da: "DataArray", dim: Optional[Union[Hashable, Iterable[Hashable]]] = None
    ) -> "DataArray":
        """ Calculate the sum of weights, accounting for missing values """

        # we need to mask data values that are nan; else the weights are wrong
        mask = da.notnull()

        # bool -> int, because ``xr.dot([True, True], [True, True])`` -> True
        # (and not 2); GH4074
        if self.weights.dtype == bool:
            sum_of_weights = self._reduce(
                mask, self.weights.astype(int), dim=dim, skipna=False
            )
        else:
            sum_of_weights = self._reduce(mask, self.weights, dim=dim, skipna=False)

        # 0-weights are not valid
        valid_weights = sum_of_weights != 0.0

        return sum_of_weights.where(valid_weights)