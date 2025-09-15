import datetime
import functools
import warnings
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
import numpy as np
import pandas as pd
from ..plot.plot import _PlotMethods
from . import (
    computation,
    dtypes,
    groupby,
    indexing,
    ops,
    pdcompat,
    resample,
    rolling,
    utils,
    weighted,
)
from .accessor_dt import CombinedDatetimelikeAccessor
from .accessor_str import StringAccessor
from .alignment import (
    _broadcast_helper,
    _get_broadcast_dims_map_common_coords,
    align,
    reindex_like_indexers,
)
from .common import AbstractArray, DataWithCoords
from .coordinates import (
    DataArrayCoordinates,
    LevelCoordinatesSource,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .dataset import Dataset, split_indexes
from .formatting import format_item
from .indexes import Indexes, default_indexes, propagate_indexes
from .indexing import is_fancy_indexer
from .merge import PANDAS_TYPES, _extract_indexes_from_coords
from .options import OPTIONS
from .utils import Default, ReprObject, _check_inplace, _default, either_dict_or_kwargs
from .variable import (
    IndexVariable,
    Variable,
    as_compatible_data,
    as_variable,
    assert_unique_multiindex_level_names,
)
from dask.delayed import Delayed
from cdms2 import Variable as cdms2_Variable
from iris.cube import Cube as iris_Cube
from .dataset import _get_virtual_variable
from dask.base import normalize_token
from .missing import interp_na
from .missing import ffill
from .missing import bfill
from ..backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE
from ..convert import to_cdms2
from ..convert import from_cdms2
from ..convert import to_iris
from ..convert import from_iris
from .parallel import map_blocks

_THIS_ARRAY = ReprObject("<this-array>")

class DataArray(AbstractArray, DataWithCoords):
    __slots__ = (
        "_cache",
        "_coords",
        "_file_obj",
        "_indexes",
        "_name",
        "_variable",
        "__weakref__",
    )
    _groupby_cls = groupby.DataArrayGroupBy
    _rolling_cls = rolling.DataArrayRolling
    _coarsen_cls = rolling.DataArrayCoarsen
    _resample_cls = resample.DataArrayResample
    _weighted_cls = weighted.DataArrayWeighted
    dt = property(CombinedDatetimelikeAccessor)
    __hash__ = None  # type: ignore
    str = property(StringAccessor)
    def polyfit(
        self,
        dim: Hashable,
        deg: int,
        skipna: bool = None,
        rcond: float = None,
        w: Union[Hashable, Any] = None,
        full: bool = False,
        cov: bool = False,
    ):
        """
        Least squares polynomial fit.

        This replicates the behaviour of `numpy.polyfit` but differs by skipping
        invalid values when `skipna = True`.

        Parameters
        ----------
        dim : hashable
            Coordinate along which to fit the polynomials.
        deg : int
            Degree of the fitting polynomial.
        skipna : bool, optional
            If True, removes all invalid values before fitting each 1D slices of the array.
            Default is True if data is stored in a dask.array or if there is any
            invalid values, False otherwise.
        rcond : float, optional
            Relative condition number to the fit.
        w : Union[Hashable, Any], optional
            Weights to apply to the y-coordinate of the sample points.
            Can be an array-like object or the name of a coordinate in the dataset.
        full : bool, optional
            Whether to return the residuals, matrix rank and singular values in addition
            to the coefficients.
        cov : Union[bool, str], optional
            Whether to return to the covariance matrix in addition to the coefficients.
            The matrix is not scaled if `cov='unscaled'`.

        Returns
        -------
        polyfit_results : Dataset
            A single dataset which contains:

            polyfit_coefficients
                The coefficients of the best fit.
            polyfit_residuals
                The residuals of the least-square computation (only included if `full=True`)
            [dim]_matrix_rank
                The effective rank of the scaled Vandermonde coefficient matrix (only included if `full=True`)
            [dim]_singular_value
                The singular values of the scaled Vandermonde coefficient matrix (only included if `full=True`)
            polyfit_covariance
                The covariance matrix of the polynomial coefficient estimates (only included if `full=False` and `cov=True`)

        See also
        --------
        numpy.polyfit
        """
        return self._to_temp_dataset().polyfit(
            dim, deg, skipna=skipna, rcond=rcond, w=w, full=full, cov=cov
        )