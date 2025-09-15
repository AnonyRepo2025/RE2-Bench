from __future__ import annotations
from typing import TYPE_CHECKING, Any, Hashable, Iterable, cast, overload
import pandas as pd
from xarray.core import dtypes, utils
from xarray.core.alignment import align, reindex_variables
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.core.indexes import Index, PandasIndex
from xarray.core.merge import (
    _VALID_COMPAT,
    collect_variables_and_indexes,
    merge_attrs,
    merge_collected,
)
from xarray.core.types import T_DataArray, T_Dataset
from xarray.core.variable import Variable
from xarray.core.variable import concat as concat_vars
from xarray.core.types import (
        CombineAttrsOptions,
        CompatOptions,
        ConcatOptions,
        JoinOptions,
    )
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray



def _parse_datasets(
    datasets: list[T_Dataset],
) -> tuple[
    dict[Hashable, Variable],
    dict[Hashable, int],
    set[Hashable],
    set[Hashable],
    list[Hashable],
]:
    dims: set[Hashable] = set()
    all_coord_names: set[Hashable] = set()
    data_vars: set[Hashable] = set()  # list of data_vars
    dim_coords: dict[Hashable, Variable] = {}  # maps dim name to variable
    dims_sizes: dict[Hashable, int] = {}  # shared dimension sizes to expand variables
    variables_order: dict[Hashable, Variable] = {}  # variables in order of appearance

    for ds in datasets:
        dims_sizes.update(ds.dims)
        all_coord_names.update(ds.coords)
        data_vars.update(ds.data_vars)
        variables_order.update(ds.variables)

        # preserves ordering of dimensions
        for dim in ds.dims:
            if dim in dims:
                continue

            if dim not in dim_coords:
                dim_coords[dim] = ds.coords[dim].variable
        dims = dims | set(ds.dims)

    return dim_coords, dims_sizes, all_coord_names, data_vars, list(variables_order)
