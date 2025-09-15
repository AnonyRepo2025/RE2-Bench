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



def _dataset_concat(
    datasets: list[T_Dataset],
    dim: str | T_DataArray | pd.Index,
    data_vars: str | list[str],
    coords: str | list[str],
    compat: CompatOptions,
    positions: Iterable[Iterable[int]] | None,
    fill_value: Any = dtypes.NA,
    join: JoinOptions = "outer",
    combine_attrs: CombineAttrsOptions = "override",
) -> T_Dataset:
    """
    Concatenate a sequence of datasets along a new or existing dimension
    """
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset

    datasets = list(datasets)

    if not all(isinstance(dataset, Dataset) for dataset in datasets):
        raise TypeError(
            "The elements in the input list need to be either all 'Dataset's or all 'DataArray's"
        )

    if isinstance(dim, DataArray):
        dim_var = dim.variable
    elif isinstance(dim, Variable):
        dim_var = dim
    else:
        dim_var = None

    dim, index = _calc_concat_dim_index(dim)

    # Make sure we're working on a copy (we'll be loading variables)
    datasets = [ds.copy() for ds in datasets]
    datasets = list(
        align(*datasets, join=join, copy=False, exclude=[dim], fill_value=fill_value)
    )

    dim_coords, dims_sizes, coord_names, data_names, vars_order = _parse_datasets(
        datasets
    )
    dim_names = set(dim_coords)
    unlabeled_dims = dim_names - coord_names

    both_data_and_coords = coord_names & data_names
    if both_data_and_coords:
        raise ValueError(
            f"{both_data_and_coords!r} is a coordinate in some datasets but not others."
        )
    # we don't want the concat dimension in the result dataset yet
    dim_coords.pop(dim, None)
    dims_sizes.pop(dim, None)

    # case where concat dimension is a coordinate or data_var but not a dimension
    if (dim in coord_names or dim in data_names) and dim not in dim_names:
        # TODO: Overriding type because .expand_dims has incorrect typing:
        datasets = [cast(T_Dataset, ds.expand_dims(dim)) for ds in datasets]

    # determine which variables to concatenate
    concat_over, equals, concat_dim_lengths = _calc_concat_over(
        datasets, dim, dim_names, data_vars, coords, compat
    )

    # determine which variables to merge, and then merge them according to compat
    variables_to_merge = (coord_names | data_names) - concat_over - unlabeled_dims

    result_vars = {}
    result_indexes = {}

    if variables_to_merge:
        grouped = {
            k: v
            for k, v in collect_variables_and_indexes(list(datasets)).items()
            if k in variables_to_merge
        }
        merged_vars, merged_indexes = merge_collected(
            grouped, compat=compat, equals=equals
        )
        result_vars.update(merged_vars)
        result_indexes.update(merged_indexes)

    result_vars.update(dim_coords)

    # assign attrs and encoding from first dataset
    result_attrs = merge_attrs([ds.attrs for ds in datasets], combine_attrs)
    result_encoding = datasets[0].encoding

    # check that global attributes are fixed across all datasets if necessary
    for ds in datasets[1:]:
        if compat == "identical" and not utils.dict_equiv(ds.attrs, result_attrs):
            raise ValueError("Dataset global attributes not equal.")

    # we've already verified everything is consistent; now, calculate
    # shared dimension sizes so we can expand the necessary variables
    def ensure_common_dims(vars, concat_dim_lengths):
        # ensure each variable with the given name shares the same
        # dimensions and the same shape for all of them except along the
        # concat dimension
        common_dims = tuple(pd.unique([d for v in vars for d in v.dims]))
        if dim not in common_dims:
            common_dims = (dim,) + common_dims
        for var, dim_len in zip(vars, concat_dim_lengths):
            if var.dims != common_dims:
                common_shape = tuple(dims_sizes.get(d, dim_len) for d in common_dims)
                var = var.set_dims(common_dims, common_shape)
            yield var

    # get the indexes to concatenate together, create a PandasIndex
    # for any scalar coordinate variable found with ``name`` matching ``dim``.
    # TODO: depreciate concat a mix of scalar and dimensional indexed coordinates?
    # TODO: (benbovy - explicit indexes): check index types and/or coordinates
    # of all datasets?
    def get_indexes(name):
        for ds in datasets:
            if name in ds._indexes:
                yield ds._indexes[name]
            elif name == dim:
                var = ds._variables[name]
                if not var.dims:
                    data = var.set_dims(dim).values
                    yield PandasIndex(data, dim, coord_dtype=var.dtype)

    # create concatenation index, needed for later reindexing
    concat_index = list(range(sum(concat_dim_lengths)))

    # stack up each variable and/or index to fill-out the dataset (in order)
    # n.b. this loop preserves variable order, needed for groupby.
    for name in vars_order:
        if name in concat_over and name not in result_indexes:
            variables = []
            variable_index = []
            var_concat_dim_length = []
            for i, ds in enumerate(datasets):
                if name in ds.variables:
                    variables.append(ds[name].variable)
                    # add to variable index, needed for reindexing
                    var_idx = [
                        sum(concat_dim_lengths[:i]) + k
                        for k in range(concat_dim_lengths[i])
                    ]
                    variable_index.extend(var_idx)
                    var_concat_dim_length.append(len(var_idx))
                else:
                    # raise if coordinate not in all datasets
                    if name in coord_names:
                        raise ValueError(
                            f"coordinate {name!r} not present in all datasets."
                        )
            vars = ensure_common_dims(variables, var_concat_dim_length)

            # Try to concatenate the indexes, concatenate the variables when no index
            # is found on all datasets.
            indexes: list[Index] = list(get_indexes(name))
            if indexes:
                if len(indexes) < len(datasets):
                    raise ValueError(
                        f"{name!r} must have either an index or no index in all datasets, "
                        f"found {len(indexes)}/{len(datasets)} datasets with an index."
                    )
                combined_idx = indexes[0].concat(indexes, dim, positions)
                if name in datasets[0]._indexes:
                    idx_vars = datasets[0].xindexes.get_all_coords(name)
                else:
                    # index created from a scalar coordinate
                    idx_vars = {name: datasets[0][name].variable}
                result_indexes.update({k: combined_idx for k in idx_vars})
                combined_idx_vars = combined_idx.create_variables(idx_vars)
                for k, v in combined_idx_vars.items():
                    v.attrs = merge_attrs(
                        [ds.variables[k].attrs for ds in datasets],
                        combine_attrs=combine_attrs,
                    )
                    result_vars[k] = v
            else:
                combined_var = concat_vars(
                    vars, dim, positions, combine_attrs=combine_attrs
                )
                # reindex if variable is not present in all datasets
                if len(variable_index) < len(concat_index):
                    combined_var = reindex_variables(
                        variables={name: combined_var},
                        dim_pos_indexers={
                            dim: pd.Index(variable_index).get_indexer(concat_index)
                        },
                        fill_value=fill_value,
                    )[name]
                result_vars[name] = combined_var

        elif name in result_vars:
            # preserves original variable order
            result_vars[name] = result_vars.pop(name)

    result = type(datasets[0])(result_vars, attrs=result_attrs)

    absent_coord_names = coord_names - set(result.variables)
    if absent_coord_names:
        raise ValueError(
            f"Variables {absent_coord_names!r} are coordinates in some datasets but not others."
        )
    result = result.set_coords(coord_names)
    result.encoding = result_encoding

    result = result.drop_vars(unlabeled_dims, errors="ignore")

    if index is not None:
        # add concat index / coordinate last to ensure that its in the final Dataset
        if dim_var is not None:
            index_vars = index.create_variables({dim: dim_var})
        else:
            index_vars = index.create_variables()
        result[dim] = index_vars[dim]
        result_indexes[dim] = index

    # TODO: add indexes at Dataset creation (when it is supported)
    result = result._overwrite_indexes(result_indexes)

    return result
