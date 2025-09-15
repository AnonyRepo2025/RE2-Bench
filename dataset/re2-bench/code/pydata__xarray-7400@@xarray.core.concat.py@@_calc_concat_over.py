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



def _calc_concat_over(datasets, dim, dim_names, data_vars, coords, compat):
    """
    Determine which dataset variables need to be concatenated in the result,
    """
    # Return values
    concat_over = set()
    equals = {}

    if dim in dim_names:
        concat_over_existing_dim = True
        concat_over.add(dim)
    else:
        concat_over_existing_dim = False

    concat_dim_lengths = []
    for ds in datasets:
        if concat_over_existing_dim:
            if dim not in ds.dims:
                if dim in ds:
                    ds = ds.set_coords(dim)
        concat_over.update(k for k, v in ds.variables.items() if dim in v.dims)
        concat_dim_lengths.append(ds.dims.get(dim, 1))

    def process_subset_opt(opt, subset):
        if isinstance(opt, str):
            if opt == "different":
                if compat == "override":
                    raise ValueError(
                        f"Cannot specify both {subset}='different' and compat='override'."
                    )
                # all nonindexes that are not the same in each dataset
                for k in getattr(datasets[0], subset):
                    if k not in concat_over:
                        equals[k] = None

                        variables = [
                            ds.variables[k] for ds in datasets if k in ds.variables
                        ]

                        if len(variables) == 1:
                            # coords="different" doesn't make sense when only one object
                            # contains a particular variable.
                            break
                        elif len(variables) != len(datasets) and opt == "different":
                            raise ValueError(
                                f"{k!r} not present in all datasets and coords='different'. "
                                f"Either add {k!r} to datasets where it is missing or "
                                "specify coords='minimal'."
                            )

                        # first check without comparing values i.e. no computes
                        for var in variables[1:]:
                            equals[k] = getattr(variables[0], compat)(
                                var, equiv=lazy_array_equiv
                            )
                            if equals[k] is not True:
                                # exit early if we know these are not equal or that
                                # equality cannot be determined i.e. one or all of
                                # the variables wraps a numpy array
                                break

                        if equals[k] is False:
                            concat_over.add(k)

                        elif equals[k] is None:
                            # Compare the variable of all datasets vs. the one
                            # of the first dataset. Perform the minimum amount of
                            # loads in order to avoid multiple loads from disk
                            # while keeping the RAM footprint low.
                            v_lhs = datasets[0].variables[k].load()
                            # We'll need to know later on if variables are equal.
                            computed = []
                            for ds_rhs in datasets[1:]:
                                v_rhs = ds_rhs.variables[k].compute()
                                computed.append(v_rhs)
                                if not getattr(v_lhs, compat)(v_rhs):
                                    concat_over.add(k)
                                    equals[k] = False
                                    # computed variables are not to be re-computed
                                    # again in the future
                                    for ds, v in zip(datasets[1:], computed):
                                        ds.variables[k].data = v.data
                                    break
                            else:
                                equals[k] = True

            elif opt == "all":
                concat_over.update(
                    set().union(
                        *list(set(getattr(d, subset)) - set(d.dims) for d in datasets)
                    )
                )
            elif opt == "minimal":
                pass
            else:
                raise ValueError(f"unexpected value for {subset}: {opt}")
        else:
            invalid_vars = [k for k in opt if k not in getattr(datasets[0], subset)]
            if invalid_vars:
                if subset == "coords":
                    raise ValueError(
                        "some variables in coords are not coordinates on "
                        f"the first dataset: {invalid_vars}"
                    )
                else:
                    raise ValueError(
                        "some variables in data_vars are not data variables "
                        f"on the first dataset: {invalid_vars}"
                    )
            concat_over.update(opt)

    process_subset_opt(data_vars, "data_vars")
    process_subset_opt(coords, "coords")
    return concat_over, equals, concat_dim_lengths
