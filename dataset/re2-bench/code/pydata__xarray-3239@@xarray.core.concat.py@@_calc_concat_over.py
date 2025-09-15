from collections import OrderedDict
import pandas as pd
from . import dtypes, utils
from .alignment import align
from .merge import unique_variable, _VALID_COMPAT
from .variable import IndexVariable, Variable, as_variable
from .variable import concat as concat_vars
from .dataset import Dataset
from .dataarray import DataArray
from .dataarray import DataArray
from .dataset import Dataset



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
                else:
                    raise ValueError("%r is not present in all datasets" % dim)
        concat_over.update(k for k, v in ds.variables.items() if dim in v.dims)
        concat_dim_lengths.append(ds.dims.get(dim, 1))

    def process_subset_opt(opt, subset):
        if isinstance(opt, str):
            if opt == "different":
                if compat == "override":
                    raise ValueError(
                        "Cannot specify both %s='different' and compat='override'."
                        % subset
                    )
                # all nonindexes that are not the same in each dataset
                for k in getattr(datasets[0], subset):
                    if k not in concat_over:
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
                    set(getattr(datasets[0], subset)) - set(datasets[0].dims)
                )
            elif opt == "minimal":
                pass
            else:
                raise ValueError("unexpected value for %s: %s" % (subset, opt))
        else:
            invalid_vars = [k for k in opt if k not in getattr(datasets[0], subset)]
            if invalid_vars:
                if subset == "coords":
                    raise ValueError(
                        "some variables in coords are not coordinates on "
                        "the first dataset: %s" % (invalid_vars,)
                    )
                else:
                    raise ValueError(
                        "some variables in data_vars are not data variables "
                        "on the first dataset: %s" % (invalid_vars,)
                    )
            concat_over.update(opt)

    process_subset_opt(data_vars, "data_vars")
    process_subset_opt(coords, "coords")
    return concat_over, equals, concat_dim_lengths
