import itertools
import warnings
from collections import Counter, OrderedDict
from textwrap import dedent
import pandas as pd
from . import dtypes
from .concat import concat
from .dataarray import DataArray
from .dataset import Dataset
from .merge import merge

_CONCAT_DIM_DEFAULT = "__infer_concat_dim__"

def _auto_concat(
    datasets,
    dim=None,
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
    compat="no_conflicts",
):
    if len(datasets) == 1 and dim is None:
        # There is nothing more to combine, so kick out early.
        return datasets[0]
    else:
        if dim is None:
            ds0 = datasets[0]
            ds1 = datasets[1]
            concat_dims = set(ds0.dims)
            if ds0.dims != ds1.dims:
                dim_tuples = set(ds0.dims.items()) - set(ds1.dims.items())
                concat_dims = {i for i, _ in dim_tuples}
            if len(concat_dims) > 1:
                concat_dims = {d for d in concat_dims if not ds0[d].equals(ds1[d])}
            if len(concat_dims) > 1:
                raise ValueError(
                    "too many different dimensions to " "concatenate: %s" % concat_dims
                )
            elif len(concat_dims) == 0:
                raise ValueError(
                    "cannot infer dimension to concatenate: "
                    "supply the ``concat_dim`` argument "
                    "explicitly"
                )
            dim, = concat_dims
        return concat(
            datasets,
            dim=dim,
            data_vars=data_vars,
            coords=coords,
            fill_value=fill_value,
            compat=compat,
        )
