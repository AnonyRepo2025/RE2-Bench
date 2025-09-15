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



def _dataarray_concat(
    arrays,
    dim,
    data_vars,
    coords,
    compat,
    positions,
    fill_value=dtypes.NA,
    join="outer",
):
    arrays = list(arrays)

    if data_vars != "all":
        raise ValueError(
            "data_vars is not a valid argument when concatenating DataArray objects"
        )

    datasets = []
    for n, arr in enumerate(arrays):
        if n == 0:
            name = arr.name
        elif name != arr.name:
            if compat == "identical":
                raise ValueError("array names not identical")
            else:
                arr = arr.rename(name)
        datasets.append(arr._to_temp_dataset())

    ds = _dataset_concat(
        datasets,
        dim,
        data_vars,
        coords,
        compat,
        positions,
        fill_value=fill_value,
        join=join,
    )
    return arrays[0]._from_temp_dataset(ds, name)
