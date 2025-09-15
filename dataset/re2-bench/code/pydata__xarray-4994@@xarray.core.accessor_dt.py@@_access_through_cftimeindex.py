import warnings
from distutils.version import LooseVersion
import numpy as np
import pandas as pd
from .common import (
    _contains_datetime_like_objects,
    is_np_datetime_like,
    is_np_timedelta_like,
)
from .npcompat import DTypeLike
from .pycompat import is_duck_dask_array
from ..coding.cftimeindex import CFTimeIndex
from ..coding.cftimeindex import CFTimeIndex
from ..coding.cftimeindex import CFTimeIndex
from dask.array import map_blocks
from dask.array import map_blocks
from dask.array import map_blocks
from .dataset import Dataset



def _access_through_cftimeindex(values, name):
    """Coerce an array of datetime-like values to a CFTimeIndex
    and access requested datetime component
    """
    from ..coding.cftimeindex import CFTimeIndex

    values_as_cftimeindex = CFTimeIndex(values.ravel())
    if name == "season":
        months = values_as_cftimeindex.month
        field_values = _season_from_months(months)
    elif name == "date":
        raise AttributeError(
            "'CFTimeIndex' object has no attribute `date`. Consider using the floor method instead, for instance: `.time.dt.floor('D')`."
        )
    else:
        field_values = getattr(values_as_cftimeindex, name)
    return field_values.reshape(values.shape)
