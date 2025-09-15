import warnings
from functools import partial
from typing import Any, Hashable
import numpy as np
import pandas as pd
from ..core import dtypes, duck_array_ops, indexing
from ..core.pycompat import is_duck_dask_array
from ..core.variable import Variable
import dask.array as da



class UnsignedIntegerCoder(VariableCoder):
    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)

        if "_Unsigned" in attrs:
            unsigned = pop_to(attrs, encoding, "_Unsigned")

            if data.dtype.kind == "i":
                if unsigned == "true":
                    unsigned_dtype = np.dtype("u%s" % data.dtype.itemsize)
                    transform = partial(np.asarray, dtype=unsigned_dtype)
                    data = lazy_elemwise_func(data, transform, unsigned_dtype)
                    if "_FillValue" in attrs:
                        new_fill = unsigned_dtype.type(attrs["_FillValue"])
                        attrs["_FillValue"] = new_fill
            elif data.dtype.kind == "u":
                if unsigned == "false":
                    signed_dtype = np.dtype("i%s" % data.dtype.itemsize)
                    transform = partial(np.asarray, dtype=signed_dtype)
                    data = lazy_elemwise_func(data, transform, signed_dtype)
                    if "_FillValue" in attrs:
                        new_fill = signed_dtype.type(attrs["_FillValue"])
                        attrs["_FillValue"] = new_fill
            else:
                warnings.warn(
                    "variable %r has _Unsigned attribute but is not "
                    "of integer type. Ignoring attribute." % name,
                    SerializationWarning,
                    stacklevel=3,
                )

        return Variable(dims, data, attrs, encoding)