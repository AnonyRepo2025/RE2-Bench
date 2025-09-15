import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from .coding import strings, times, variables
from .coding.variables import SerializationWarning, pop_to
from .core import duck_array_ops, indexing
from .core.common import _contains_datetime_like_objects, contains_cftime_datetimes
from .core.pycompat import is_duck_dask_array
from .core.variable import IndexVariable, Variable, as_variable
from .backends.common import AbstractDataStore
from .core.dataset import Dataset

CF_RELATED_DATA = (
    "bounds",
    "grid_mapping",
    "climatology",
    "geometry",
    "node_coordinates",
    "node_count",
    "part_node_count",
    "interior_ring",
    "cell_measures",
    "formula_terms",
)
CF_RELATED_DATA_NEEDS_PARSING = (
    "cell_measures",
    "formula_terms",
)

def decode_cf_variables(
    variables,
    attributes,
    concat_characters=True,
    mask_and_scale=True,
    decode_times=True,
    decode_coords=True,
    drop_variables=None,
    use_cftime=None,
    decode_timedelta=None,
):
    """
    Decode several CF encoded variables.

    See: decode_cf_variable
    """
    dimensions_used_by = defaultdict(list)
    for v in variables.values():
        for d in v.dims:
            dimensions_used_by[d].append(v)

    def stackable(dim):
        # figure out if a dimension can be concatenated over
        if dim in variables:
            return False
        for v in dimensions_used_by[dim]:
            if v.dtype.kind != "S" or dim != v.dims[-1]:
                return False
        return True

    coord_names = set()

    if isinstance(drop_variables, str):
        drop_variables = [drop_variables]
    elif drop_variables is None:
        drop_variables = []
    drop_variables = set(drop_variables)

    # Time bounds coordinates might miss the decoding attributes
    if decode_times:
        _update_bounds_attributes(variables)

    new_vars = {}
    for k, v in variables.items():
        if k in drop_variables:
            continue
        stack_char_dim = (
            concat_characters
            and v.dtype == "S1"
            and v.ndim > 0
            and stackable(v.dims[-1])
        )
        try:
            new_vars[k] = decode_cf_variable(
                k,
                v,
                concat_characters=concat_characters,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                stack_char_dim=stack_char_dim,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        except Exception as e:
            raise type(e)(f"Failed to decode variable {k!r}: {e}")
        if decode_coords in [True, "coordinates", "all"]:
            var_attrs = new_vars[k].attrs
            if "coordinates" in var_attrs:
                coord_str = var_attrs["coordinates"]
                var_coord_names = coord_str.split()
                if all(k in variables for k in var_coord_names):
                    new_vars[k].encoding["coordinates"] = coord_str
                    del var_attrs["coordinates"]
                    coord_names.update(var_coord_names)

        if decode_coords == "all":
            for attr_name in CF_RELATED_DATA:
                if attr_name in var_attrs:
                    attr_val = var_attrs[attr_name]
                    if attr_name not in CF_RELATED_DATA_NEEDS_PARSING:
                        var_names = attr_val.split()
                    else:
                        roles_and_names = [
                            role_or_name
                            for part in attr_val.split(":")
                            for role_or_name in part.split()
                        ]
                        if len(roles_and_names) % 2 == 1:
                            warnings.warn(
                                f"Attribute {attr_name:s} malformed", stacklevel=5
                            )
                        var_names = roles_and_names[1::2]
                    if all(var_name in variables for var_name in var_names):
                        new_vars[k].encoding[attr_name] = attr_val
                        coord_names.update(var_names)
                    else:
                        referenced_vars_not_in_variables = [
                            proj_name
                            for proj_name in var_names
                            if proj_name not in variables
                        ]
                        warnings.warn(
                            f"Variable(s) referenced in {attr_name:s} not in variables: {referenced_vars_not_in_variables!s}",
                            stacklevel=5,
                        )
                    del var_attrs[attr_name]

    if decode_coords and "coordinates" in attributes:
        attributes = dict(attributes)
        coord_names.update(attributes.pop("coordinates").split())

    return new_vars, attributes, coord_names
