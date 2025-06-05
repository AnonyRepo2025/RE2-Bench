from __future__ import division
from itertools import chain
import numpy as np
import warnings
from scipy import sparse
from ..base import clone, TransformerMixin
from ..utils._joblib import Parallel, delayed
from ..externals import six
from ..pipeline import _fit_transform_one, _transform_one, _name_estimators
from ..preprocessing import FunctionTransformer
from ..utils import Bunch
from ..utils.metaestimators import _BaseComposition
from ..utils.validation import check_array, check_is_fitted

__all__ = ['ColumnTransformer', 'make_column_transformer']
_ERR_MSG_1DCOLUMN = ("1D data passed to a transformer that expects 2D data. "
                     "Try to specify the column selection as a list of one "
                     "item instead of a scalar.")

def _validate_transformers(transformers):
    """Checks if given transformers are valid.

    This is a helper function to support the deprecated tuple order.
    XXX Remove in v0.22
    """
    if not transformers:
        return True

    for t in transformers:
        if isinstance(t, six.string_types) and t in ('drop', 'passthrough'):
            continue
        if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                hasattr(t, "transform")):
            return False

    return True
