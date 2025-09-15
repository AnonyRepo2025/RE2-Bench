from itertools import chain
import numpy as np
from scipy import sparse
from ..base import clone, TransformerMixin
from ..externals.joblib import Parallel, delayed
from ..externals import six
from ..pipeline import (
    _fit_one_transformer, _fit_transform_one, _transform_one, _name_estimators)
from ..preprocessing import FunctionTransformer
from ..utils import Bunch
from ..utils.metaestimators import _BaseComposition
from ..utils.validation import check_is_fitted

__all__ = ['ColumnTransformer', 'make_column_transformer']
_ERR_MSG_1DCOLUMN = ("1D data passed to a transformer that expects 2D data. "
                     "Try to specify the column selection as a list of one "
                     "item instead of a scalar.")

def _hstack(X):
    """
    Stacks X horizontally.

    Supports input types (X): list of
        numpy arrays, sparse arrays and DataFrames
    """
    if any(sparse.issparse(f) for f in X):
        return sparse.hstack(X).tocsr()
    else:
        return np.hstack(X)
