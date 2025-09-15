from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn
import threading
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack
from ..base import is_classifier
from ..base import ClassifierMixin, MultiOutputMixin, RegressorMixin, TransformerMixin
from ..metrics import accuracy_score, r2_score
from ..preprocessing import OneHotEncoder
from ..tree import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from ..tree._tree import DTYPE, DOUBLE
from ..utils import check_random_state, compute_sample_weight
from ..exceptions import DataConversionWarning
from ._base import BaseEnsemble, _partition_estimators
from ..utils.parallel import delayed, Parallel
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.validation import (
    check_is_fitted,
    _check_sample_weight,
    _check_feature_names_in,
)
from ..utils.validation import _num_samples
from ..utils._param_validation import Interval, StrOptions

__all__ = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "RandomTreesEmbedding",
]
MAX_INT = np.iinfo(np.int32).max

def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0.0, 1.0]`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.

    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, Integral):
        if max_samples > n_samples:
            msg = "`max_samples` must be <= n_samples={} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, Real):
        return max(round(n_samples * max_samples), 1)
