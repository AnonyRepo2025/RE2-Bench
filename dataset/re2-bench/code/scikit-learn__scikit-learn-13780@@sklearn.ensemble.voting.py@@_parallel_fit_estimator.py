import numpy as np
from abc import abstractmethod
from ..base import ClassifierMixin
from ..base import RegressorMixin
from ..base import TransformerMixin
from ..base import clone
from ..preprocessing import LabelEncoder
from ..utils._joblib import Parallel, delayed
from ..utils.validation import has_fit_parameter, check_is_fitted
from ..utils.metaestimators import _BaseComposition
from ..utils import Bunch



def _parallel_fit_estimator(estimator, X, y, sample_weight=None):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        try:
            estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise ValueError(
                    "Underlying estimator {} does not support sample weights."
                    .format(estimator.__class__.__name__)
                ) from exc
            raise
    else:
        estimator.fit(X, y)
    return estimator
