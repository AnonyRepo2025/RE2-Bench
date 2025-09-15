def column_or_1d(y, warn=False):
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn('A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().', DataConversionWarning, stacklevel=2)
        return np.ravel(y)
    raise ValueError('bad input shape {0}'.format(shape))

def __array__(self, dtype=None):
    return self.data



from abc import abstractmethod
import numpy as np
from joblib import Parallel, delayed
from ..base import ClassifierMixin
from ..base import RegressorMixin
from ..base import TransformerMixin
from ..base import clone
from .base import _parallel_fit_estimator
from .base import _BaseHeterogeneousEnsemble
from ..preprocessing import LabelEncoder
from ..utils import Bunch
from ..utils.validation import check_is_fitted
from ..utils.multiclass import check_classification_targets
from ..utils.validation import column_or_1d

class _BaseVoting(TransformerMixin, _BaseHeterogeneousEnsemble):

    @property
    def _weights_not_none(self):
        if self.weights is None:
            return None
        return [w for est, w in zip(self.estimators, self.weights) if est[1] not in (None, 'drop')]

    def _predict(self, X):
        return np.asarray([est.predict(X) for est in self.estimators_]).T

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        names, clfs = self._validate_estimators()
        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError('Number of `estimators` and weights must be equal; got %d weights, %d estimators' % (len(self.weights), len(self.estimators)))
        self.estimators_ = Parallel(n_jobs=self.n_jobs)((delayed(_parallel_fit_estimator)(clone(clf), X, y, sample_weight=sample_weight) for clf in clfs if clf not in (None, 'drop')))
        self.named_estimators_ = Bunch()
        for k, e in zip(self.estimators, self.estimators_):
            self.named_estimators_[k[0]] = e
        return self