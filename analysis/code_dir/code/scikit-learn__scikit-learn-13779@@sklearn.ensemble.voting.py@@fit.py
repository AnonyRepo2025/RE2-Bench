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

class _BaseVoting(_BaseComposition, TransformerMixin):
    _required_parameters = ['estimators']

    @property
    def named_estimators(self):
        return Bunch(**dict(self.estimators))

    @property
    def _weights_not_none(self):
        if self.weights is None:
            return None
        return [w for est, w in zip(self.estimators, self.weights) if est[1] is not None]

    def _predict(self, X):
        return np.asarray([clf.predict(X) for clf in self.estimators_]).T

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators` should be a list of (string, estimator) tuples')
        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError('Number of `estimators` and weights must be equal; got %d weights, %d estimators' % (len(self.weights), len(self.estimators)))
        if sample_weight is not None:
            for name, step in self.estimators:
                if step is None:
                    continue
                if not has_fit_parameter(step, 'sample_weight'):
                    raise ValueError("Underlying estimator '%s' does not support sample weights." % name)
        names, clfs = zip(*self.estimators)
        self._validate_names(names)
        n_isnone = np.sum([clf is None for _, clf in self.estimators])
        if n_isnone == len(self.estimators):
            raise ValueError('All estimators are None. At least one is required!')
        self.estimators_ = Parallel(n_jobs=self.n_jobs)((delayed(_parallel_fit_estimator)(clone(clf), X, y, sample_weight=sample_weight) for clf in clfs if clf is not None))
        self.named_estimators_ = Bunch()
        for k, e in zip(self.estimators, self.estimators_):
            self.named_estimators_[k[0]] = e
        return self

    def set_params(self, **params):
        return self._set_params('estimators', **params)

    def get_params(self, deep=True):
        return self._get_params('estimators', deep=deep)