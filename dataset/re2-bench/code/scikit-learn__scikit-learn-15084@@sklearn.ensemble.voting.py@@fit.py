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
    """Base class for voting.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators"""
        if self.weights is None:
            return None
        return [w for est, w in zip(self.estimators, self.weights)
                if est[1] not in (None, 'drop')]

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([est.predict(X) for est in self.estimators_]).T

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """
        common fit operations.
        """
        names, clfs = self._validate_estimators()

        if (self.weights is not None and
                len(self.weights) != len(self.estimators)):
            raise ValueError('Number of `estimators` and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_fit_estimator)(clone(clf), X, y,
                                                 sample_weight=sample_weight)
                for clf in clfs if clf not in (None, 'drop')
            )

        self.named_estimators_ = Bunch()
        for k, e in zip(self.estimators, self.estimators_):
            self.named_estimators_[k[0]] = e
        return self
