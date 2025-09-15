from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from .base import LinearClassifierMixin, LinearModel, _rescale_data
from .sag import sag_solver
from ..base import RegressorMixin, MultiOutputMixin
from ..utils.extmath import safe_sparse_dot
from ..utils.extmath import row_norms
from ..utils import check_X_y
from ..utils import check_array
from ..utils import check_consistent_length
from ..utils import compute_sample_weight
from ..utils import column_or_1d
from ..preprocessing import LabelBinarizer
from ..model_selection import GridSearchCV
from ..metrics.scorer import check_scoring
from ..exceptions import ConvergenceWarning



class _BaseRidge(LinearModel, MultiOutputMixin, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=None, tol=1e-3, solver="auto",
                 random_state=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):

        if self.solver in ('sag', 'saga'):
            _dtype = np.float64
        else:
            # all other solvers work at both float precision levels
            _dtype = [np.float64, np.float32]

        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=_dtype,
                         multi_output=True, y_numeric=True)

        if ((sample_weight is not None) and
                np.atleast_1d(sample_weight).ndim > 1):
            raise ValueError("Sample weights must be 1D array or scalar")

        # when X is sparse we only remove offset from y
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight, return_mean=True)

        # temporary fix for fitting the intercept with sparse data using 'sag'
        if (sparse.issparse(X) and self.fit_intercept and
           self.solver != 'sparse_cg'):
            self.coef_, self.n_iter_, self.intercept_ = _ridge_regression(
                X, y, alpha=self.alpha, sample_weight=sample_weight,
                max_iter=self.max_iter, tol=self.tol, solver=self.solver,
                random_state=self.random_state, return_n_iter=True,
                return_intercept=True)
            # add the offset which was subtracted by _preprocess_data
            self.intercept_ += y_offset
        else:
            if sparse.issparse(X) and self.solver == 'sparse_cg':
                # required to fit intercept with sparse_cg solver
                params = {'X_offset': X_offset, 'X_scale': X_scale}
            else:
                # for dense matrices or when intercept is set to 0
                params = {}

            self.coef_, self.n_iter_ = _ridge_regression(
                X, y, alpha=self.alpha, sample_weight=sample_weight,
                max_iter=self.max_iter, tol=self.tol, solver=self.solver,
                random_state=self.random_state, return_n_iter=True,
                return_intercept=False, **params)

            self._set_intercept(X_offset, y_offset, X_scale)

        return self
