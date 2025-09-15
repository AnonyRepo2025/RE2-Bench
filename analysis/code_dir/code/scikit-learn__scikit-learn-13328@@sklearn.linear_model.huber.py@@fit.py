import numpy as np
from scipy import optimize
from ..base import BaseEstimator, RegressorMixin
from .base import LinearModel
from ..utils import check_X_y
from ..utils import check_consistent_length
from ..utils import axis0_safe_slice
from ..utils.extmath import safe_sparse_dot

class HuberRegressor(LinearModel, RegressorMixin, BaseEstimator):

    def __init__(self, epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.tol = tol

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, copy=False, accept_sparse=['csr'], y_numeric=True, dtype=[np.float64, np.float32])
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            check_consistent_length(y, sample_weight)
        else:
            sample_weight = np.ones_like(y)
        if self.epsilon < 1.0:
            raise ValueError('epsilon should be greater than or equal to 1.0, got %f' % self.epsilon)
        if self.warm_start and hasattr(self, 'coef_'):
            parameters = np.concatenate((self.coef_, [self.intercept_, self.scale_]))
        else:
            if self.fit_intercept:
                parameters = np.zeros(X.shape[1] + 2)
            else:
                parameters = np.zeros(X.shape[1] + 1)
            parameters[-1] = 1
        bounds = np.tile([-np.inf, np.inf], (parameters.shape[0], 1))
        bounds[-1][0] = np.finfo(np.float64).eps * 10
        parameters, f, dict_ = optimize.fmin_l_bfgs_b(_huber_loss_and_gradient, parameters, args=(X, y, self.epsilon, self.alpha, sample_weight), maxiter=self.max_iter, pgtol=self.tol, bounds=bounds, iprint=0)
        if dict_['warnflag'] == 2:
            raise ValueError('HuberRegressor convergence failed: l-BFGS-b solver terminated with %s' % dict_['task'].decode('ascii'))
        self.n_iter_ = min(dict_['nit'], self.max_iter)
        self.scale_ = parameters[-1]
        if self.fit_intercept:
            self.intercept_ = parameters[-2]
        else:
            self.intercept_ = 0.0
        self.coef_ = parameters[:X.shape[1]]
        residual = np.abs(y - safe_sparse_dot(X, self.coef_) - self.intercept_)
        self.outliers_ = residual > self.scale_ * self.epsilon
        return self