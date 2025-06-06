from math import log
import numpy as np
from scipy import linalg
from scipy.linalg import pinvh
from .base import LinearModel, _rescale_data
from ..base import RegressorMixin
from ..utils.extmath import fast_logdet
from ..utils import check_X_y



class BayesianRidge(LinearModel, RegressorMixin):

    def __init__(self, n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6,
                 lambda_1=1.e-6, lambda_2=1.e-6, alpha_init=None,
                 lambda_init=None, compute_score=False, fit_intercept=True,
                 normalize=False, copy_X=True, verbose=False):
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):

        if self.n_iter < 1:
            raise ValueError('n_iter should be greater than or equal to 1.'
                             ' Got {!r}.'.format(self.n_iter))

        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        X, y, X_offset_, y_offset_, X_scale_ = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_
        n_samples, n_features = X.shape

        eps = np.finfo(np.float64).eps
        alpha_ = self.alpha_init
        lambda_ = self.lambda_init
        if alpha_ is None:
            alpha_ = 1. / (np.var(y) + eps)
        if lambda_ is None:
            lambda_ = 1.

        verbose = self.verbose
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2

        self.scores_ = list()
        coef_old_ = None

        XT_y = np.dot(X.T, y)
        U, S, Vh = linalg.svd(X, full_matrices=False)
        eigen_vals_ = S ** 2

        for iter_ in range(self.n_iter):

            # update posterior mean coef_ based on alpha_ and lambda_ and
            # compute corresponding rmse
            coef_, rmse_ = self._update_coef_(X, y, n_samples, n_features,
                                              XT_y, U, Vh, eigen_vals_,
                                              alpha_, lambda_)
            if self.compute_score:
                # compute the log marginal likelihood
                s = self._log_marginal_likelihood(n_samples, n_features,
                                                  eigen_vals_,
                                                  alpha_, lambda_,
                                                  coef_, rmse_)
                self.scores_.append(s)

            # Update alpha and lambda according to (MacKay, 1992)
            gamma_ = np.sum((alpha_ * eigen_vals_) /
                            (lambda_ + alpha_ * eigen_vals_))
            lambda_ = ((gamma_ + 2 * lambda_1) /
                       (np.sum(coef_ ** 2) + 2 * lambda_2))
            alpha_ = ((n_samples - gamma_ + 2 * alpha_1) /
                      (rmse_ + 2 * alpha_2))

            # Check for convergence
            if iter_ != 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break
            coef_old_ = np.copy(coef_)

        self.n_iter_ = iter_ + 1

        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.coef_, rmse_ = self._update_coef_(X, y, n_samples, n_features,
                                               XT_y, U, Vh, eigen_vals_,
                                               alpha_, lambda_)
        if self.compute_score:
            # compute the log marginal likelihood
            s = self._log_marginal_likelihood(n_samples, n_features,
                                              eigen_vals_,
                                              alpha_, lambda_,
                                              coef_, rmse_)
            self.scores_.append(s)
            self.scores_ = np.array(self.scores_)

        # posterior covariance is given by 1/alpha_ * scaled_sigma_
        scaled_sigma_ = np.dot(Vh.T,
                               Vh / (eigen_vals_ +
                                     lambda_ / alpha_)[:, np.newaxis])
        self.sigma_ = (1. / alpha_) * scaled_sigma_

        self._set_intercept(X_offset_, y_offset_, X_scale_)

        return self



    def _update_coef_(self, X, y, n_samples, n_features, XT_y, U, Vh,
                      eigen_vals_, alpha_, lambda_):
        """Update posterior mean and compute corresponding rmse.

        Posterior mean is given by coef_ = scaled_sigma_ * X.T * y where
        scaled_sigma_ = (lambda_/alpha_ * np.eye(n_features)
                         + np.dot(X.T, X))^-1
        """

        if n_samples > n_features:
            coef_ = np.dot(Vh.T,
                           Vh / (eigen_vals_ +
                                 lambda_ / alpha_)[:, np.newaxis])
            coef_ = np.dot(coef_, XT_y)
        else:
            coef_ = np.dot(X.T, np.dot(
                U / (eigen_vals_ + lambda_ / alpha_)[None, :], U.T))
            coef_ = np.dot(coef_, y)

        rmse_ = np.sum((y - np.dot(X, coef_)) ** 2)

        return coef_, rmse_

    def _log_marginal_likelihood(self, n_samples, n_features, eigen_vals,
                                 alpha_, lambda_, coef, rmse):
        """Log marginal likelihood."""
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        # compute the log of the determinant of the posterior covariance.
        # posterior covariance is given by
        # sigma = (lambda_ * np.eye(n_features) + alpha_ * np.dot(X.T, X))^-1
        if n_samples > n_features:
            logdet_sigma = - np.sum(np.log(lambda_ + alpha_ * eigen_vals))
        else:
            logdet_sigma = np.full(n_features, lambda_,
                                   dtype=np.array(lambda_).dtype)
            logdet_sigma[:n_samples] += alpha_ * eigen_vals
            logdet_sigma = - np.sum(np.log(logdet_sigma))

        score = lambda_1 * log(lambda_) - lambda_2 * lambda_
        score += alpha_1 * log(alpha_) - alpha_2 * alpha_
        score += 0.5 * (n_features * log(lambda_) +
                        n_samples * log(alpha_) -
                        alpha_ * rmse -
                        lambda_ * np.sum(coef ** 2) +
                        logdet_sigma -
                        n_samples * log(2 * np.pi))

        return score
