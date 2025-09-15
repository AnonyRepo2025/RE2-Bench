import warnings
import numpy as np
from ..base import BaseEstimator, RegressorMixin, clone
from ..utils.validation import check_is_fitted
from ..utils import check_array, safe_indexing
from ..preprocessing import FunctionTransformer
from ..linear_model import LinearRegression
__all__ = ['TransformedTargetRegressor']

class TransformedTargetRegressor(RegressorMixin, BaseEstimator):

    def __init__(self, regressor=None, transformer=None, func=None, inverse_func=None, check_inverse=True):
        self.regressor = regressor
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse

    def _fit_transformer(self, y):
        if self.transformer is not None and (self.func is not None or self.inverse_func is not None):
            raise ValueError("'transformer' and functions 'func'/'inverse_func' cannot both be set.")
        elif self.transformer is not None:
            self.transformer_ = clone(self.transformer)
        else:
            if self.func is not None and self.inverse_func is None:
                raise ValueError("When 'func' is provided, 'inverse_func' must also be provided")
            self.transformer_ = FunctionTransformer(func=self.func, inverse_func=self.inverse_func, validate=True, check_inverse=self.check_inverse)
        self.transformer_.fit(y)
        if self.check_inverse:
            idx_selected = slice(None, None, max(1, y.shape[0] // 10))
            y_sel = safe_indexing(y, idx_selected)
            y_sel_t = self.transformer_.transform(y_sel)
            if not np.allclose(y_sel, self.transformer_.inverse_transform(y_sel_t)):
                warnings.warn("The provided functions or transformer are not strictly inverse of each other. If you are sure you want to proceed regardless, set 'check_inverse=False'", UserWarning)

    def fit(self, X, y, **fit_params):
        y = check_array(y, accept_sparse=False, force_all_finite=True, ensure_2d=False, dtype='numeric')
        self._training_dim = y.ndim
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self._fit_transformer(y_2d)
        y_trans = self.transformer_.transform(y_2d)
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)
        if self.regressor is None:
            from ..linear_model import LinearRegression
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y_trans, **fit_params)
        return self

    def predict(self, X):
        check_is_fitted(self)
        pred = self.regressor_.predict(X)
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)
        if self._training_dim == 1 and pred_trans.ndim == 2 and (pred_trans.shape[1] == 1):
            pred_trans = pred_trans.squeeze(axis=1)
        return pred_trans

    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}