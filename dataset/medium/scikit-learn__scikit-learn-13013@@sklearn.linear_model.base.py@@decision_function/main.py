from abc import ABCMeta, abstractmethod
import numbers
import warnings
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy import sparse
from scipy.special import expit
from ..utils._joblib import Parallel, delayed
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                    MultiOutputMixin)
from ..utils import check_array, check_X_y
from ..utils.validation import FLOAT_DTYPES
from ..utils import check_random_state
from ..utils.extmath import safe_sparse_dot
from ..utils.sparsefuncs import mean_variance_axis, inplace_column_scale
from ..utils.fixes import sparse_lsqr
from ..utils.seq_dataset import ArrayDataset32, CSRDataset32
from ..utils.seq_dataset import ArrayDataset64, CSRDataset64
from ..utils.validation import check_is_fitted
from ..preprocessing.data import normalize as f_normalize

SPARSE_INTERCEPT_DECAY = 0.01

class LinearClassifierMixin(ClassifierMixin):

    def decision_function(self, X):
        check_is_fitted(self, 'coef_')

        X = self.check_array(X, accept_sparse='csr')

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))

        scores = safe_sparse_dot(X, self.coef_.T,
                                 dense_output=True) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def _predict_proba_lr(self, X):
        prob = self.decision_function(X)
        expit(prob, out=prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob
    def check_array(self, array, accept_sparse=False, accept_large_sparse=True,
                    dtype="numeric", order=None, copy=False, force_all_finite=True,
                    ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                    ensure_min_features=1, warn_on_dtype=None, estimator=None):
        if warn_on_dtype is not None:
            warnings.warn(
                "'warn_on_dtype' is deprecated in version 0.21 and will be "
                "removed in 0.23. Don't set `warn_on_dtype` to remove this "
                "warning.",
                DeprecationWarning)

        array_orig = array

        dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

        dtype_orig = getattr(array, "dtype", None)
        if not hasattr(dtype_orig, 'kind'):
            dtype_orig = None

        dtypes_orig = None
        if hasattr(array, "dtypes") and hasattr(array.dtypes, '__array__'):
            dtypes_orig = np.array(array.dtypes)

        if dtype_numeric:
            if dtype_orig is not None and dtype_orig.kind == "O":
                # if input is object, convert to float.
                dtype = np.float64
            else:
                dtype = None

        if isinstance(dtype, (list, tuple)):
            if dtype_orig is not None and dtype_orig in dtype:
                # no dtype conversion required
                dtype = None
            else:
                # dtype conversion required. Let's select the first element of the
                # list of accepted types.
                dtype = dtype[0]

        if force_all_finite not in (True, False, 'allow-nan'):
            raise ValueError('force_all_finite should be a bool or "allow-nan"'
                            '. Got {!r} instead'.format(force_all_finite))

        if estimator is not None:
            if isinstance(estimator, str):
                estimator_name = estimator
            else:
                estimator_name = estimator.__class__.__name__
        else:
            estimator_name = "Estimator"
        context = " by %s" % estimator_name if estimator is not None else ""

        if sp.issparse(array):
            _ensure_no_complex_data(array)
            array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                        dtype=dtype, copy=copy,
                                        force_all_finite=force_all_finite,
                                        accept_large_sparse=accept_large_sparse)
        else:
            with warnings.catch_warnings():
                try:
                    warnings.simplefilter('error', ComplexWarning)
                    array = np.asarray(array, dtype=dtype, order=order)
                except ComplexWarning:
                    raise ValueError("Complex data not supported\n"
                                    "{}\n".format(array))
            _ensure_no_complex_data(array)

            if ensure_2d:
                # If input is scalar raise error
                if array.ndim == 0:
                    raise ValueError(
                        "Expected 2D array, got scalar array instead:\narray={}.\n"
                        "Reshape your data either using array.reshape(-1, 1) if "
                        "your data has a single feature or array.reshape(1, -1) "
                        "if it contains a single sample.".format(array))
                # If input is 1D raise error
                if array.ndim == 1:
                    raise ValueError(
                        "Expected 2D array, got 1D array instead:\narray={}.\n"
                        "Reshape your data either using array.reshape(-1, 1) if "
                        "your data has a single feature or array.reshape(1, -1) "
                        "if it contains a single sample.".format(array))

            # in the future np.flexible dtypes will be handled like object dtypes
            if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
                warnings.warn(
                    "Beginning in version 0.22, arrays of bytes/strings will be "
                    "converted to decimal numbers if dtype='numeric'. "
                    "It is recommended that you convert the array to "
                    "a float dtype before using it in scikit-learn, "
                    "for example by using "
                    "your_array = your_array.astype(np.float64).",
                    FutureWarning)

            # make sure we actually converted to numeric:
            if dtype_numeric and array.dtype.kind == "O":
                array = array.astype(np.float64)
            if not allow_nd and array.ndim >= 3:
                raise ValueError("Found array with dim %d. %s expected <= 2."
                                % (array.ndim, estimator_name))
            if force_all_finite:
                _assert_all_finite(array,
                                allow_nan=force_all_finite == 'allow-nan')

        if ensure_min_samples > 0:
            n_samples = _num_samples(array)
            if n_samples < ensure_min_samples:
                raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                                " minimum of %d is required%s."
                                % (n_samples, array.shape, ensure_min_samples,
                                    context))

        if ensure_min_features > 0 and array.ndim == 2:
            n_features = array.shape[1]
            if n_features < ensure_min_features:
                raise ValueError("Found array with %d feature(s) (shape=%s) while"
                                " a minimum of %d is required%s."
                                % (n_features, array.shape, ensure_min_features,
                                    context))

        if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
            msg = ("Data with input dtype %s was converted to %s%s."
                % (dtype_orig, array.dtype, context))
            warnings.warn(msg, DataConversionWarning)

        if copy and np.may_share_memory(array, array_orig):
            array = np.array(array, dtype=dtype, order=order)

        if (warn_on_dtype and dtypes_orig is not None and
                {array.dtype} != set(dtypes_orig)):
            msg = ("Data with input dtype %s were all converted to %s%s."
                % (', '.join(map(str, sorted(set(dtypes_orig)))), array.dtype,
                    context))
            warnings.warn(msg, DataConversionWarning, stacklevel=3)

        return array
