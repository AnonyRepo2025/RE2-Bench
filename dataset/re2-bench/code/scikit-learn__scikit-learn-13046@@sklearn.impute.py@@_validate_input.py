import warnings
import numbers
import numpy as np
import numpy.ma as ma
from scipy import sparse
from scipy import stats
from .base import BaseEstimator, TransformerMixin
from .utils import check_array
from .utils.sparsefuncs import _get_median
from .utils.validation import check_is_fitted
from .utils.validation import FLOAT_DTYPES
from .utils.fixes import _object_dtype_isnan
from .utils import is_scalar_nan

__all__ = [
    'MissingIndicator',
    'SimpleImputer',
]

class MissingIndicator(BaseEstimator, TransformerMixin):
    """Binary indicators for missing values.

    Note that this component typically should not not be used in a vanilla
    :class:`Pipeline` consisting of transformers and a classifier, but rather
    could be added using a :class:`FeatureUnion` or :class:`ColumnTransformer`.

    Read more in the :ref:`User Guide <impute>`.

    Parameters
    ----------
    missing_values : number, string, np.nan (default) or None
        The placeholder for the missing values. All occurrences of
        `missing_values` will be indicated (True in the output array), the
        other values will be marked as False.

    features : str, optional
        Whether the imputer mask should represent all or a subset of
        features.

        - If "missing-only" (default), the imputer mask will only represent
          features containing missing values during fit time.
        - If "all", the imputer mask will represent all features.

    sparse : boolean or "auto", optional
        Whether the imputer mask format should be sparse or dense.

        - If "auto" (default), the imputer mask will be of same type as
          input.
        - If True, the imputer mask will be a sparse matrix.
        - If False, the imputer mask will be a numpy array.

    error_on_new : boolean, optional
        If True (default), transform will raise an error when there are
        features with missing values in transform that have no missing values
        in fit. This is applicable only when ``features="missing-only"``.

    Attributes
    ----------
    features_ : ndarray, shape (n_missing_features,) or (n_features,)
        The features indices which will be returned when calling ``transform``.
        They are computed during ``fit``. For ``features='all'``, it is
        to ``range(n_features)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import MissingIndicator
    >>> X1 = np.array([[np.nan, 1, 3],
    ...                [4, 0, np.nan],
    ...                [8, 1, 0]])
    >>> X2 = np.array([[5, 1, np.nan],
    ...                [np.nan, 2, 3],
    ...                [2, 4, 0]])
    >>> indicator = MissingIndicator()
    >>> indicator.fit(X1)  # doctest: +NORMALIZE_WHITESPACE
    MissingIndicator(error_on_new=True, features='missing-only',
             missing_values=nan, sparse='auto')
    >>> X2_tr = indicator.transform(X2)
    >>> X2_tr
    array([[False,  True],
           [ True, False],
           [False, False]])

    """

    def __init__(self, missing_values=np.nan, features="missing-only",
                 sparse="auto", error_on_new=True):
        self.missing_values = missing_values
        self.features = features
        self.sparse = sparse
        self.error_on_new = error_on_new

    def _get_missing_features_info(self, X):
        """Compute the imputer mask and the indices of the features
        containing missing values.

        Parameters
        ----------
        X : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The input data with missing values. Note that ``X`` has been
            checked in ``fit`` and ``transform`` before to call this function.

        Returns
        -------
        imputer_mask : {ndarray or sparse matrix}, shape \
(n_samples, n_features) or (n_samples, n_features_with_missing)
            The imputer mask of the original data.

        features_with_missing : ndarray, shape (n_features_with_missing)
            The features containing missing values.

        """
        if sparse.issparse(X) and self.missing_values != 0:
            mask = _get_mask(X.data, self.missing_values)

            # The imputer mask will be constructed with the same sparse format
            # as X.
            sparse_constructor = (sparse.csr_matrix if X.format == 'csr'
                                  else sparse.csc_matrix)
            imputer_mask = sparse_constructor(
                (mask, X.indices.copy(), X.indptr.copy()),
                shape=X.shape, dtype=bool)

            missing_values_mask = imputer_mask.copy()
            missing_values_mask.eliminate_zeros()
            features_with_missing = (
                np.flatnonzero(np.diff(missing_values_mask.indptr))
                if missing_values_mask.format == 'csc'
                else np.unique(missing_values_mask.indices))

            if self.sparse is False:
                imputer_mask = imputer_mask.toarray()
            elif imputer_mask.format == 'csr':
                imputer_mask = imputer_mask.tocsc()
        else:
            if sparse.issparse(X):
                # case of sparse matrix with 0 as missing values. Implicit and
                # explicit zeros are considered as missing values.
                X = X.toarray()
            imputer_mask = _get_mask(X, self.missing_values)
            features_with_missing = np.flatnonzero(imputer_mask.sum(axis=0))

            if self.sparse is True:
                imputer_mask = sparse.csc_matrix(imputer_mask)

        return imputer_mask, features_with_missing

    def _validate_input(self, X):
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"
        X = check_array(X, accept_sparse=('csc', 'csr'), dtype=None,
                        force_all_finite=force_all_finite)
        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError("MissingIndicator does not support data with "
                             "dtype {0}. Please provide either a numeric array"
                             " (with a floating point or integer dtype) or "
                             "categorical data represented either as an array "
                             "with integer dtype or an array of string values "
                             "with an object dtype.".format(X.dtype))
        return X

    def fit(self, X, y=None):
        """Fit the transformer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_input(X)
        self._n_features = X.shape[1]

        if self.features not in ('missing-only', 'all'):
            raise ValueError("'features' has to be either 'missing-only' or "
                             "'all'. Got {} instead.".format(self.features))

        if not ((isinstance(self.sparse, str) and
                self.sparse == "auto") or isinstance(self.sparse, bool)):
            raise ValueError("'sparse' has to be a boolean or 'auto'. "
                             "Got {!r} instead.".format(self.sparse))

        self.features_ = (self._get_missing_features_info(X)[1]
                          if self.features == 'missing-only'
                          else np.arange(self._n_features))

        return self

    def transform(self, X):
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        """
        check_is_fitted(self, "features_")
        X = self._validate_input(X)

        if X.shape[1] != self._n_features:
            raise ValueError("X has a different number of features "
                             "than during fitting.")

        imputer_mask, features = self._get_missing_features_info(X)

        if self.features == "missing-only":
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            if (self.error_on_new and features_diff_fit_trans.size > 0):
                raise ValueError("The features {} have missing values "
                                 "in transform but have no missing values "
                                 "in fit.".format(features_diff_fit_trans))

            if (self.features_.size > 0 and
                    self.features_.size < self._n_features):
                imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def fit_transform(self, X, y=None):
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        """
        return self.fit(X, y).transform(X)
