from __future__ import unicode_literals, division
import array
from collections import Mapping, defaultdict
import numbers
from operator import itemgetter
import re
import unicodedata
import warnings
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, TransformerMixin
from ..externals import six
from ..externals.six.moves import xrange
from ..preprocessing import normalize
from .hashing import FeatureHasher
from .stop_words import ENGLISH_STOP_WORDS
from ..utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
from ..utils.fixes import sp_version
__all__ = ['CountVectorizer', 'ENGLISH_STOP_WORDS', 'TfidfTransformer', 'TfidfVectorizer', 'strip_accents_ascii', 'strip_accents_unicode', 'strip_tags']

class TfidfTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X).astype(dtype)
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)
            idf = np.log(n_samples / df) + 1
            self._idf_diag = sp.diags(idf, offsets=0, shape=(n_features, n_features), format='csr', dtype=dtype)
        return self

    def transform(self, X, copy=True):
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)
        n_samples, n_features = X.shape
        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1
        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')
            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError('Input has n_features=%d while the model has been trained with n_features=%d' % (n_features, expected_n_features))
            X = X * self._idf_diag
        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    @property
    def idf_(self):
        return np.ravel(self._idf_diag.sum(axis=0))

    @idf_.setter
    def idf_(self, value):
        value = np.asarray(value, dtype=np.float64)
        n_features = value.shape[0]
        self._idf_diag = sp.spdiags(value, diags=0, m=n_features, n=n_features, format='csr')