def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    if msg is None:
        msg = "This %(name)s instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
    if not hasattr(estimator, 'fit'):
        raise TypeError('%s is not an estimator instance.' % estimator)
    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]
    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})

def transform(self, raw_documents):
    if isinstance(raw_documents, six.string_types):
        raise ValueError('Iterable over raw text documents expected, string object received.')
    if not hasattr(self, 'vocabulary_'):
        self._validate_vocabulary()
    self._check_vocabulary()
    _, X = self._count_vocab(raw_documents, fixed_vocab=True)
    if self.binary:
        X.data.fill(1)
    return X

def _check_vocabulary(self):
    msg = "%(name)s - Vocabulary wasn't fitted."
    (check_is_fitted(self, 'vocabulary_', msg=msg),)
    if len(self.vocabulary_) == 0:
        raise ValueError('Vocabulary is empty')

def _count_vocab(self, raw_documents, fixed_vocab):
    if fixed_vocab:
        vocabulary = self.vocabulary_
    else:
        vocabulary = defaultdict()
        vocabulary.default_factory = vocabulary.__len__
    analyze = self.build_analyzer()
    j_indices = []
    indptr = []
    values = _make_int_array()
    indptr.append(0)
    for doc in raw_documents:
        feature_counter = {}
        for feature in analyze(doc):
            try:
                feature_idx = vocabulary[feature]
                if feature_idx not in feature_counter:
                    feature_counter[feature_idx] = 1
                else:
                    feature_counter[feature_idx] += 1
            except KeyError:
                continue
        j_indices.extend(feature_counter.keys())
        values.extend(feature_counter.values())
        indptr.append(len(j_indices))
    if not fixed_vocab:
        vocabulary = dict(vocabulary)
        if not vocabulary:
            raise ValueError('empty vocabulary; perhaps the documents only contain stop words')
    if indptr[-1] > 2147483648:
        if sp_version >= (0, 14):
            indices_dtype = np.int64
        else:
            raise ValueError('sparse CSR array has {} non-zero elements and requires 64 bit indexing,  which is unsupported with scipy {}. Please upgrade to scipy >=0.14'.format(indptr[-1], '.'.join(sp_version)))
    else:
        indices_dtype = np.int32
    j_indices = np.asarray(j_indices, dtype=indices_dtype)
    indptr = np.asarray(indptr, dtype=indices_dtype)
    values = np.frombuffer(values, dtype=np.intc)
    X = sp.csr_matrix((values, j_indices, indptr), shape=(len(indptr) - 1, len(vocabulary)), dtype=self.dtype)
    X.sort_indices()
    return (vocabulary, X)

def build_analyzer(self):
    if callable(self.analyzer):
        return self.analyzer
    preprocess = self.build_preprocessor()
    if self.analyzer == 'char':
        return lambda doc: self._char_ngrams(preprocess(self.decode(doc)))
    elif self.analyzer == 'char_wb':
        return lambda doc: self._char_wb_ngrams(preprocess(self.decode(doc)))
    elif self.analyzer == 'word':
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()
        return lambda doc: self._word_ngrams(tokenize(preprocess(self.decode(doc))), stop_words)
    else:
        raise ValueError('%s is not a valid tokenization scheme/analyzer' % self.analyzer)

def build_preprocessor(self):
    if self.preprocessor is not None:
        return self.preprocessor
    noop = lambda x: x
    if not self.strip_accents:
        strip_accents = noop
    elif callable(self.strip_accents):
        strip_accents = self.strip_accents
    elif self.strip_accents == 'ascii':
        strip_accents = strip_accents_ascii
    elif self.strip_accents == 'unicode':
        strip_accents = strip_accents_unicode
    else:
        raise ValueError('Invalid value for "strip_accents": %s' % self.strip_accents)
    if self.lowercase:
        return lambda x: strip_accents(x.lower())
    else:
        return strip_accents

def get_stop_words(self):
    return _check_stop_list(self.stop_words)

def _check_stop_list(stop):
    if stop == 'english':
        return ENGLISH_STOP_WORDS
    elif isinstance(stop, six.string_types):
        raise ValueError('not a built-in stop list: %s' % stop)
    elif stop is None:
        return None
    else:
        return frozenset(stop)

def build_tokenizer(self):
    if self.tokenizer is not None:
        return self.tokenizer
    token_pattern = re.compile(self.token_pattern)
    return lambda doc: token_pattern.findall(doc)

def _make_int_array():
    return array.array(str('i'))

def decode(self, doc):
    if self.input == 'filename':
        with open(doc, 'rb') as fh:
            doc = fh.read()
    elif self.input == 'file':
        doc = doc.read()
    if isinstance(doc, bytes):
        doc = doc.decode(self.encoding, self.decode_error)
    if doc is np.nan:
        raise ValueError('np.nan is an invalid document, expected byte or unicode string.')
    return doc

def _word_ngrams(self, tokens, stop_words=None):
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]
    min_n, max_n = self.ngram_range
    if max_n != 1:
        original_tokens = tokens
        if min_n == 1:
            tokens = list(original_tokens)
            min_n += 1
        else:
            tokens = []
        n_original_tokens = len(original_tokens)
        tokens_append = tokens.append
        space_join = ' '.join
        for n in xrange(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in xrange(n_original_tokens - n + 1):
                tokens_append(space_join(original_tokens[i:i + n]))
    return tokens

def assert_raise_message(exceptions, message, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exceptions as e:
        error_message = str(e)
        if message not in error_message:
            raise AssertionError('Error message does not include the expected string: %r. Observed error message: %r' % (message, error_message))
    else:
        if isinstance(exceptions, tuple):
            names = ' or '.join((e.__name__ for e in exceptions))
        else:
            names = exceptions.__name__
        raise AssertionError('%s not raised by %s' % (names, function.__name__))

def fit_transform(self, X, y=None):
    return self.fit(X, y).transform(X)

def fit(self, X, y=None):
    if isinstance(X, six.string_types):
        raise ValueError('Iterable over raw text documents expected, string object received.')
    self._validate_params()
    self._get_hasher().fit(X, y=y)
    return self



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