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
    if isinstance(raw_documents, str):
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
        if _IS_32BIT:
            raise ValueError('sparse CSR array has {} non-zero elements and requires 64 bit indexing, which is unsupported with 32 bit Python.'.format(indptr[-1]))
        indices_dtype = np.int64
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
        if self.input in ['file', 'filename']:
            self._validate_custom_analyzer()
        return partial(_analyze, analyzer=self.analyzer, decoder=self.decode)
    preprocess = self.build_preprocessor()
    if self.analyzer == 'char':
        return partial(_analyze, ngrams=self._char_ngrams, preprocessor=preprocess, decoder=self.decode)
    elif self.analyzer == 'char_wb':
        return partial(_analyze, ngrams=self._char_wb_ngrams, preprocessor=preprocess, decoder=self.decode)
    elif self.analyzer == 'word':
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()
        self._check_stop_words_consistency(stop_words, preprocess, tokenize)
        return partial(_analyze, ngrams=self._word_ngrams, tokenizer=tokenize, preprocessor=preprocess, decoder=self.decode, stop_words=stop_words)
    else:
        raise ValueError('%s is not a valid tokenization scheme/analyzer' % self.analyzer)

def build_preprocessor(self):
    if self.preprocessor is not None:
        return self.preprocessor
    if not self.strip_accents:
        strip_accents = None
    elif callable(self.strip_accents):
        strip_accents = self.strip_accents
    elif self.strip_accents == 'ascii':
        strip_accents = strip_accents_ascii
    elif self.strip_accents == 'unicode':
        strip_accents = strip_accents_unicode
    else:
        raise ValueError('Invalid value for "strip_accents": %s' % self.strip_accents)
    return partial(_preprocess, accent_function=strip_accents, lower=self.lowercase)

def get_stop_words(self):
    return _check_stop_list(self.stop_words)

def _check_stop_list(stop):
    if stop == 'english':
        return ENGLISH_STOP_WORDS
    elif isinstance(stop, str):
        raise ValueError('not a built-in stop list: %s' % stop)
    elif stop is None:
        return None
    else:
        return frozenset(stop)

def build_tokenizer(self):
    if self.tokenizer is not None:
        return self.tokenizer
    token_pattern = re.compile(self.token_pattern)
    return token_pattern.findall

def _check_stop_words_consistency(self, stop_words, preprocess, tokenize):
    if id(self.stop_words) == getattr(self, '_stop_words_id', None):
        return None
    try:
        inconsistent = set()
        for w in stop_words or ():
            tokens = list(tokenize(preprocess(w)))
            for token in tokens:
                if token not in stop_words:
                    inconsistent.add(token)
        self._stop_words_id = id(self.stop_words)
        if inconsistent:
            warnings.warn('Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens %r not in stop_words.' % sorted(inconsistent))
        return not inconsistent
    except Exception:
        self._stop_words_id = id(self.stop_words)
        return 'error'

def _make_int_array():
    return array.array(str('i'))

def _analyze(doc, analyzer=None, tokenizer=None, ngrams=None, preprocessor=None, decoder=None, stop_words=None):
    if decoder is not None:
        doc = decoder(doc)
    if analyzer is not None:
        doc = analyzer(doc)
    else:
        if preprocessor is not None:
            doc = preprocessor(doc)
        if tokenizer is not None:
            doc = tokenizer(doc)
        if ngrams is not None:
            if stop_words is not None:
                doc = ngrams(doc, stop_words)
            else:
                doc = ngrams(doc)
    return doc

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

def _preprocess(doc, accent_function=None, lower=False):
    if lower:
        doc = doc.lower()
    if accent_function is not None:
        doc = accent_function(doc)
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
        for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens_append(space_join(original_tokens[i:i + n]))
    return tokens



import array
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
import numbers
from operator import itemgetter
import re
import unicodedata
import warnings
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, TransformerMixin
from ..preprocessing import normalize
from .hashing import FeatureHasher
from .stop_words import ENGLISH_STOP_WORDS
from ..utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
from ..utils import _IS_32BIT
from ..utils.fixes import _astype_copy_false
from ..exceptions import ChangedBehaviorWarning
import tempfile
__all__ = ['HashingVectorizer', 'CountVectorizer', 'ENGLISH_STOP_WORDS', 'TfidfTransformer', 'TfidfVectorizer', 'strip_accents_ascii', 'strip_accents_unicode', 'strip_tags']

class TfidfVectorizer(CountVectorizer):

    def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        super().__init__(input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer, stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype)
        self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)

    @property
    def norm(self):
        return self._tfidf.norm

    @norm.setter
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def smooth_idf(self):
        return self._tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        return self._tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tfidf.sublinear_tf = value

    @property
    def idf_(self):
        return self._tfidf.idf_

    @idf_.setter
    def idf_(self, value):
        self._validate_vocabulary()
        if hasattr(self, 'vocabulary_'):
            if len(self.vocabulary_) != len(value):
                raise ValueError('idf length = %d must be equal to vocabulary size = %d' % (len(value), len(self.vocabulary)))
        self._tfidf.idf_ = value

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn("Only {} 'dtype' should be used. {} 'dtype' will be converted to np.float64.".format(FLOAT_DTYPES, self.dtype), UserWarning)

    def fit(self, raw_documents, y=None):
        self._check_params()
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        self._check_params()
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents, copy='deprecated'):
        check_is_fitted(self, '_tfidf', 'The tfidf vector is not fitted')
        if copy != 'deprecated':
            msg = "'copy' param is unused and has been deprecated since version 0.22. Backward compatibility for 'copy' will be removed in 0.24."
            warnings.warn(msg, DeprecationWarning)
        X = super().transform(raw_documents)
        return self._tfidf.transform(X, copy=False)

    def _more_tags(self):
        return {'X_types': ['string'], '_skip_test': True}