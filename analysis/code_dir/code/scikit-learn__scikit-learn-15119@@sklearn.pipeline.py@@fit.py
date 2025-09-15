from collections import defaultdict
from itertools import islice
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from .base import clone, TransformerMixin
from .utils.metaestimators import if_delegate_has_method
from .utils import Bunch, _print_elapsed_time
from .utils.validation import check_memory
from .utils.metaestimators import _BaseComposition
__all__ = ['Pipeline', 'FeatureUnion', 'make_pipeline', 'make_union']

class FeatureUnion(TransformerMixin, _BaseComposition):
    _required_parameters = ['transformer_list']

    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self._validate_transformers()

    def get_params(self, deep=True):
        return self._get_params('transformer_list', deep=deep)

    def set_params(self, **kwargs):
        self._set_params('transformer_list', **kwargs)
        return self

    def _validate_transformers(self):
        names, transformers = zip(*self.transformer_list)
        self._validate_names(names)
        for t in transformers:
            if t is None or t == 'drop':
                continue
            if not (hasattr(t, 'fit') or hasattr(t, 'fit_transform')) or not hasattr(t, 'transform'):
                raise TypeError("All estimators should implement fit and transform. '%s' (type %s) doesn't" % (t, type(t)))

    def _iter(self):
        get_weight = (self.transformer_weights or {}).get
        return ((name, trans, get_weight(name)) for name, trans in self.transformer_list if trans is not None and trans != 'drop')

    def get_feature_names(self):
        feature_names = []
        for name, trans, weight in self._iter():
            if not hasattr(trans, 'get_feature_names'):
                raise AttributeError('Transformer %s (type %s) does not provide get_feature_names.' % (str(name), type(trans).__name__))
            feature_names.extend([name + '__' + f for f in trans.get_feature_names()])
        return feature_names

    def fit(self, X, y=None, **fit_params):
        transformers = self._parallel_func(X, y, fit_params, _fit_one)
        if not transformers:
            return self
        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)
        if any((sparse.issparse(f) for f in Xs)):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return '(step %d of %d) Processing %s' % (idx, total, name)

    def _parallel_func(self, X, y, fit_params, func):
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        transformers = list(self._iter())
        return Parallel(n_jobs=self.n_jobs)((delayed(func)(transformer, X, y, weight, message_clsname='FeatureUnion', message=self._log_message(name, idx, len(transformers)), **fit_params) for idx, (name, transformer, weight) in enumerate(transformers, 1)))

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)((delayed(_transform_one)(trans, X, None, weight) for name, trans, weight in self._iter()))
        if not Xs:
            return np.zeros((X.shape[0], 0))
        if any((sparse.issparse(f) for f in Xs)):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def _update_transformer_list(self, transformers):
        transformers = iter(transformers)
        self.transformer_list[:] = [(name, old if old is None or old == 'drop' else next(transformers)) for name, old in self.transformer_list]