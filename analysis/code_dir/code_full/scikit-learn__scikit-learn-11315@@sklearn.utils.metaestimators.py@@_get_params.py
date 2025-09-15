def get_params(self, deep=True):
    out = dict()
    for key in self._get_param_names():
        value = getattr(self, key, None)
        if deep and hasattr(value, 'get_params'):
            deep_items = value.get_params().items()
            out.update(((key + '__' + k, val) for k, val in deep_items))
        out[key] = value
    return out

def _get_param_names(cls):
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    if init is object.__init__:
        return []
    init_signature = signature(init)
    parameters = [p for p in init_signature.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD]
    for p in parameters:
        if p.kind == p.VAR_POSITIONAL:
            raise RuntimeError("scikit-learn estimators should always specify their parameters in the signature of their __init__ (no varargs). %s with constructor %s doesn't  follow this convention." % (cls, init_signature))
    return sorted([p.name for p in parameters])

def _transformers(self):
    return [(name, trans) for name, trans, _ in self.transformers]

def iteritems(d, **kw):
    return iter(getattr(d, _iteritems)(**kw))



from abc import ABCMeta, abstractmethod
from operator import attrgetter
from functools import update_wrapper
import numpy as np
from ..utils import safe_indexing
from ..externals import six
from ..base import BaseEstimator
__all__ = ['if_delegate_has_method']

class _BaseComposition(six.with_metaclass(ABCMeta, BaseEstimator)):

    @abstractmethod
    def __init__(self):
        pass

    def _get_params(self, attr, deep=True):
        out = super(_BaseComposition, self).get_params(deep=deep)
        if not deep:
            return out
        estimators = getattr(self, attr)
        out.update(estimators)
        for name, estimator in estimators:
            if hasattr(estimator, 'get_params'):
                for key, value in six.iteritems(estimator.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
        return out

    def _set_params(self, attr, **params):
        if attr in params:
            setattr(self, attr, params.pop(attr))
        names, _ = zip(*getattr(self, attr))
        for name in list(six.iterkeys(params)):
            if '__' not in name and name in names:
                self._replace_estimator(attr, name, params.pop(name))
        super(_BaseComposition, self).set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: {0!r}'.format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor arguments: {0!r}'.format(sorted(invalid_names)))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got {0!r}'.format(invalid_names))