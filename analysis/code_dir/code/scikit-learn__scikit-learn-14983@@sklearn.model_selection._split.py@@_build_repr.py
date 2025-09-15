from collections.abc import Iterable
import warnings
from itertools import chain, combinations
from math import ceil, floor
import numbers
from abc import ABCMeta, abstractmethod
from inspect import signature
import numpy as np
from ..utils import indexable, check_random_state, safe_indexing
from ..utils import _approximate_mode
from ..utils.validation import _num_samples, column_or_1d
from ..utils.validation import check_array
from ..utils.multiclass import type_of_target
from ..utils.fixes import comb
from ..base import _pprint
__all__ = ['BaseCrossValidator', 'KFold', 'GroupKFold', 'LeaveOneGroupOut', 'LeaveOneOut', 'LeavePGroupsOut', 'LeavePOut', 'RepeatedStratifiedKFold', 'RepeatedKFold', 'ShuffleSplit', 'GroupShuffleSplit', 'StratifiedKFold', 'StratifiedShuffleSplit', 'PredefinedSplit', 'train_test_split', 'check_cv']
train_test_split.__test__ = False

def _build_repr(self):
    cls = self.__class__
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    init_signature = signature(init)
    if init is object.__init__:
        args = []
    else:
        args = sorted([p.name for p in init_signature.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD])
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        warnings.simplefilter('always', DeprecationWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
                if value is None and hasattr(self, 'cvargs'):
                    value = self.cvargs.get(key, None)
            if len(w) and w[0].category == DeprecationWarning:
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value
    return '%s(%s)' % (class_name, _pprint(params, offset=len(class_name)))