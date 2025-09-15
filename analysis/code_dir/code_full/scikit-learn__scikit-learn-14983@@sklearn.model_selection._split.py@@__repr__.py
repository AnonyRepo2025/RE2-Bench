

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

class BaseCrossValidator(metaclass=ABCMeta):

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield (train_index, test_index)

    def _iter_test_masks(self, X=None, y=None, groups=None):
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(_num_samples(X), dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        pass
    def __repr__(self):
        return _build_repr(self)