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

class StratifiedKFold(_BaseKFold):

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle, random_state)

    def _make_test_folds(self, X, y=None):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError('Supported target types are: {}. Got {!r} instead.'.format(allowed_target_types, type_of_target_y))
        y = column_or_1d(y)
        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]
        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError('n_splits=%d cannot be greater than the number of members in each class.' % self.n_splits)
        if self.n_splits > min_groups:
            warnings.warn('The least populated class in y has only %d members, which is less than n_splits=%d.' % (min_groups, self.n_splits), UserWarning)
        y_order = np.sort(y_encoded)
        allocation = np.asarray([np.bincount(y_order[i::self.n_splits], minlength=n_classes) for i in range(self.n_splits)])
        test_folds = np.empty(len(y), dtype='i')
        for k in range(n_classes):
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield (test_folds == i)

    def split(self, X, y, groups=None):
        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)