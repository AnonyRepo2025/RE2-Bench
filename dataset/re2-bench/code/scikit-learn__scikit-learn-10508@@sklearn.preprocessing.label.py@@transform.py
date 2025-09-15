from collections import defaultdict
import itertools
import array
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, TransformerMixin
from ..utils.fixes import sparse_min_max
from ..utils import column_or_1d
from ..utils.validation import check_array
from ..utils.validation import check_is_fitted
from ..utils.validation import _num_samples
from ..utils.multiclass import unique_labels
from ..utils.multiclass import type_of_target
from ..externals import six

zip = six.moves.zip
map = six.moves.map
__all__ = [
    'label_binarize',
    'LabelBinarizer',
    'LabelEncoder',
    'MultiLabelBinarizer',
]

class LabelEncoder(BaseEstimator, TransformerMixin):
    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        classes = np.unique(y)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            raise ValueError(
                    "y contains previously unseen labels: %s" % str(diff))
        return np.searchsorted(self.classes_, y)