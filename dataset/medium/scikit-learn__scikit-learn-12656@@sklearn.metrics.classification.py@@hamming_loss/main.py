from __future__ import division
import warnings
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from ..preprocessing import LabelBinarizer, label_binarize
from ..preprocessing import LabelEncoder
from ..utils import assert_all_finite
from ..utils import check_array
from ..utils import check_consistent_length
from ..utils import column_or_1d
from ..utils.multiclass import unique_labels
from ..utils.multiclass import type_of_target
from ..utils.validation import _num_samples
from ..utils.sparsefuncs import count_nonzero
from ..exceptions import UndefinedMetricWarning



def hamming_loss(y_true, y_pred, labels=None, sample_weight=None):
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred, sample_weight)

    if labels is not None:
        warnings.warn("The labels parameter is unused. It was"
                      " deprecated in version 0.21 and"
                      " will be removed in version 0.23",
                      DeprecationWarning)

    if sample_weight is None:
        weight_average = 1.
    else:
        weight_average = np.mean(sample_weight)

    if y_type.startswith('multilabel'):
        n_differences = count_nonzero(y_true - y_pred,
                                      sample_weight=sample_weight)
        return (n_differences /
                (y_true.shape[0] * y_true.shape[1] * weight_average))

    elif y_type in ["binary", "multiclass"]:
        return _weighted_sum(y_true != y_pred, sample_weight, normalize=True)
    else:
        raise ValueError("{0} is not supported".format(y_type))
