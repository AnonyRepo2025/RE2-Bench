from numbers import Integral, Real
import warnings
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.special import xlogy
from ..preprocessing import LabelBinarizer
from ..preprocessing import LabelEncoder
from ..utils import assert_all_finite
from ..utils import check_array
from ..utils import check_consistent_length
from ..utils import column_or_1d
from ..utils.multiclass import unique_labels
from ..utils.multiclass import type_of_target
from ..utils.validation import _num_samples
from ..utils.sparsefuncs import count_nonzero
from ..utils._param_validation import StrOptions, Options, Interval, validate_params
from ..exceptions import UndefinedMetricWarning
from ._base import _check_pos_label_consistency

def log_loss(y_true, y_pred, *, eps='auto', normalize=True, sample_weight=None, labels=None):
    y_pred = check_array(y_pred, ensure_2d=False, dtype=[np.float64, np.float32, np.float16])
    if eps == 'auto':
        eps = np.finfo(y_pred.dtype).eps
    else:
        warnings.warn('Setting the eps parameter is deprecated and will be removed in 1.5. Instead eps will always havea default value of `np.finfo(y_pred.dtype).eps`.', FutureWarning)
    check_consistent_length(y_pred, y_true, sample_weight)
    lb = LabelBinarizer()
    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_true)
    if len(lb.classes_) == 1:
        if labels is None:
            raise ValueError('y_true contains only one label ({0}). Please provide the true labels explicitly through the labels argument.'.format(lb.classes_[0]))
        else:
            raise ValueError('The labels array needs to contain at least two labels for log_loss, got {0}.'.format(lb.classes_))
    transformed_labels = lb.transform(y_true)
    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(1 - transformed_labels, transformed_labels, axis=1)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)
    transformed_labels = check_array(transformed_labels)
    if len(lb.classes_) != y_pred.shape[1]:
        if labels is None:
            raise ValueError('y_true and y_pred contain different number of classes {0}, {1}. Please provide the true labels explicitly through the labels argument. Classes found in y_true: {2}'.format(transformed_labels.shape[1], y_pred.shape[1], lb.classes_))
        else:
            raise ValueError('The number of classes in labels is different from that in y_pred. Classes found in labels: {0}'.format(lb.classes_))
    y_pred_sum = y_pred.sum(axis=1)
    if not np.isclose(y_pred_sum, 1, rtol=1e-15, atol=5 * eps).all():
        warnings.warn('The y_pred values do not sum to one. Starting from 1.5 thiswill result in an error.', UserWarning)
    y_pred = y_pred / y_pred_sum[:, np.newaxis]
    loss = -xlogy(transformed_labels, y_pred).sum(axis=1)
    return _weighted_sum(loss, sample_weight, normalize)