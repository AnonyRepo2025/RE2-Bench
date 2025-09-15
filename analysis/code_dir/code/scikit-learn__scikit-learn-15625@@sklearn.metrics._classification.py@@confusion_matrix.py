import warnings
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
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
from ..exceptions import UndefinedMetricWarning

def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if y_type not in ('binary', 'multiclass'):
        raise ValueError('%s is not supported' % y_type)
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)
        if np.all([l not in y_true for l in labels]):
            raise ValueError('At least one label specified must be in y_true')
    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)
    check_consistent_length(y_true, y_pred, sample_weight)
    n_labels = labels.size
    label_to_ind = {y: x for x, y in enumerate(labels)}
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    sample_weight = sample_weight[ind]
    if sample_weight.dtype.kind in {'i', 'u', 'b'}:
        dtype = np.int64
    else:
        dtype = np.float64
    cm = coo_matrix((sample_weight, (y_true, y_pred)), shape=(n_labels, n_labels), dtype=dtype).toarray()
    with np.errstate(all='ignore'):
        if normalize == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)
    return cm