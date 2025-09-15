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

def classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False):
    labels_given = True
    if labels is None:
        labels = unique_labels(y_true, y_pred)
        labels_given = False
    else:
        labels = np.asarray(labels)
    if target_names is not None and len(labels) != len(target_names):
        if labels_given:
            warnings.warn('labels size, {0}, does not match size of target_names, {1}'.format(len(labels), len(target_names)))
        else:
            raise ValueError('Number of classes, {0}, does not match size of target_names, {1}. Try specifying the labels parameter'.format(len(labels), len(target_names)))
    last_line_heading = 'avg / total'
    if target_names is None:
        target_names = [u'%s' % l for l in labels]
    name_width = max((len(cn) for cn in target_names))
    width = max(name_width, len(last_line_heading), digits)
    headers = ['precision', 'recall', 'f1-score', 'support']
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, sample_weight=sample_weight)
    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
    rows = zip(target_names, p, r, f1, s)
    avg_total = [np.average(p, weights=s), np.average(r, weights=s), np.average(f1, weights=s), np.sum(s)]
    if output_dict:
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers, scores))
        report_dict['avg / total'] = dict(zip(headers, avg_total))
        return report_dict
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)
    report += u'\n'
    report += row_fmt.format(last_line_heading, *avg_total, width=width, digits=digits)
    return report