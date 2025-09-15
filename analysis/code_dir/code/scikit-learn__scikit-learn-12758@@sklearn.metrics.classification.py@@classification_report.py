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
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    labels_given = True
    if labels is None:
        labels = unique_labels(y_true, y_pred)
        labels_given = False
    else:
        labels = np.asarray(labels)
    micro_is_accuracy = (y_type == 'multiclass' or y_type == 'binary') and (not labels_given or set(labels) == set(unique_labels(y_true, y_pred)))
    if target_names is not None and len(labels) != len(target_names):
        if labels_given:
            warnings.warn('labels size, {0}, does not match size of target_names, {1}'.format(len(labels), len(target_names)))
        else:
            raise ValueError('Number of classes, {0}, does not match size of target_names, {1}. Try specifying the labels parameter'.format(len(labels), len(target_names)))
    if target_names is None:
        target_names = [u'%s' % l for l in labels]
    headers = ['precision', 'recall', 'f1-score', 'support']
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, sample_weight=sample_weight)
    rows = zip(target_names, p, r, f1, s)
    if y_type.startswith('multilabel'):
        average_options = ('micro', 'macro', 'weighted', 'samples')
    else:
        average_options = ('micro', 'macro', 'weighted')
    if output_dict:
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers, [i.item() for i in scores]))
    else:
        longest_last_line_heading = 'weighted avg'
        name_width = max((len(cn) for cn in target_names))
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
        report = head_fmt.format(u'', *headers, width=width)
        report += u'\n\n'
        row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += u'\n'
    for average in average_options:
        if average.startswith('micro') and micro_is_accuracy:
            line_heading = 'accuracy'
        else:
            line_heading = average + ' avg'
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=average, sample_weight=sample_weight)
        avg = [avg_p, avg_r, avg_f1, np.sum(s)]
        if output_dict:
            report_dict[line_heading] = dict(zip(headers, [i.item() for i in avg]))
        elif line_heading == 'accuracy':
            row_fmt_accuracy = u'{:>{width}s} ' + u' {:>9.{digits}}' * 2 + u' {:>9.{digits}f}' + u' {:>9}\n'
            report += row_fmt_accuracy.format(line_heading, '', '', *avg[2:], width=width, digits=digits)
        else:
            report += row_fmt.format(line_heading, *avg, width=width, digits=digits)
    if output_dict:
        if 'accuracy' in report_dict.keys():
            report_dict['accuracy'] = report_dict['accuracy']['precision']
        return report_dict
    else:
        return report