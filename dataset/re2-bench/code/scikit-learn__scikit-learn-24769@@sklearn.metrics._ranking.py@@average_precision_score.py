import warnings
from functools import partial
from numbers import Real, Integral
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.stats import rankdata
from ..utils import assert_all_finite
from ..utils import check_consistent_length
from ..utils.validation import _check_pos_label_consistency, _check_sample_weight
from ..utils import column_or_1d, check_array
from ..utils.multiclass import type_of_target
from ..utils.extmath import stable_cumsum
from ..utils.sparsefuncs import count_nonzero
from ..utils._param_validation import validate_params, StrOptions, Interval
from ..exceptions import UndefinedMetricWarning
from ..preprocessing import label_binarize
from ..utils._encode import _encode, _unique
from ._base import _average_binary_score, _average_multiclass_ovo_score



def average_precision_score(
    y_true, y_score, *, average="macro", pos_label=1, sample_weight=None
):
    """Compute average precision (AP) from prediction scores.

    AP summarizes a precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous
    threshold used as the weight:

    .. math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n

    where :math:`P_n` and :math:`R_n` are the precision and recall at the nth
    threshold [1]_. This implementation is not interpolated and is different
    from computing the area under the precision-recall curve with the
    trapezoidal rule, which uses linear interpolation and can be too
    optimistic.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        True binary labels or binary label indicators.

    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by :term:`decision_function` on some classifiers).

    average : {'micro', 'samples', 'weighted', 'macro'} or None, \
            default='macro'
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    pos_label : int, float, bool or str, default=1
        The label of the positive class. Only applied to binary ``y_true``.
        For multilabel-indicator ``y_true``, ``pos_label`` is fixed to 1.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    average_precision : float
        Average precision score.

    See Also
    --------
    roc_auc_score : Compute the area under the ROC curve.
    precision_recall_curve : Compute precision-recall pairs for different
        probability thresholds.

    Notes
    -----
    .. versionchanged:: 0.19
      Instead of linearly interpolating between operating points, precisions
      are weighted by the change in recall since the last operating point.

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
           <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
           oldid=793358396#Average_precision>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import average_precision_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> average_precision_score(y_true, y_scores)
    0.83...
    >>> y_true = np.array([0, 0, 1, 1, 2, 2])
    >>> y_scores = np.array([
    ...     [0.7, 0.2, 0.1],
    ...     [0.4, 0.3, 0.3],
    ...     [0.1, 0.8, 0.1],
    ...     [0.2, 0.3, 0.5],
    ...     [0.4, 0.4, 0.2],
    ...     [0.1, 0.2, 0.7],
    ... ])
    >>> average_precision_score(y_true, y_scores)
    0.77...
    """

    def _binary_uninterpolated_average_precision(
        y_true, y_score, pos_label=1, sample_weight=None
    ):
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
        )
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

    y_type = type_of_target(y_true, input_name="y_true")

    # Convert to Python primitive type to avoid NumPy type / Python str
    # comparison. See https://github.com/numpy/numpy/issues/6784
    present_labels = np.unique(y_true).tolist()

    if y_type == "binary":
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError(
                f"pos_label={pos_label} is not a valid label. It should be "
                f"one of {present_labels}"
            )

    elif y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError(
            "Parameter pos_label is fixed to 1 for multilabel-indicator y_true. "
            "Do not set pos_label or set pos_label to 1."
        )

    elif y_type == "multiclass":
        if pos_label != 1:
            raise ValueError(
                "Parameter pos_label is fixed to 1 for multiclass y_true. "
                "Do not set pos_label or set pos_label to 1."
            )
        y_true = label_binarize(y_true, classes=present_labels)

    average_precision = partial(
        _binary_uninterpolated_average_precision, pos_label=pos_label
    )
    return _average_binary_score(
        average_precision, y_true, y_score, average, sample_weight=sample_weight
    )
