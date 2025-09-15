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



def brier_score_loss(y_true, y_prob, sample_weight=None, pos_label=None):
    """Compute the Brier score.
    The smaller the Brier score, the better, hence the naming with "loss".
    Across all items in a set N predictions, the Brier score measures the
    mean squared difference between (1) the predicted probability assigned
    to the possible outcomes for item i, and (2) the actual outcome.
    Therefore, the lower the Brier score is for a set of predictions, the
    better the predictions are calibrated. Note that the Brier score always
    takes on a value between zero and one, since this is the largest
    possible difference between a predicted probability (which must be
    between zero and one) and the actual outcome (which can take on values
    of only 0 and 1). The Brier loss is composed of refinement loss and
    calibration loss.
    The Brier score is appropriate for binary and categorical outcomes that
    can be structured as true or false, but is inappropriate for ordinal
    variables which can take on three or more values (this is because the
    Brier score assumes that all possible outcomes are equivalently
    "distant" from one another). Which label is considered to be the positive
    label is controlled via the parameter pos_label, which defaults to 1.
    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.

    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    pos_label : int or str, default=None
        Label of the positive class.
        Defaults to the greater label unless y_true is all 0 or all -1
        in which case pos_label defaults to 1.

    Returns
    -------
    score : float
        Brier score

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import brier_score_loss
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_true_categorical = np.array(["spam", "ham", "ham", "spam"])
    >>> y_prob = np.array([0.1, 0.9, 0.8, 0.3])
    >>> brier_score_loss(y_true, y_prob)  # doctest: +ELLIPSIS
    0.037...
    >>> brier_score_loss(y_true, 1-y_prob, pos_label=0)  # doctest: +ELLIPSIS
    0.037...
    >>> brier_score_loss(y_true_categorical, y_prob, \
                         pos_label="ham")  # doctest: +ELLIPSIS
    0.037...
    >>> brier_score_loss(y_true, np.array(y_prob) > 0.5)
    0.0

    References
    ----------
    .. [1] `Wikipedia entry for the Brier score.
            <https://en.wikipedia.org/wiki/Brier_score>`_
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    assert_all_finite(y_true)
    assert_all_finite(y_prob)
    check_consistent_length(y_true, y_prob, sample_weight)

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. "
                         "Labels in y_true: %s." % labels)
    if y_prob.max() > 1:
        raise ValueError("y_prob contains values greater than 1.")
    if y_prob.min() < 0:
        raise ValueError("y_prob contains values less than 0.")

    # if pos_label=None, when y_true is in {-1, 1} or {0, 1},
    # pos_labe is set to 1 (consistent with precision_recall_curve/roc_curve),
    # otherwise pos_label is set to the greater label
    # (different from precision_recall_curve/roc_curve,
    # the purpose is to keep backward compatibility).
    if pos_label is None:
        if (np.array_equal(labels, [0]) or
                np.array_equal(labels, [-1])):
            pos_label = 1
        else:
            pos_label = y_true.max()
    y_true = np.array(y_true == pos_label, int)
    return np.average((y_true - y_prob) ** 2, weights=sample_weight)
