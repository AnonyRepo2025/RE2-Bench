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



def log_loss(
    y_true, y_pred, *, eps="auto", normalize=True, sample_weight=None, labels=None
):
    r"""Log loss, aka logistic loss or cross-entropy loss.

    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of a logistic model that returns ``y_pred`` probabilities
    for its training data ``y_true``.
    The log loss is only defined for two or more labels.
    For a single sample with true label :math:`y \in \{0,1\}` and
    a probability estimate :math:`p = \operatorname{Pr}(y = 1)`, the log
    loss is:

    .. math::
        L_{\log}(y, p) = -(y \log (p) + (1 - y) \log (1 - p))

    Read more in the :ref:`User Guide <log_loss>`.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.

    y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method. If ``y_pred.shape = (n_samples,)``
        the probabilities provided are assumed to be that of the
        positive class. The labels in ``y_pred`` are assumed to be
        ordered alphabetically, as done by
        :class:`preprocessing.LabelBinarizer`.

    eps : float or "auto", default="auto"
        Log loss is undefined for p=0 or p=1, so probabilities are
        clipped to `max(eps, min(1 - eps, p))`. The default will depend on the
        data type of `y_pred` and is set to `np.finfo(y_pred.dtype).eps`.

        .. versionadded:: 1.2

        .. versionchanged:: 1.2
           The default value changed from `1e-15` to `"auto"` that is
           equivalent to `np.finfo(y_pred.dtype).eps`.

        .. deprecated:: 1.3
           `eps` is deprecated in 1.3 and will be removed in 1.5.

    normalize : bool, default=True
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    labels : array-like, default=None
        If not provided, labels will be inferred from y_true. If ``labels``
        is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
        assumed to be binary and are inferred from ``y_true``.

        .. versionadded:: 0.18

    Returns
    -------
    loss : float
        Log loss, aka logistic loss or cross-entropy loss.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).

    References
    ----------
    C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
    p. 209.

    Examples
    --------
    >>> from sklearn.metrics import log_loss
    >>> log_loss(["spam", "ham", "ham", "spam"],
    ...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
    0.21616...
    """
    y_pred = check_array(
        y_pred, ensure_2d=False, dtype=[np.float64, np.float32, np.float16]
    )
    if eps == "auto":
        eps = np.finfo(y_pred.dtype).eps
    else:
        # TODO: Remove user defined eps in 1.5
        warnings.warn(
            "Setting the eps parameter is deprecated and will "
            "be removed in 1.5. Instead eps will always have"
            "a default value of `np.finfo(y_pred.dtype).eps`.",
            FutureWarning,
        )

    check_consistent_length(y_pred, y_true, sample_weight)
    lb = LabelBinarizer()

    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_true)

    if len(lb.classes_) == 1:
        if labels is None:
            raise ValueError(
                "y_true contains only one label ({0}). Please "
                "provide the true labels explicitly through the "
                "labels argument.".format(lb.classes_[0])
            )
        else:
            raise ValueError(
                "The labels array needs to contain at least two "
                "labels for log_loss, "
                "got {0}.".format(lb.classes_)
            )

    transformed_labels = lb.transform(y_true)

    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(
            1 - transformed_labels, transformed_labels, axis=1
        )

    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # Check if dimensions are consistent.
    transformed_labels = check_array(transformed_labels)
    if len(lb.classes_) != y_pred.shape[1]:
        if labels is None:
            raise ValueError(
                "y_true and y_pred contain different number of "
                "classes {0}, {1}. Please provide the true "
                "labels explicitly through the labels argument. "
                "Classes found in "
                "y_true: {2}".format(
                    transformed_labels.shape[1], y_pred.shape[1], lb.classes_
                )
            )
        else:
            raise ValueError(
                "The number of classes in labels is different "
                "from that in y_pred. Classes found in "
                "labels: {0}".format(lb.classes_)
            )

    # Renormalize
    y_pred_sum = y_pred.sum(axis=1)
    if not np.isclose(y_pred_sum, 1, rtol=1e-15, atol=5 * eps).all():
        warnings.warn(
            "The y_pred values do not sum to one. Starting from 1.5 this"
            "will result in an error.",
            UserWarning,
        )
    y_pred = y_pred / y_pred_sum[:, np.newaxis]
    loss = -xlogy(transformed_labels, y_pred).sum(axis=1)

    return _weighted_sum(loss, sample_weight, normalize)
