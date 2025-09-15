from io import StringIO
from numbers import Integral
import numpy as np
from ..utils.validation import check_is_fitted, check_array
from ..utils._param_validation import Interval, validate_params, StrOptions
from ..base import is_classifier
from . import _criterion
from . import _tree
from ._reingold_tilford import buchheim, Tree
from . import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import matplotlib.pyplot as plt

SENTINEL = Sentinel()

def export_text(
    decision_tree,
    *,
    feature_names=None,
    class_names=None,
    max_depth=10,
    spacing=3,
    decimals=2,
    show_weights=False,
):
    """Build a text report showing the rules of a decision tree.

    Note that backwards compatibility may not be supported.

    Parameters
    ----------
    decision_tree : object
        The decision tree estimator to be exported.
        It can be an instance of
        DecisionTreeClassifier or DecisionTreeRegressor.

    feature_names : array-like of shape (n_features,), default=None
        An array containing the feature names.
        If None generic names will be used ("feature_0", "feature_1", ...).

    class_names : array-like of shape (n_classes,), default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.

        - if `None`, the class names are delegated to `decision_tree.classes_`;
        - otherwise, `class_names` will be used as class names instead of
          `decision_tree.classes_`. The length of `class_names` must match
          the length of `decision_tree.classes_`.

        .. versionadded:: 1.3

    max_depth : int, default=10
        Only the first max_depth levels of the tree are exported.
        Truncated branches will be marked with "...".

    spacing : int, default=3
        Number of spaces between edges. The higher it is, the wider the result.

    decimals : int, default=2
        Number of decimal digits to display.

    show_weights : bool, default=False
        If true the classification weights will be exported on each leaf.
        The classification weights are the number of samples each class.

    Returns
    -------
    report : str
        Text summary of all the rules in the decision tree.

    Examples
    --------

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.tree import export_text
    >>> iris = load_iris()
    >>> X = iris['data']
    >>> y = iris['target']
    >>> decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    >>> decision_tree = decision_tree.fit(X, y)
    >>> r = export_text(decision_tree, feature_names=iris['feature_names'])
    >>> print(r)
    |--- petal width (cm) <= 0.80
    |   |--- class: 0
    |--- petal width (cm) >  0.80
    |   |--- petal width (cm) <= 1.75
    |   |   |--- class: 1
    |   |--- petal width (cm) >  1.75
    |   |   |--- class: 2
    """
    if feature_names is not None:
        feature_names = check_array(
            feature_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )
    if class_names is not None:
        class_names = check_array(
            class_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )

    check_is_fitted(decision_tree)
    tree_ = decision_tree.tree_
    if is_classifier(decision_tree):
        if class_names is None:
            class_names = decision_tree.classes_
        elif len(class_names) != len(decision_tree.classes_):
            raise ValueError(
                "When `class_names` is an array, it should contain as"
                " many items as `decision_tree.classes_`. Got"
                f" {len(class_names)} while the tree was fitted with"
                f" {len(decision_tree.classes_)} classes."
            )
    right_child_fmt = "{} {} <= {}\n"
    left_child_fmt = "{} {} >  {}\n"
    truncation_fmt = "{} {}\n"

    if feature_names is not None and len(feature_names) != tree_.n_features:
        raise ValueError(
            "feature_names must contain %d elements, got %d"
            % (tree_.n_features, len(feature_names))
        )

    if isinstance(decision_tree, DecisionTreeClassifier):
        value_fmt = "{}{} weights: {}\n"
        if not show_weights:
            value_fmt = "{}{}{}\n"
    else:
        value_fmt = "{}{} value: {}\n"

    if feature_names is not None:
        feature_names_ = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else None
            for i in tree_.feature
        ]
    else:
        feature_names_ = ["feature_{}".format(i) for i in tree_.feature]

    export_text.report = ""

    def _add_leaf(value, class_name, indent):
        val = ""
        is_classification = isinstance(decision_tree, DecisionTreeClassifier)
        if show_weights or not is_classification:
            val = ["{1:.{0}f}, ".format(decimals, v) for v in value]
            val = "[" + "".join(val)[:-2] + "]"
        if is_classification:
            val += " class: " + str(class_name)
        export_text.report += value_fmt.format(indent, "", val)

    def print_tree_recurse(node, depth):
        indent = ("|" + (" " * spacing)) * depth
        indent = indent[:-spacing] + "-" * spacing

        value = None
        if tree_.n_outputs == 1:
            value = tree_.value[node][0]
        else:
            value = tree_.value[node].T[0]
        class_name = np.argmax(value)

        if tree_.n_classes[0] != 1 and tree_.n_outputs == 1:
            class_name = class_names[class_name]

        if depth <= max_depth + 1:
            info_fmt = ""
            info_fmt_left = info_fmt
            info_fmt_right = info_fmt

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names_[node]
                threshold = tree_.threshold[node]
                threshold = "{1:.{0}f}".format(decimals, threshold)
                export_text.report += right_child_fmt.format(indent, name, threshold)
                export_text.report += info_fmt_left
                print_tree_recurse(tree_.children_left[node], depth + 1)

                export_text.report += left_child_fmt.format(indent, name, threshold)
                export_text.report += info_fmt_right
                print_tree_recurse(tree_.children_right[node], depth + 1)
            else:  # leaf
                _add_leaf(value, class_name, indent)
        else:
            subtree_depth = _compute_depth(tree_, node)
            if subtree_depth == 1:
                _add_leaf(value, class_name, indent)
            else:
                trunc_report = "truncated branch of depth %d" % subtree_depth
                export_text.report += truncation_fmt.format(indent, trunc_report)

    print_tree_recurse(0, 1)
    return export_text.report
