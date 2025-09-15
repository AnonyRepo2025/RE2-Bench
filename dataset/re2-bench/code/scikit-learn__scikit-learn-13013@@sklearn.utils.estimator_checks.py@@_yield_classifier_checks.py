import types
import warnings
import sys
import traceback
import pickle
from copy import deepcopy
from functools import partial
from inspect import signature
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from . import IS_PYPY
from . import _joblib
from .testing import assert_raises, _get_args
from .testing import assert_raises_regex
from .testing import assert_raise_message
from .testing import assert_equal
from .testing import assert_not_equal
from .testing import assert_in
from .testing import assert_array_equal
from .testing import assert_array_almost_equal
from .testing import assert_allclose
from .testing import assert_allclose_dense_sparse
from .testing import assert_warns_message
from .testing import set_random_state
from .testing import assert_greater
from .testing import assert_greater_equal
from .testing import SkipTest
from .testing import ignore_warnings
from .testing import assert_dict_equal
from .testing import create_memmap_backed_data
from . import is_scalar_nan
from ..discriminant_analysis import LinearDiscriminantAnalysis
from ..linear_model import Ridge
from ..base import (clone, ClusterMixin, is_classifier, is_regressor,
                    _DEFAULT_TAGS, RegressorMixin, is_outlier_detector)
from ..metrics import accuracy_score, adjusted_rand_score, f1_score
from ..random_projection import BaseRandomProjection
from ..feature_selection import SelectKBest
from ..pipeline import make_pipeline
from ..exceptions import DataConversionWarning
from ..exceptions import NotFittedError
from ..exceptions import SkipTestWarning
from ..model_selection import train_test_split
from ..model_selection import ShuffleSplit
from ..model_selection._validation import _safe_split
from ..metrics.pairwise import (rbf_kernel, linear_kernel, pairwise_distances)
from .import shuffle
from .validation import has_fit_parameter, _num_samples
from ..preprocessing import StandardScaler
from ..datasets import load_iris, load_boston, make_blobs
import pandas as pd

BOSTON = None
CROSS_DECOMPOSITION = ['PLSCanonical', 'PLSRegression', 'CCA', 'PLSSVD']

def _yield_classifier_checks(name, classifier):
    tags = _safe_tags(classifier)

    # test classifiers can handle non-array data
    yield check_classifier_data_not_an_array
    # test classifiers trained on a single label always return this label
    yield check_classifiers_one_label
    yield check_classifiers_classes
    yield check_estimators_partial_fit_n_features
    # basic consistency testing
    yield check_classifiers_train
    yield partial(check_classifiers_train, readonly_memmap=True)
    yield check_classifiers_regression_target
    if not tags["no_validation"]:
        yield check_supervised_y_no_nan
        yield check_supervised_y_2d
    if tags["requires_fit"]:
        yield check_estimators_unfitted
    if 'class_weight' in classifier.get_params().keys():
        yield check_class_weight_classifiers

    yield check_non_transformer_estimators_n_iter
    # test if predict_proba is a monotonic transformation of decision_function
    yield check_decision_proba_consistency
