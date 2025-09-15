import types
import warnings
import sys
import traceback
import pickle
import re
from copy import deepcopy
from functools import partial
from itertools import chain
from inspect import signature
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
import joblib
from . import IS_PYPY
from .. import config_context
from .testing import assert_raises, _get_args
from .testing import assert_raises_regex
from .testing import assert_raise_message
from .testing import assert_array_equal
from .testing import assert_array_almost_equal
from .testing import assert_allclose
from .testing import assert_allclose_dense_sparse
from .testing import assert_warns_message
from .testing import set_random_state
from .testing import SkipTest
from .testing import ignore_warnings
from .testing import assert_dict_equal
from .testing import create_memmap_backed_data
from . import is_scalar_nan
from ..discriminant_analysis import LinearDiscriminantAnalysis
from ..linear_model import Ridge
from ..base import (clone, ClusterMixin, is_classifier, is_regressor,
                    _DEFAULT_TAGS, RegressorMixin, is_outlier_detector,
                    BaseEstimator)
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
from .import deprecated
from .validation import has_fit_parameter, _num_samples
from ..preprocessing import StandardScaler
from ..datasets import (load_iris, load_boston, make_blobs,
                        make_multilabel_classification, make_regression)
import pytest
import pandas as pd

BOSTON = None
CROSS_DECOMPOSITION = ['PLSCanonical', 'PLSRegression', 'CCA', 'PLSSVD']

def _yield_regressor_checks(name, regressor):
    tags = _safe_tags(regressor)
    # TODO: test with intercept
    # TODO: test with multiple responses
    # basic testing
    yield check_regressors_train
    yield partial(check_regressors_train, readonly_memmap=True)
    yield check_regressor_data_not_an_array
    yield check_estimators_partial_fit_n_features
    if tags["multioutput"]:
        yield check_regressor_multioutput
    yield check_regressors_no_decision_function
    if not tags["no_validation"]:
        yield check_supervised_y_2d
    yield check_supervised_y_no_nan
    if name != 'CCA':
        # check that the regressor handles int input
        yield check_regressors_int
    if tags["requires_fit"]:
        yield check_estimators_unfitted
    yield check_non_transformer_estimators_n_iter
