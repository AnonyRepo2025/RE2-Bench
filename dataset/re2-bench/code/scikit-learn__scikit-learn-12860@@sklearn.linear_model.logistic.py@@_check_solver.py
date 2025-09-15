import numbers
import warnings
import numpy as np
from scipy import optimize, sparse
from scipy.special import expit
from .base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
from .sag import sag_solver
from ..preprocessing import LabelEncoder, LabelBinarizer
from ..svm.base import _fit_liblinear
from ..utils import check_array, check_consistent_length, compute_class_weight
from ..utils import check_random_state
from ..utils.extmath import (log_logistic, safe_sparse_dot, softmax,
                             squared_norm)
from ..utils.extmath import row_norms
from ..utils.fixes import logsumexp
from ..utils.optimize import newton_cg
from ..utils.validation import check_X_y
from ..utils import deprecated
from ..exceptions import (NotFittedError, ConvergenceWarning,
                          ChangedBehaviorWarning)
from ..utils.multiclass import check_classification_targets
from ..utils._joblib import Parallel, delayed, effective_n_jobs
from ..utils.fixes import _joblib_parallel_args
from ..model_selection import check_cv
from ..metrics import get_scorer



def _check_solver(solver, penalty, dual):
    if solver == 'warn':
        solver = 'liblinear'
        warnings.warn("Default solver will be changed to 'lbfgs' in 0.22. "
                      "Specify a solver to silence this warning.",
                      FutureWarning)

    all_solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    if solver not in all_solvers:
        raise ValueError("Logistic Regression supports only solvers in %s, got"
                         " %s." % (all_solvers, solver))

    all_penalties = ['l1', 'l2', 'elasticnet', 'none']
    if penalty not in all_penalties:
        raise ValueError("Logistic Regression supports only penalties in %s,"
                         " got %s." % (all_penalties, penalty))

    if solver not in ['liblinear', 'saga'] and penalty not in ('l2', 'none'):
        raise ValueError("Solver %s supports only 'l2' or 'none' penalties, "
                         "got %s penalty." % (solver, penalty))
    if solver != 'liblinear' and dual:
        raise ValueError("Solver %s supports only "
                         "dual=False, got dual=%s" % (solver, dual))

    if penalty == 'elasticnet' and solver != 'saga':
        raise ValueError("Only 'saga' solver supports elasticnet penalty,"
                         " got solver={}.".format(solver))

    if solver == 'liblinear' and penalty == 'none':
        raise ValueError(
            "penalty='none' is not supported for the liblinear solver"
        )

    return solver
