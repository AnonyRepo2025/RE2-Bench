
def check_array(array, accept_sparse=False, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    # accept_sparse 'None' deprecation check
    if accept_sparse is None:
        warnings.warn(
            "Passing 'None' to parameter 'accept_sparse' in methods "
            "check_array and check_X_y is deprecated in version 0.19 "
            "and will be removed in 0.21. Use 'accept_sparse=False' "
            " instead.", DeprecationWarning)
        accept_sparse = False

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, six.string_types) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    if estimator is not None:
        if isinstance(estimator, six.string_types):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse, dtype, copy,
                                      force_all_finite)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = np.asarray(array, dtype=dtype, order=order)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))

        # in the future np.flexible dtypes will be handled like object dtypes
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn(
                "Beginning in version 0.22, arrays of strings will be "
                "interpreted as decimal numbers if parameter 'dtype' is "
                "'numeric'. It is recommended that you convert the array to "
                "type np.float64 before passing it to check_array.",
                FutureWarning)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == 'allow-nan')

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, shape_repr, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    return array



def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})

def inplace_column_scale(X, scale):
    if isinstance(X, sp.csc_matrix):
        inplace_csr_row_scale(X.T, scale)
    elif isinstance(X, sp.csr_matrix):
        inplace_csr_column_scale(X, scale)
    else:
        _raise_typeerror(X)