def check_X_y(X, y, accept_sparse=False, accept_large_sparse=True,
              dtype="numeric", order=None, copy=False, force_all_finite=True,
              ensure_2d=True, allow_nd=False, multi_output=False,
              ensure_min_samples=1, ensure_min_features=1, y_numeric=False,
              warn_on_dtype=None, estimator=None):
    if y is None:
        raise ValueError("y cannot be None")

    X = check_array(X, accept_sparse=accept_sparse,
                    accept_large_sparse=accept_large_sparse,
                    dtype=dtype, order=order, copy=copy,
                    force_all_finite=force_all_finite,
                    ensure_2d=ensure_2d, allow_nd=allow_nd,
                    ensure_min_samples=ensure_min_samples,
                    ensure_min_features=ensure_min_features,
                    warn_on_dtype=warn_on_dtype,
                    estimator=estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y

def check_array(array, accept_sparse=False, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, warn_on_dtype=None, estimator=None):
    if warn_on_dtype is not None:
        warnings.warn(
            "'warn_on_dtype' is deprecated in version 0.21 and will be "
            "removed in 0.23. Don't set `warn_on_dtype` to remove this "
            "warning.",
            DeprecationWarning)

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None
    dtypes_orig = None
    if hasattr(array, "dtypes") and hasattr(array.dtypes, '__array__'):
        dtypes_orig = np.array(array.dtypes)

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            dtype = None
        else:
            dtype = dtype[0]

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    if estimator is not None:
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                      dtype=dtype, copy=copy,
                                      force_all_finite=force_all_finite,
                                      accept_large_sparse=accept_large_sparse)
    else:
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = np.asarray(array, dtype=dtype, order=order)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

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
                "Beginning in version 0.22, arrays of bytes/strings will be "
                "converted to decimal numbers if dtype='numeric'. "
                "It is recommended that you convert the array to "
                "a float dtype before using it in scikit-learn, "
                "for example by using "
                "your_array = your_array.astype(np.float64).",
                FutureWarning)

        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == 'allow-nan')

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, array.shape, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, array.shape, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    if (warn_on_dtype and dtypes_orig is not None and
            {array.dtype} != set(dtypes_orig)):
        msg = ("Data with input dtype %s were all converted to %s%s."
               % (', '.join(map(str, sorted(set(dtypes_orig)))), array.dtype,
                  context))
        warnings.warn(msg, DataConversionWarning, stacklevel=3)

    return array