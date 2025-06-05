def _solve_svd(X, y, alpha):
    U, s, Vt = linalg.svd(X, full_matrices=False)
    idx = s > 1e-15  # same default value as scipy.linalg.pinv
    s_nnz = s[idx][:, np.newaxis]
    UTy = np.dot(U.T, y)
    d = np.zeros((s.size, alpha.size), dtype=X.dtype)
    d[idx] = s_nnz / (s_nnz ** 2 + alpha)
    d_UT_y = d * UTy
    return np.dot(Vt.T, d_UT_y).T

def _solve_sparse_cg(X, y, alpha, max_iter=None, tol=1e-3, verbose=0,
                     X_offset=None, X_scale=None):

    def _get_rescaled_operator(X):

        X_offset_scale = X_offset / X_scale

        def matvec(b):
            return X.dot(b) - b.dot(X_offset_scale)

        def rmatvec(b):
            return X.T.dot(b) - X_offset_scale * np.sum(b)

        X1 = sparse.linalg.LinearOperator(shape=X.shape,
                                          matvec=matvec,
                                          rmatvec=rmatvec)
        return X1

    n_samples, n_features = X.shape

    if X_offset is None or X_scale is None:
        X1 = sp_linalg.aslinearoperator(X)
    else:
        X1 = _get_rescaled_operator(X)

    coefs = np.empty((y.shape[1], n_features), dtype=X.dtype)

    if n_features > n_samples:
        def create_mv(curr_alpha):
            def _mv(x):
                return X1.matvec(X1.rmatvec(x)) + curr_alpha * x
            return _mv
    else:
        def create_mv(curr_alpha):
            def _mv(x):
                return X1.rmatvec(X1.matvec(x)) + curr_alpha * x
            return _mv

    for i in range(y.shape[1]):
        y_column = y[:, i]

        mv = create_mv(alpha[i])
        if n_features > n_samples:
            # kernel ridge
            # w = X.T * inv(X X^t + alpha*Id) y
            C = sp_linalg.LinearOperator(
                (n_samples, n_samples), matvec=mv, dtype=X.dtype)
            # FIXME atol
            try:
                coef, info = sp_linalg.cg(C, y_column, tol=tol, atol='legacy')
            except TypeError:
                # old scipy
                coef, info = sp_linalg.cg(C, y_column, tol=tol)
            coefs[i] = X1.rmatvec(coef)
        else:
            # linear ridge
            # w = inv(X^t X + alpha*Id) * X.T y
            y_column = X1.rmatvec(y_column)
            C = sp_linalg.LinearOperator(
                (n_features, n_features), matvec=mv, dtype=X.dtype)
            # FIXME atol
            try:
                coefs[i], info = sp_linalg.cg(C, y_column, maxiter=max_iter,
                                              tol=tol, atol='legacy')
            except TypeError:
                # old scipy
                coefs[i], info = sp_linalg.cg(C, y_column, maxiter=max_iter,
                                              tol=tol)

        if info < 0:
            raise ValueError("Failed with error code %d" % info)

        if max_iter is None and info > 0 and verbose:
            warnings.warn("sparse_cg did not converge after %d iterations." %
                          info, ConvergenceWarning)

    return coefs
