
def _check_positive_coding(method, positive):
    if positive and method in ["omp", "lars"]:
        raise ValueError(
                "Positive constraint not supported for '{}' "
                "coding method.".format(method)
            )



def randomized_svd(M, n_components, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=0):
    if isinstance(M, (sparse.lil_matrix, sparse.dok_matrix)):
        warnings.warn("Calculating SVD of a {} is expensive. "
                      "csr_matrix is more efficient.".format(
                          type(M).__name__),
                      sparse.SparseEfficiencyWarning)

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < .1 * min(M.shape) else 4

    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T

    Q = randomized_range_finder(M, n_random, n_iter,
                                power_iteration_normalizer, random_state)

    # project M to the (k + p) dimensional space using the basis vectors
    B = safe_sparse_dot(Q.T, M)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = linalg.svd(B, full_matrices=False)

    del B
    U = np.dot(Q, Uhat)

    if flip_sign:
        if not transpose:
            U, V = svd_flip(U, V)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, V = svd_flip(U, V, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return V[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], V[:n_components, :]

def gen_batches(n, batch_size, min_batch_size=0):
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)

def _update_dict(dictionary, Y, code, verbose=False, return_r2=False,
                 random_state=None, positive=False):
    n_components = len(code)
    n_features = Y.shape[0]
    random_state = check_random_state(random_state)
    # Get BLAS functions
    gemm, = linalg.get_blas_funcs(('gemm',), (dictionary, code, Y))
    ger, = linalg.get_blas_funcs(('ger',), (dictionary, code))
    nrm2, = linalg.get_blas_funcs(('nrm2',), (dictionary,))
    # Residuals, computed with BLAS for speed and efficiency
    # R <- -1.0 * U * V^T + 1.0 * Y
    # Outputs R as Fortran array for efficiency
    R = gemm(-1.0, dictionary, code, 1.0, Y)
    for k in range(n_components):
        # R <- 1.0 * U_k * V_k^T + R
        R = ger(1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
        dictionary[:, k] = np.dot(R, code[k, :])
        if positive:
            np.clip(dictionary[:, k], 0, None, out=dictionary[:, k])
        # Scale k'th atom
        # (U_k * U_k) ** 0.5
        atom_norm = nrm2(dictionary[:, k])
        if atom_norm < 1e-10:
            if verbose == 1:
                sys.stdout.write("+")
                sys.stdout.flush()
            elif verbose:
                print("Adding new random atom")
            dictionary[:, k] = random_state.randn(n_features)
            if positive:
                np.clip(dictionary[:, k], 0, None, out=dictionary[:, k])
            # Setting corresponding coefs to 0
            code[k, :] = 0.0
            # (U_k * U_k) ** 0.5
            atom_norm = nrm2(dictionary[:, k])
            dictionary[:, k] /= atom_norm
        else:
            dictionary[:, k] /= atom_norm
            # R <- -1.0 * U_k * V_k^T + R
            R = ger(-1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
    if return_r2:
        R = nrm2(R) ** 2.0
        return dictionary, R
    return dictionary