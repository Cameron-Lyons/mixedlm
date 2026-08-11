from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

if TYPE_CHECKING:
    from mixedlm.matrices.design import RandomEffectStructure
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


_CONDVAR_BATCH_COLUMNS = 64


def _dense_condvar_solver(
    precision: NDArray[np.floating],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    try:
        chol = linalg.cho_factor(precision, lower=True)

        def solve(rhs: NDArray[np.floating]) -> NDArray[np.floating]:
            return linalg.cho_solve(chol, rhs)

    except linalg.LinAlgError:

        def solve(rhs: NDArray[np.floating]) -> NDArray[np.floating]:
            return linalg.lstsq(precision, rhs)[0]

    return solve


def _conditional_variance_blocks(
    precision: sparse.spmatrix | NDArray[np.floating],
    lambda_factor: sparse.spmatrix | NDArray[np.floating],
    structures: list[RandomEffectStructure],
    *,
    scale: float = 1.0,
    include_cov: bool = False,
) -> dict[str, dict[str, NDArray[np.floating]]]:
    """Extract per-level conditional covariance blocks without a dense q-by-q inverse."""
    q = precision.shape[0]
    if q == 0:
        return {}

    precision_csc = sparse.csc_matrix(precision)
    lambda_csc = sparse.csc_matrix(lambda_factor)

    solver: Callable[[NDArray[np.floating]], NDArray[np.floating]]
    try:
        factor = sparse_linalg.splu(precision_csc)
        solver = factor.solve
    except RuntimeError:
        precision_dense = precision_csc.toarray()
        solver = _dense_condvar_solver(precision_dense)

    grouped_structures: dict[str, list[tuple[RandomEffectStructure, int]]] = {}
    offset = 0
    for struct in structures:
        grouped_structures.setdefault(struct.grouping_factor, []).append((struct, offset))
        offset += struct.n_levels * struct.n_terms

    result: dict[str, dict[str, NDArray[np.floating]]] = {}
    for group_name, group_structures in grouped_structures.items():
        n_levels = group_structures[0][0].n_levels
        if any(struct.n_levels != n_levels for struct, _ in group_structures):
            raise ValueError(f"Random-effect structures for '{group_name}' have unequal levels")

        term_names = list(
            dict.fromkeys(
                term_name
                for struct, _ in group_structures
                for term_name in struct.term_names
            )
        )
        raw_term_names = [
            term_name
            for struct, _ in group_structures
            for term_name in struct.term_names
        ]
        n_terms = len(term_names)
        n_raw_terms = len(raw_term_names)
        aggregation = np.zeros((n_terms, n_raw_terms), dtype=np.float64)
        term_positions = {name: index for index, name in enumerate(term_names)}
        for raw_index, term_name in enumerate(raw_term_names):
            aggregation[term_positions[term_name], raw_index] = 1.0

        term_variances = np.empty((n_levels, n_terms), dtype=np.float64)
        covariance_blocks = (
            np.empty((n_levels, n_terms, n_terms), dtype=np.float64)
            if include_cov and n_terms > 1
            else None
        )
        levels_per_batch = max(1, _CONDVAR_BATCH_COLUMNS // n_raw_terms)

        for first_level in range(0, n_levels, levels_per_batch):
            last_level = min(first_level + levels_per_batch, n_levels)
            indices = np.asarray(
                [
                    struct_offset + level * struct.n_terms + term_index
                    for level in range(first_level, last_level)
                    for struct, struct_offset in group_structures
                    for term_index in range(struct.n_terms)
                ],
                dtype=np.intp,
            )
            lambda_rows = lambda_csc[indices, :]
            rhs = lambda_rows.T.toarray()
            solved = np.asarray(solver(rhs))
            batch_cov = np.asarray(lambda_rows @ solved) * scale

            for level in range(first_level, last_level):
                local_level = level - first_level
                block_start = local_level * n_raw_terms
                block_end = block_start + n_raw_terms
                raw_block = batch_cov[block_start:block_end, block_start:block_end]
                block = aggregation @ raw_block @ aggregation.T
                block = (block + block.T) * 0.5
                term_variances[level] = np.diag(block)
                if covariance_blocks is not None:
                    covariance_blocks[level] = block

        group_result: dict[str, NDArray[np.floating]] = {}
        for term_index, term_name in enumerate(term_names):
            group_result[term_name] = term_variances[:, term_index]
        if covariance_blocks is not None:
            group_result["_cov"] = covariance_blocks
        result[group_name] = group_result

    return result


def sdcor2cov(
    sd: NDArray[np.floating],
    corr: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Convert standard deviations and correlations to a covariance matrix.

    Parameters
    ----------
    sd : ndarray
        Vector of standard deviations (length q).
    corr : ndarray, optional
        Correlation matrix (q x q). If None, assumes identity (uncorrelated).

    Returns
    -------
    ndarray
        Covariance matrix (q x q).

    Examples
    --------
    >>> sd = np.array([1.0, 2.0])
    >>> corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> sdcor2cov(sd, corr)
    array([[1. , 1. ],
           [1. , 4. ]])
    """
    sd = np.asarray(sd)
    q = len(sd)

    if corr is None:
        return np.diag(sd**2)

    corr = np.asarray(corr)
    if corr.shape != (q, q):
        raise ValueError(f"corr must be {q}x{q}, got {corr.shape}")

    D = np.diag(sd)
    return D @ corr @ D


def cov2sdcor(
    cov: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Convert a covariance matrix to standard deviations and correlations.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix (q x q).

    Returns
    -------
    sd : ndarray
        Vector of standard deviations (length q).
    corr : ndarray
        Correlation matrix (q x q).

    Examples
    --------
    >>> cov = np.array([[1.0, 1.0], [1.0, 4.0]])
    >>> sd, corr = cov2sdcor(cov)
    >>> sd
    array([1., 2.])
    >>> corr
    array([[1. , 0.5],
           [0.5, 1. ]])
    """
    cov = np.asarray(cov)
    sd = np.sqrt(np.diag(cov))

    with np.errstate(divide="ignore", invalid="ignore"):
        D_inv = np.diag(1.0 / sd)
        corr = D_inv @ cov @ D_inv

    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    return sd, corr


def Vv_to_Cv(
    vv: NDArray[np.floating],
    q: int,
    sigma: float = 1.0,
) -> NDArray[np.floating]:
    """Convert variance-covariance vector (theta) to covariance vector.

    The variance vector uses the relative covariance factor parameterization
    (lower triangular Cholesky factor scaled by sigma).

    Parameters
    ----------
    vv : ndarray
        Variance vector (theta parameters) of length q*(q+1)/2.
    q : int
        Dimension of the covariance matrix.
    sigma : float, default 1.0
        Residual standard deviation for scaling.

    Returns
    -------
    ndarray
        Covariance vector (lower triangular elements of cov matrix).

    Examples
    --------
    >>> theta = np.array([1.0, 0.0, 1.0])  # 2x2 identity Cholesky
    >>> Vv_to_Cv(theta, q=2, sigma=1.0)
    array([1., 0., 1.])
    """
    vv = np.asarray(vv)
    n_theta = q * (q + 1) // 2
    if len(vv) != n_theta:
        raise ValueError(f"Expected {n_theta} theta values for q={q}, got {len(vv)}")

    L = np.zeros((q, q), dtype=np.float64)
    row_indices, col_indices = np.tril_indices(q)
    L[row_indices, col_indices] = vv

    cov = L @ L.T * sigma**2

    return cov[row_indices, col_indices]


def Cv_to_Vv(
    cv: NDArray[np.floating],
    q: int,
    sigma: float = 1.0,
) -> NDArray[np.floating]:
    """Convert covariance vector to variance vector (theta).

    This is the inverse of Vv_to_Cv.

    Parameters
    ----------
    cv : ndarray
        Covariance vector (lower triangular elements) of length q*(q+1)/2.
    q : int
        Dimension of the covariance matrix.
    sigma : float, default 1.0
        Residual standard deviation for scaling.

    Returns
    -------
    ndarray
        Variance vector (theta parameters).

    Examples
    --------
    >>> cv = np.array([1.0, 0.0, 1.0])  # diagonal covariance
    >>> Cv_to_Vv(cv, q=2, sigma=1.0)
    array([1., 0., 1.])
    """
    cv = np.asarray(cv)
    n_theta = q * (q + 1) // 2
    if len(cv) != n_theta:
        raise ValueError(f"Expected {n_theta} cov values for q={q}, got {len(cv)}")

    cov = np.zeros((q, q), dtype=np.float64)
    row_indices, col_indices = np.tril_indices(q)
    cov[row_indices, col_indices] = cv
    diag = np.diag(cov).copy()
    cov += cov.T
    cov[np.diag_indices(q)] = diag

    cov_scaled = cov / sigma**2

    try:
        L = linalg.cholesky(cov_scaled, lower=True)
    except linalg.LinAlgError:
        eigvals, eigvecs = linalg.eigh(cov_scaled)
        eigvals = np.maximum(eigvals, 1e-10)
        cov_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = linalg.cholesky(cov_fixed, lower=True)

    return L[row_indices, col_indices]


def Sv_to_Cv(
    sv: NDArray[np.floating],
    q: int,
) -> NDArray[np.floating]:
    """Convert SD-correlation vector to covariance vector.

    The SD-correlation vector contains standard deviations followed by
    unique correlation values (lower triangular, excluding diagonal).

    Parameters
    ----------
    sv : ndarray
        SD-correlation vector: [sd1, sd2, ..., sdq, corr21, corr31, corr32, ...].
        Length is q + q*(q-1)/2.
    q : int
        Dimension of the covariance matrix.

    Returns
    -------
    ndarray
        Covariance vector (lower triangular elements).

    Examples
    --------
    >>> sv = np.array([1.0, 2.0, 0.5])  # sd1=1, sd2=2, corr=0.5
    >>> Sv_to_Cv(sv, q=2)
    array([1., 1., 4.])
    """
    sv = np.asarray(sv)
    expected_len = q + q * (q - 1) // 2
    if len(sv) != expected_len:
        raise ValueError(f"Expected {expected_len} values for q={q}, got {len(sv)}")

    sd = sv[:q]
    corr_vals = sv[q:]

    corr = np.eye(q)
    idx = 0
    for i in range(1, q):
        for j in range(i):
            corr[i, j] = corr_vals[idx]
            corr[j, i] = corr_vals[idx]
            idx += 1

    cov = sdcor2cov(sd, corr)

    row_indices, col_indices = np.tril_indices(q)
    return cov[row_indices, col_indices]


def Cv_to_Sv(
    cv: NDArray[np.floating],
    q: int,
) -> NDArray[np.floating]:
    """Convert covariance vector to SD-correlation vector.

    This is the inverse of Sv_to_Cv.

    Parameters
    ----------
    cv : ndarray
        Covariance vector (lower triangular elements).
    q : int
        Dimension of the covariance matrix.

    Returns
    -------
    ndarray
        SD-correlation vector: [sd1, sd2, ..., sdq, corr21, corr31, ...].

    Examples
    --------
    >>> cv = np.array([1.0, 1.0, 4.0])
    >>> Cv_to_Sv(cv, q=2)
    array([1. , 2. , 0.5])
    """
    cv = np.asarray(cv)
    n_cov = q * (q + 1) // 2
    if len(cv) != n_cov:
        raise ValueError(f"Expected {n_cov} cov values for q={q}, got {len(cv)}")

    cov = np.zeros((q, q), dtype=np.float64)
    row_indices, col_indices = np.tril_indices(q)
    cov[row_indices, col_indices] = cv
    diag = np.diag(cov).copy()
    cov += cov.T
    cov[np.diag_indices(q)] = diag

    sd, corr = cov2sdcor(cov)

    corr_vals = []
    for i in range(1, q):
        for j in range(i):
            corr_vals.append(corr[i, j])

    return np.concatenate([sd, np.array(corr_vals)])


def mlist2vec(mlist: list[NDArray[np.floating]]) -> NDArray[np.floating]:
    """Convert a list of matrices to a single vector.

    Each matrix is vectorized using column-major (Fortran) order,
    and all vectors are concatenated.

    Parameters
    ----------
    mlist : list of ndarray
        List of matrices.

    Returns
    -------
    ndarray
        Concatenated vector of all matrix elements.

    Examples
    --------
    >>> mlist = [np.array([[1, 2], [3, 4]]), np.array([[5]])]
    >>> mlist2vec(mlist)
    array([1, 3, 2, 4, 5])
    """
    vecs = [m.flatten(order="F") for m in mlist]
    return np.concatenate(vecs) if vecs else np.array([])


def vec2mlist(
    vec: NDArray[np.floating],
    dims: list[tuple[int, int]],
) -> list[NDArray[np.floating]]:
    """Convert a vector back to a list of matrices.

    This is the inverse of mlist2vec.

    Parameters
    ----------
    vec : ndarray
        Vector of matrix elements.
    dims : list of tuple
        List of (nrow, ncol) for each matrix.

    Returns
    -------
    list of ndarray
        List of matrices.

    Examples
    --------
    >>> vec = np.array([1, 3, 2, 4, 5])
    >>> dims = [(2, 2), (1, 1)]
    >>> vec2mlist(vec, dims)
    [array([[1, 2],
           [3, 4]]), array([[5]])]
    """
    vec = np.asarray(vec)
    matrices: list[NDArray[np.floating]] = []
    idx = 0
    for nrow, ncol in dims:
        size = nrow * ncol
        m = vec[idx : idx + size].reshape((nrow, ncol), order="F")
        matrices.append(m)
        idx += size
    return matrices


def vec2STlist(
    vec: NDArray[np.floating],
    dims: list[int],
) -> list[NDArray[np.floating]]:
    """Convert a vector to a list of lower-triangular matrices.

    Each dimension specifies the size of a lower-triangular matrix.
    The vector contains the lower-triangular elements in column-major order.

    Parameters
    ----------
    vec : ndarray
        Vector of lower-triangular elements.
    dims : list of int
        List of matrix dimensions (q values).

    Returns
    -------
    list of ndarray
        List of lower-triangular matrices.

    Examples
    --------
    >>> vec = np.array([1.0, 0.5, 2.0, 3.0])
    >>> dims = [2, 1]
    >>> vec2STlist(vec, dims)
    [array([[1. , 0. ],
           [0.5, 2. ]]), array([[3.]])]
    """
    vec = np.asarray(vec)
    matrices: list[NDArray[np.floating]] = []
    idx = 0
    for q in dims:
        n_elem = q * (q + 1) // 2
        L = np.zeros((q, q), dtype=np.float64)
        row_indices, col_indices = np.tril_indices(q)
        L[row_indices, col_indices] = vec[idx : idx + n_elem]
        matrices.append(L)
        idx += n_elem
    return matrices


def getL(
    theta: NDArray[np.floating],
    structures: list[RandomEffectStructure],
    sigma: float = 1.0,
    as_blocks: bool = False,
) -> NDArray[np.floating] | sparse.csc_matrix | list[NDArray[np.floating]]:
    """Extract the Cholesky factor L from theta parameters.

    The Cholesky factor L is such that the covariance matrix is
    Cov = sigma^2 * L @ L.T for each random effect block.

    Parameters
    ----------
    theta : ndarray
        Variance component parameters (relative covariance factors).
    structures : list of RandomEffectStructure
        Random effect structures from the model.
    sigma : float, default 1.0
        Residual standard deviation.
    as_blocks : bool, default False
        If True, return a list of L blocks (one per structure).
        If False, return the full block-diagonal Lambda matrix.

    Returns
    -------
    ndarray or sparse matrix or list
        If as_blocks=True: list of L matrices (one per grouping factor).
        If as_blocks=False: sparse block-diagonal Lambda matrix.

    Examples
    --------
    >>> from mixedlm import lmer, load_sleepstudy
    >>> data = load_sleepstudy()
    >>> result = lmer("Reaction ~ Days + (Days|Subject)", data)
    >>> structs = result.matrices.random_structures
    >>> L_blocks = getL(result.theta, structs, result.sigma, as_blocks=True)
    >>> len(L_blocks)  # One block per grouping factor
    1
    """
    from mixedlm.estimation.reml import _build_lambda

    theta = np.asarray(theta)

    if not as_blocks:
        Lambda = _build_lambda(theta, structures)
        return Lambda

    blocks = []
    theta_idx = 0

    for struct in structures:
        q = struct.n_terms
        cov_type = getattr(struct, "cov_type", "us")

        if cov_type == "cs":
            sigma_rel = theta[theta_idx]
            rho = theta[theta_idx + 1] if q > 1 else 0.0
            theta_idx += 2 if q > 1 else 1
            from mixedlm.estimation.reml import _build_cs_cholesky

            L_corr = _build_cs_cholesky(q, rho)
            L_block = sigma_rel * L_corr * sigma
        elif cov_type == "ar1":
            sigma_rel = theta[theta_idx]
            rho = theta[theta_idx + 1] if q > 1 else 0.0
            theta_idx += 2 if q > 1 else 1
            from mixedlm.estimation.reml import _build_ar1_cholesky

            L_corr = _build_ar1_cholesky(q, rho)
            L_block = sigma_rel * L_corr * sigma
        elif struct.correlated:
            n_theta = q * (q + 1) // 2
            theta_block = theta[theta_idx : theta_idx + n_theta]
            theta_idx += n_theta
            L_block = np.zeros((q, q), dtype=np.float64)
            row_indices, col_indices = np.tril_indices(q)
            L_block[row_indices, col_indices] = theta_block
            L_block = L_block * sigma
        else:
            theta_block = theta[theta_idx : theta_idx + q]
            theta_idx += q
            L_block = np.diag(theta_block) * sigma

        blocks.append(L_block)

    return blocks


def condVar(
    model: LmerResult | GlmerResult,
) -> dict[str, dict[str, NDArray[np.floating]]]:
    """Extract conditional variances of random effects.

    The conditional variance is the variance of the random effects
    conditional on the observed data. This is useful for constructing
    prediction intervals for random effects.

    Parameters
    ----------
    model : LmerResult or GlmerResult
        A fitted mixed model.

    Returns
    -------
    dict
        Nested dictionary: {grouping_factor: {term: variance_array}}.
        Each named term has an array of shape ``(n_levels,)``. Groups with
        multiple random terms also include ``"_cov"`` with shape
        ``(n_levels, n_terms, n_terms)``.

    Examples
    --------
    >>> from mixedlm import lmer, load_sleepstudy
    >>> data = load_sleepstudy()
    >>> result = lmer("Reaction ~ Days + (Days|Subject)", data)
    >>> cv = condVar(result)
    >>> cv['Subject']['(Intercept)'].shape
    (18,)
    """
    return model._compute_condVar(include_cov=True)


def safe_chol(
    x: NDArray[np.floating],
    tol: float = 1e-10,
) -> NDArray[np.floating]:
    """Compute Cholesky decomposition with handling for near-singular matrices.

    If the standard Cholesky fails, attempts to regularize the matrix
    by adding a small value to the diagonal.

    Parameters
    ----------
    x : ndarray
        Symmetric positive semi-definite matrix.
    tol : float, default 1e-10
        Tolerance for eigenvalue adjustment.

    Returns
    -------
    ndarray
        Lower-triangular Cholesky factor.

    Examples
    --------
    >>> x = np.array([[1.0, 0.99], [0.99, 1.0]])
    >>> L = safe_chol(x)
    >>> np.allclose(L @ L.T, x)
    True
    """
    x = np.asarray(x)
    try:
        return linalg.cholesky(x, lower=True)
    except linalg.LinAlgError:
        eigvals, eigvecs = linalg.eigh(x)
        eigvals = np.maximum(eigvals, tol)
        x_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return linalg.cholesky(x_fixed, lower=True)
