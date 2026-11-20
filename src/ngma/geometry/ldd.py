"""Local distance distributions: signatures, landmark selection and the third-lower-bound cost.

A point's local distance distribution is the empirical CDF of its distances to every other
point. `compute_ldd_signatures` evaluates those CDFs on a common grid of radii;
`greedy_dopt_selection` picks landmarks whose signatures span the largest volume;
`tlb_cost` is the ground cost of Memoli's third lower bound, the linear term of the fused
coupling.
"""

import numpy as np
from scipy.linalg import solve_triangular


def _ecdf_from_D(D, radii, chunk=None):
    """H[i, b] = (1/N) #{j : D[i, j] <= radii[b]} for ascending radii, row block by row block."""
    k, N = D.shape
    B = radii.shape[0]
    if chunk is None:
        chunk = int(max(1, min(k, 4e7 // max(N, 1))))
    H = np.empty((k, B), dtype=np.float64)
    for i0 in range(0, k, chunk):
        i1 = min(i0 + chunk, k)
        idx = np.searchsorted(radii, D[i0:i1], side="left")
        for r in range(i1 - i0):
            counts = np.bincount(idx[r], minlength=B + 1)
            H[i0 + r] = np.cumsum(counts)[:B] / float(N)
    return H


def compute_ldd_signatures(
    C: np.ndarray,
    n_bins: int = 2000,
    r_max: float = None,
    use_quantiles: bool = True,
) -> np.ndarray:
    """The signature matrix H of shape (n, n_bins) and the radii it is evaluated on.

    C:             (n, n) pairwise distance matrix
    n_bins:        number of radii B
    r_max:         largest radius of the uniform grid (default: max(C)); used when
                   use_quantiles is False
    use_quantiles: place the radii at the quantiles of all pairwise distances (True) or on
                   a uniform grid on [0, r_max] (False)

    H[i, k] is the fraction of points j with C[i, j] <= r_k.
    """
    n = C.shape[0]

    if use_quantiles:
        all_dists = C[np.triu_indices(n, k=1)]
        radii = np.quantile(all_dists, np.linspace(0.0, 1.0, n_bins))
    else:
        r_max = r_max if r_max is not None else C.max()
        radii = np.linspace(0.0, r_max, n_bins)

    H = _ecdf_from_D(C, radii).astype(np.float32)
    return H, radii


def greedy_dopt_selection(H: np.ndarray, k: int, fill: str | None = None,
                          rel_tol: float = 1e-9, return_switch: bool = False):
    """Greedy D-optimal landmark selection.

    Selects k of the n rows of H whose Gram matrix G_S = H_S H_S^T has the largest log-det,
    one point at a time. The Schur complement of every candidate is evaluated at once from
    an incrementally maintained Cholesky factor of G_S and the matrix M = H_S^T G_S^{-1} H_S,
    so a step costs O(n B^2).

    H:  (n, B) signature matrix
    k:  number of landmarks
    fill: None keeps the greedy selection to the end. "maxmin" stops it once its largest
        Schur increment falls below max(rel_tol * max ||h||^2, 1e-7) or once B points have
        been chosen (past the rank of H every increment is round-off), and continues by
        farthest-point sampling in signature L2 distance from the points already chosen.
    return_switch: also return the number of points the greedy selection chose before the
        switch (k when it never switched).

    Returns an integer index array of length k (and the switch index with return_switch).
    """
    n, B = H.shape
    if fill not in (None, "maxmin"):
        raise ValueError(f"fill must be None or 'maxmin', got {fill!r}")
    if k >= n:
        return (np.arange(n), n) if return_switch else np.arange(n)

    H64 = H.astype(np.float64)
    norms_sq = (H64 ** 2).sum(1)
    selected = []
    mask = np.ones(n, dtype=bool)

    best = int(np.argmax(norms_sq))
    selected.append(best)
    mask[best] = False

    h0 = H64[best]
    l00 = np.sqrt(float(norms_sq[best]) + 1e-8)
    L = np.array([[l00]])                   # Cholesky factor of the selected Gram matrix
    H_sel = h0[None, :]
    A = H_sel / l00                         # L^{-1} H_sel, so that M = A^T A
    M = A.T @ A

    switch = k
    floor = max(rel_tol * float(norms_sq.max()), 1e-7)
    for step in range(1, k):
        schurs = norms_sq - (H64 @ M * H64).sum(1)
        schurs[~mask] = -np.inf

        if fill == "maxmin" and (step >= B or float(schurs.max()) < floor):
            switch = step
            break

        best = int(np.argmax(schurs))
        mask[best] = False
        h_new = H64[best]

        m = len(selected)
        v = H_sel @ h_new
        z = solve_triangular(L, v, lower=True)
        s = float(norms_sq[best]) + 1e-8
        l_new = np.sqrt(max(s - float(np.dot(z, z)), 1e-8))

        L_new = np.zeros((m + 1, m + 1))
        L_new[:m, :m] = L
        L_new[m,  :m] = z
        L_new[m,   m] = l_new
        L = L_new

        a_new = (h_new - z @ A) / l_new
        A = np.vstack([A, a_new])
        M = A.T @ A

        selected.append(best)
        H_sel = H64[np.array(selected)]

    if switch < k:
        d2 = np.full(n, np.inf)
        for j in selected:
            d2 = np.minimum(d2, ((H64 - H64[j]) ** 2).sum(1))
        d2[~mask] = -np.inf
        while len(selected) < k:
            best = int(np.argmax(d2))
            selected.append(best)
            mask[best] = False
            d2 = np.minimum(d2, ((H64 - H64[best]) ** 2).sum(1))
            d2[~mask] = -np.inf

    out = np.array(selected)
    return (out, switch) if return_switch else out


def tlb_cost(Ca: np.ndarray, Cb: np.ndarray, trim: float = 0.0) -> np.ndarray:
    """Ground cost of Memoli's third lower bound for the squared loss.

    c(i, j) is W_2^2 between the distance distribution of point i of side a and that of point
    j of side b: the mean squared difference of their sorted distance rows (exact for equal
    sample sizes). For every coupling T with uniform marginals c(i, j) is at most GW's
    linearised cost at T, entrywise, and in the same units.

    trim > 0 keeps only the quantile levels in [trim, 1 - trim]. Returns (n_a, n_b) float64.
    """
    Ca = np.asarray(Ca, np.float64); Cb = np.asarray(Cb, np.float64)
    if len(Ca) != len(Cb):
        raise NotImplementedError("tlb_cost needs equal sample sizes")
    Sa, Sb = np.sort(Ca, axis=1), np.sort(Cb, axis=1)
    if trim > 0:
        n = Sa.shape[1]
        lo, hi = int(np.ceil(trim * n)), int(np.floor((1 - trim) * n))
        Sa, Sb = Sa[:, lo:hi], Sb[:, lo:hi]
    m = Sa.shape[1]
    return np.maximum((Sa ** 2).sum(1)[:, None] + (Sb ** 2).sum(1)[None, :] - 2.0 * Sa @ Sb.T, 0.0) / m
