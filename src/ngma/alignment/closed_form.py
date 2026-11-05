"""Closed-form, coupling-weighted linear map fits (the weighted Procrustes family)."""
from __future__ import annotations

import numpy as np


def fit_affine(pi: np.ndarray, X: np.ndarray, Y: np.ndarray,
               G: np.ndarray | None = None):
    """argmin_{W,b} sum_ij pi_ij || (W x_i + b) - y_j ||^2_{G_j}; returns (W, b).

    G None: Euclidean weighted least squares against barycentric targets. G (n_b, d, d):
    generalised least squares, one linear system in the d*(d+1) unknowns of [W|b].
    """
    pi = np.asarray(pi, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n_a, d_in = X.shape
    d_out = Y.shape[1]
    Xt = np.concatenate([X, np.ones((n_a, 1))], axis=1)          # (n_a, d_in+1)
    p = d_in + 1

    if G is None:
        r = pi.sum(axis=1)                                        # (n_a,)
        T = pi @ Y                                                # (n_a, d_out), un-normalised
        S = Xt.T @ (r[:, None] * Xt)                              # (p, p)
        R = T.T @ Xt                                              # (d_out, p)
        A = np.linalg.solve(S + 1e-12 * np.trace(S) / p * np.eye(p), R.T).T
    else:
        G = np.asarray(G, dtype=np.float64)
        M = np.einsum("ij,jab->iab", pi, G)                       # (n_a, d, d)
        v = np.einsum("ij,jab,jb->ia", pi, G, Y)                  # (n_a, d)
        N = np.einsum("iab,ic,ie->acbe", M, Xt, Xt).reshape(d_out * p, d_out * p)
        rhs = np.einsum("ia,ic->ac", v, Xt).reshape(d_out * p)
        sol = np.linalg.solve(N + 1e-12 * np.trace(N) / N.shape[0] * np.eye(N.shape[0]), rhs)
        A = sol.reshape(d_out, p)

    return A[:, :d_in], A[:, d_in]                                # W, b


def fit_orthogonal(pi: np.ndarray, X: np.ndarray, Y: np.ndarray,
                   translation: bool = False, allow_reflection: bool = False,
                   scale: bool = False):
    """argmin_{Q} sum_ij pi_ij ||Q x_i + b - y_j||^2 over Q in SO(d) (or O(d) with
    allow_reflection): weighted Procrustes, by the SVD of the pi-weighted cross-covariance.

    translation=True subtracts the pi-weighted means first and returns (Q, b) with
    b = mu_Y - Q mu_X; otherwise Q alone. scale=True returns the similarity fit s*Q with
    s = tr(D S) / tr(Xc^T diag(pi 1) Xc).
    """
    pi = np.asarray(pi, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if translation:
        w = pi.sum()
        mu_x = (pi.sum(axis=1) @ X) / w
        mu_y = (pi.sum(axis=0) @ Y) / w
        Xc, Yc = X - mu_x, Y - mu_y
    else:
        Xc, Yc = X, Y
    C = Yc.T @ pi.T @ Xc                                          # (d, d)
    U, _, Vt = np.linalg.svd(C)
    D = np.eye(C.shape[0])
    if not allow_reflection:
        D[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    Q = U @ D @ Vt
    if scale:
        _, S, _ = np.linalg.svd(C)
        wx = pi.sum(axis=1)
        denom = float((wx[:, None] * Xc * Xc).sum())
        s_hat = float((np.diag(D) * S).sum() / denom) if denom > 1e-12 else 1.0
        Q = s_hat * Q
    return (Q, mu_y - Q @ mu_x) if translation else Q


def orientation_pair(pi: np.ndarray, X: np.ndarray, Y: np.ndarray,
                     translation: bool = False, scale: bool = False):
    """Both orientations of the weighted Procrustes fit, and how far apart they are.

    Returns `(procrustes, mirror, info)`. Each map is `(Q, b)`, with `b` zero when
    `translation` is False. `procrustes` is the unconstrained optimum in O(d) and `mirror`
    the same fit with the last singular direction flipped. With C = U S V^T the two score
    sum(S) and sum(S) - 2*S[-1]; info reports the relative margin 2*S[-1] / sum(S), the
    smallest singular value, their sum, and the sign of det(U V^T).
    """
    pi = np.asarray(pi, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    d = X.shape[1]
    if translation:
        w = pi.sum()
        mu_x = (pi.sum(axis=1) @ X) / w
        mu_y = (pi.sum(axis=0) @ Y) / w
        Xc, Yc = X - mu_x, Y - mu_y
    else:
        mu_x = mu_y = np.zeros(d)
        Xc, Yc = X, Y
    C = Yc.T @ pi.T @ Xc
    U, S, Vt = np.linalg.svd(C)
    wx = pi.sum(axis=1)
    denom = float((wx[:, None] * Xc * Xc).sum())

    def build(flip):
        D = np.eye(d)
        if flip:
            D[-1, -1] = -1.0
        Q = U @ D @ Vt
        if scale:
            Q = (float((np.diag(D) * S).sum() / denom) if denom > 1e-12 else 1.0) * Q
        return Q, (mu_y - Q @ mu_x if translation else np.zeros(d))

    procrustes, mirror = build(False), build(True)
    ssum, smin = float(S.sum()), float(S[-1])
    info = {"reflect_margin": (2.0 * smin / ssum) if ssum > 1e-12 else 0.0,
            "sigma_min": smin, "sigma_sum": ssum,
            "det_sign": float(np.sign(np.linalg.det(U @ Vt))) or 1.0}
    return procrustes, mirror, info
