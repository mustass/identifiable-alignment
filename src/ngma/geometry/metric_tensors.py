"""Pointwise expected pullback metric tensors G(z) = E_k[J_k(z)^T J_k(z)].

Near a point y the pullback distance is a quadratic form, d_b(z, y)^2 ~= (z - y)^T G(y) (z - y),
with J_k the Jacobian of decoder k. G is one (d, d) matrix per sample point, computed once
from the decoder ensemble; everything downstream reads the cached field. The functions take
any map z (d,) -> (K, D), i.e. the models' `decode_single_ensemble`.
"""
from __future__ import annotations

import logging

import numpy as np


def expected_metric_tensors(decode_single_ensemble, Z: np.ndarray,
                            chunk: int = 64) -> np.ndarray:
    """G[i] = mean_k J_k(Z[i])^T J_k(Z[i]), shape (n, d, d), float64.

    decode_single_ensemble: z (d,) -> (K, D) flat decoder outputs (all heads). Uses
    forward-mode Jacobians, vmapped over a chunk of points.
    """
    import jax
    import jax.numpy as jnp

    Z = np.asarray(Z)
    n, d = Z.shape

    # The (chunk, K) batch of Jacobians is collapsed into one batch dimension before the
    # contraction, so that it lowers to a plain batched matmul.
    def _g_chunk_impl(Zc):
        J = jax.vmap(jax.jacfwd(decode_single_ensemble))(Zc)   # (c, K, D, d)
        c, K, D = J.shape[0], J.shape[1], J.shape[2]
        Jf = J.reshape(c * K, D, d)
        G = jnp.matmul(jnp.swapaxes(Jf, 1, 2), Jf)             # (c*K, d, d)
        return G.reshape(c, K, d, d).mean(axis=1)              # (c, d, d)

    _g_chunk = jax.jit(_g_chunk_impl)

    out = np.empty((n, d, d), dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        out[i0:i1] = np.asarray(_g_chunk(jnp.asarray(Z[i0:i1])), dtype=np.float64)
        if i0 == 0 or (i0 // chunk) % 10 == 0:
            logging.info(f"  metric tensors: {i1}/{n}")
    return out


def spd_cholesky(G: np.ndarray, jitter_rel: float = 1e-6):
    """Cholesky factors of G with a relative jitter floor; returns (L, stats).

    Where G is numerically rank-deficient the factorisation is retried with a jitter
    relative to the point's mean eigenvalue (trace / d). stats reports how often that was
    needed and the condition numbers.
    """
    G = np.asarray(G, dtype=np.float64)
    n, d, _ = G.shape
    scale = np.trace(G, axis1=1, axis2=2) / d                       # (n,)
    eye = np.eye(d)
    L = np.empty_like(G)
    jittered = 0
    for i in range(n):
        try:
            L[i] = np.linalg.cholesky(G[i])
        except np.linalg.LinAlgError:
            L[i] = np.linalg.cholesky(G[i] + jitter_rel * max(scale[i], 1e-30) * eye)
            jittered += 1
    eig = np.linalg.eigvalsh(G)                                      # (n, d), ascending
    cond = eig[:, -1] / np.maximum(eig[:, 0], 1e-30)
    stats = {
        "frac_jittered": jittered / n,
        "cond_median": float(np.median(cond)),
        "cond_p95": float(np.quantile(cond, 0.95)),
        "min_eig_min": float(eig[:, 0].min()),
        "scale_median": float(np.median(scale)),
    }
    return L, stats


def metric_whitener(G: np.ndarray, mode: str = "mean", alpha: float = 1.0):
    """Symmetric square root of a cloud's mean pullback metric, or None when mode == "none".

    Returns {"W", "Wi", "cond"} with W = Gbar^(alpha/2) and Wi its inverse, both symmetric and
    scaled to unit determinant. Eigenvalues of Gbar are floored at 1e-6 of the largest. alpha
    interpolates between no whitening (0) and making Gbar the identity (1).
    """
    if mode == "none":
        return None
    if mode != "mean":
        raise ValueError(f"unknown whitening mode {mode!r}")
    Gbar = np.asarray(G, dtype=np.float64).mean(0)
    Gbar = 0.5 * (Gbar + Gbar.T)
    w, V = np.linalg.eigh(Gbar)
    hi = float(w.max())
    cond = hi / max(float(w.min()), hi * 1e-30)
    w = np.clip(w, hi * 1e-6, None)
    r = np.power(w, 0.5 * float(alpha))
    r = r / float(np.exp(np.log(r).mean()))
    return {"W": (V * r) @ V.T, "Wi": (V / r) @ V.T,
            "cond": cond, "alpha": float(alpha),
            "cond_effective": float((float(r.max()) / float(r.min())) ** 2)}


def whiten_apply(Z: np.ndarray, W):
    """Z is (n, d) of row vectors; W["W"] is symmetric, so Z @ W is W z per row."""
    return Z if W is None else np.asarray(Z, dtype=np.float64) @ W["W"]


def whiten_unapply(Z: np.ndarray, W):
    return Z if W is None else np.asarray(Z, dtype=np.float64) @ W["Wi"]


def whiten_metric(G: np.ndarray, W, s: float):
    """The metric in the whitened, rescaled frame.

    With z = mu + s Wi z~ we get dz = s Wi dz~, so dz^T G dz = s^2 dz~^T Wi G Wi dz~. W = None
    reduces to the scalar rescaling G' = s^2 G.
    """
    G = np.asarray(G, dtype=np.float64)
    if W is None:
        return G * s ** 2
    return s ** 2 * np.einsum("ab,nbc,cd->nad", W["Wi"], G, W["Wi"])


def whiten_targets(pi: np.ndarray, Y: np.ndarray, G: np.ndarray):
    """Per-source sufficient statistics of the metric-weighted coupling-fit loss.

    sum_ij pi_ij (f(x_i) - y_j)^T G_j (f(x_i) - y_j)
        = sum_i [ f(x_i)^T M_i f(x_i) - 2 f(x_i)^T v_i ] + const,
    with M_i = sum_j pi_ij G_j  (n_a, d, d)  and  v_i = sum_j pi_ij G_j y_j  (n_a, d).
    """
    pi = np.asarray(pi, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    M = np.einsum("ij,jab->iab", pi, G)                 # (n_a, d, d)
    v = np.einsum("ij,jab,jb->ia", pi, G, Y)            # (n_a, d)
    return M, v
