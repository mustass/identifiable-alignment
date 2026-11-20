"""Paired retrieval from a distance matrix, plain and CSLS-rescored.

Both scorers rank each query's true partner (the diagonal of the matrix) among all
candidates with ties averaged, and return `tie_frac`, the mean fraction of off-diagonal
entries tied with the diagonal, beside the scores.
"""
from __future__ import annotations

import numpy as np


def sq_dists(A: np.ndarray, B: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Squared Euclidean distances (n_a, n_b), float64 accumulation, chunked over rows."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    out = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)
    bb = (B ** 2).sum(1)
    for i0 in range(0, A.shape[0], chunk):
        i1 = min(i0 + chunk, A.shape[0])
        out[i0:i1] = ((A[i0:i1] ** 2).sum(1)[:, None] + bb[None, :]
                      - 2.0 * A[i0:i1] @ B.T)
    np.maximum(out, 0.0, out=out)
    return out


def _tie_averaged_ranks(score: np.ndarray, tol: float) -> tuple:
    """Rank of the diagonal in each row of a higher-is-better score matrix, ties averaged.

    Returns (ranks, tie_frac): tie_frac is the mean fraction of off-diagonal entries tied
    with the diagonal, near 0 when the distances are distinct and near 1 on a collapsed matrix.
    """
    diag = np.diagonal(score)[:, None]
    scale = np.maximum(np.abs(diag), 1.0)
    greater = (score > diag + tol * scale).sum(axis=1)
    ties = np.isclose(score, diag, rtol=0.0, atol=tol * scale).sum(axis=1) - 1   # drop the diagonal
    n_off = max(score.shape[1] - 1, 1)
    return greater + ties / 2.0, float((ties / n_off).mean())


def retrieval_from_D(D: np.ndarray, ks=(1, 5), tol: float = 1e-12) -> dict:
    """Paired retrieval@k from a distance matrix (truth = the diagonal), ties averaged."""
    n = D.shape[0]
    ranks, tie_frac = _tie_averaged_ranks(-np.asarray(D, dtype=np.float64), tol)
    return {f"at{k}": float((ranks < k).mean()) for k in ks} | {"n": n, "tie_frac": tie_frac}


def csls_retrieval(D: np.ndarray, k: int = 10, ks=(1, 5), tol: float = 1e-12) -> dict:
    """CSLS-rescored paired retrieval (Conneau et al. 2018) from a distance matrix.

    Similarity s = -D. r_T(j) is the mean of the k best s(., j) over queries and r_S(i) the
    same over targets; the score is 2 s - r_T - r_S, ranked per row, ties averaged.
    """
    S = -np.asarray(D, dtype=np.float64)
    kk = min(k, S.shape[0] - 1, S.shape[1] - 1)
    r_S = np.sort(S, axis=1)[:, -kk:].mean(axis=1)               # (n_a,)
    r_T = np.sort(S, axis=0)[-kk:, :].mean(axis=0)               # (n_b,)
    C = 2.0 * S - r_S[:, None] - r_T[None, :]
    ranks, tie_frac = _tie_averaged_ranks(C, tol)
    return {f"at{k_}": float((ranks < k_).mean()) for k_ in ks} | {"k": kk, "tie_frac": tie_frac}
