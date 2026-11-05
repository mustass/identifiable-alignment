#!/usr/bin/env python
"""Retrieval and CSLS for the baselines, from the same implementation that scores NGMA.

`ngma.evaluation.coupling_metrics` holds the retrieval code (tie-averaged ranks, CSLS with
its tie fraction). It is loaded here straight from its file, so a baseline that runs in a
PyTorch-only environment never imports the `ngma` package and therefore never imports JAX.

CSLS needs the whole n x n score matrix, so it is computed on at most `max_n` rows (the
cap is returned as `csls_pool_n`); plain retrieval is reported on the full split and, for
a same-task comparison, on the same capped rows (`plain_at_1_csls_pool`).
"""
import importlib.util
import os

import numpy as np

_CM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "ngma",
                   "evaluation", "coupling_metrics.py")


def _canonical():
    """Load `coupling_metrics` from its file, without importing the `ngma` package."""
    spec = importlib.util.spec_from_file_location("_ngma_coupling_metrics", os.path.normpath(_CM))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def retrieval_and_csls(A, B, max_n: int = 10000, seed: int = 0, prefix: str = "") -> dict:
    """Plain and CSLS retrieval between two aligned representations, row i of A matching row i
    of B. Returns CSLS@1/@5 on the (possibly capped) subsample with its tie fraction and pool
    size, and plain R@1/R@5 on the same rows."""
    cm = _canonical()
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = len(A)
    idx = (np.random.default_rng(seed).choice(n, max_n, replace=False) if n > max_n
           else np.arange(n))
    D = np.sqrt(np.maximum((A[idx] ** 2).sum(1)[:, None] + (B[idx] ** 2).sum(1)[None, :]
                           - 2.0 * A[idx] @ B[idx].T, 0.0))
    c = cm.csls_retrieval(D, ks=(1, 5))
    p = cm.retrieval_from_D(D, ks=(1, 5))
    return {f"{prefix}csls_at1": c["at1"], f"{prefix}csls_at5": c["at5"],
            f"{prefix}csls_tie_frac": c["tie_frac"], f"{prefix}csls_pool_n": float(len(idx)),
            f"{prefix}plain_at_1_csls_pool": p["at1"], f"{prefix}plain_at_5_csls_pool": p["at5"]}


CSLS_FIELDS = ["csls_at1", "csls_at5", "csls_tie_frac", "csls_pool_n",
               "plain_at_1_csls_pool", "plain_at_5_csls_pool"]
