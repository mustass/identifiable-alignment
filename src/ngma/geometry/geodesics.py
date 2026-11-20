"""Pairwise distance matrices of a latent sample.

  euclidean               L2 in latent space
  ensemble_eol_samedec    the length of the straight segment between two codes, measured
                          under the decoder ensemble: the segment is discretised into
                          n_steps steps, each step's length is the root mean over the K
                          decoders of the squared output difference, and the lengths are
                          summed (`ensemble_line_energy_matrix`)
  sphere                  great-circle distance on S^2 (`sphere_distance_matrix`)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import jax.numpy as jnp
import numpy as np
from flax import nnx


def _block_lines(Z_rows, Z_cols, ts):
    """(R, C, T, d) straight-line points, flattened to (R*C*T, d) for one decode call."""
    R, d = Z_rows.shape
    C = Z_cols.shape[0]
    T = ts.shape[0]
    pts = (Z_rows[:, None, None, :]
             + ts[None, None, :, None] * (Z_cols[None, :, None, :] - Z_rows[:, None, None, :]))
    lines = pts.reshape(R * C * T, d)
    return lines, R, C, T


@nnx.jit
def _ens_block_samedec(model, Z_rows, Z_cols, ts):
    """Segment lengths of every (row, column) pair under the same-decoder mean energy."""
    flat, R, C, T = _block_lines(Z_rows, Z_cols, ts)
    dec = model.decode_ensemble(flat)
    dec = dec.reshape(R, C, T, dec.shape[1], dec.shape[2])  # (R,C,T,K,dx)
    diffs = dec[:, :, 1:, :, :] - dec[:, :, :-1, :, :]      # (R,C,T-1,K,dx)
    step_sq = (diffs ** 2).sum(-1).mean(-1)                 # (R,C,T-1) mean over K
    return jnp.sqrt(step_sq + 1e-10).sum(-1)               # (R,C)


def sphere_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Great-circle distances between unit vectors."""
    pts = np.asarray(points, dtype=np.float32)
    cos_sim = np.clip(pts @ pts.T, -1.0, 1.0)
    return np.arccos(cos_sim).astype(np.float32)


def euclidean_distance_matrix(Z: np.ndarray) -> np.ndarray:
    """Pairwise L2 distances, accumulated in float64, with an exact-zero diagonal."""
    Z = np.asarray(Z, dtype=np.float64)
    sq = (Z**2).sum(1, keepdims=True) + (Z**2).sum(1, keepdims=True).T - 2.0 * Z @ Z.T
    C = np.sqrt(np.maximum(sq, 0.0))
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 0.0)
    return C.astype(np.float32)


def ensemble_line_energy_matrix(
    Z: np.ndarray,
    model,
    n_steps: int = 10,
    metric_mode: str = "samedec",
    row_block: int = 4,
    col_chunk: int = 128,
    checkpoint_path: Optional[str] = None,
    checkpoint_every_s: float = 1200.0,
    log_every_s: float = 300.0,
    **_ignored,
) -> np.ndarray:
    """Symmetric (n, n) matrix of straight-segment lengths under the ensemble pullback metric.

    A block of row_block x col_chunk pairs, each discretised into n_steps + 1 points, is
    decoded in one `model.decode_ensemble` call. Only the upper triangle is computed and it
    is mirrored. Pass checkpoint_path to make the computation resumable at row-block
    granularity; the checkpoint is removed once the matrix is complete.
    """
    if metric_mode not in ("samedec", "ensemble_samedec"):
        raise ValueError(f"metric_mode must be 'samedec', got {metric_mode!r}")
    kernel = _ens_block_samedec

    Zj = jnp.array(Z)
    n = Z.shape[0]
    ts = jnp.linspace(0.0, 1.0, n_steps + 1)
    out = np.zeros((n, n), dtype=np.float32)

    rows_done = 0
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path)
        out = ckpt["out"].copy()
        rows_done = int(ckpt["rows_done"])
        logging.info(f"  Line-energy checkpoint: resuming at row {rows_done}/{n}")

    def _pairs_upto(r):
        """Pairs computed by the row blocks covering rows [0, r)."""
        tot = 0
        for b0 in range(0, r, row_block):
            tot += (min(b0 + row_block, n) - b0) * (n - b0)
        return tot

    pairs_total = _pairs_upto(n)
    pairs_base = _pairs_upto(rows_done)
    t_start = time.perf_counter()
    last_log_t = t_start
    last_ckpt_t = t_start

    for r0 in range(rows_done, n, row_block):
        r1 = min(r0 + row_block, n)
        Zr = Zj[r0:r1]
        for c0 in range(r0, n, col_chunk):
            c1 = min(c0 + col_chunk, n)
            vals = np.array(kernel(model, Zr, Zj[c0:c1], ts))       # (r1-r0, c1-c0)
            out[r0:r1, c0:c1] = vals
            out[c0:c1, r0:r1] = vals.T

        now = time.perf_counter()
        first = (r0 == rows_done)
        if first or now - last_log_t >= log_every_s or r1 >= n:
            done = _pairs_upto(r1) - pairs_base
            elapsed = now - t_start
            frac = done / max(pairs_total - pairs_base, 1)
            eta_h = (elapsed / max(frac, 1e-12) - elapsed) / 3600.0
            logging.info(
                f"  Line energy rows {r1}/{n}: {100*frac:.1f}% of pairs "
                f"({done/1e6:.2f}M/{(pairs_total - pairs_base)/1e6:.2f}M)  "
                f"elapsed={elapsed/3600.0:.2f}h  ETA={eta_h:.1f}h"
                + ("  [first block includes compilation]" if first else "")
            )
            last_log_t = now

        if checkpoint_path is not None and (now - last_ckpt_t >= checkpoint_every_s
                                            or r1 >= n):
            tmp = checkpoint_path + ".tmp.npz"
            np.savez(tmp, out=out, rows_done=r1)
            os.replace(tmp, checkpoint_path)
            last_ckpt_t = time.perf_counter()

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    np.fill_diagonal(out, 0.0)
    return out


def compute_distance_matrix(
    Z: np.ndarray,
    mode: str = "euclidean",
    model=None,
    checkpoint_path: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """The (n, n) distance matrix of Z in one of the modes of `ngma.geometry.modes`.

    model: the VAE (needed for ensemble_eol_samedec). kwargs are forwarded to
    `ensemble_line_energy_matrix` (n_steps, row_block, col_chunk, checkpoint_every_s,
    log_every_s).
    """
    from ngma.geometry.modes import resolve as _resolve_mode

    mode = _resolve_mode(mode)
    if mode == "euclidean":
        return euclidean_distance_matrix(Z)
    assert model is not None, "model required for ensemble_eol_samedec"
    kw = {k: v for k, v in kwargs.items()
          if k in ("n_steps", "row_block", "col_chunk", "checkpoint_every_s", "log_every_s")}
    return ensemble_line_energy_matrix(Z, model, metric_mode="samedec",
                                       checkpoint_path=checkpoint_path, **kw)
