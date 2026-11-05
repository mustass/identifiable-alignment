#!/usr/bin/env python
"""Pullback geometry of one lung scVI model on the pool of held-out cells, in NGMA's terms.

Head k of the model (decoder_heads.py) maps a latent z to normalised expression rho_k(z), and
its observation map is

    phi_k(z) = 2 sqrt(theta) * asinh( sqrt( l_ref * rho_k(z) / theta ) ),

the coordinate in which the Fisher metric of the negative-binomial mean, at the model's gene
dispersions theta and the reference library size l_ref, is Euclidean. The M-decoder ensemble
metric in these coordinates is

    G_M(z) = (1/M) sum_{k<M} J_k(z)^T J_k(z),        J_k = d phi_k / d z,

and the distance between two cells is the length of the straight latent segment between them
under G_M, discretised into T steps,

    C_M(i, j) = sum_{t=1..T} sqrt( (1/M) sum_{k<M} || phi_k(z_t) - phi_k(z_{t-1}) ||^2 ).

One decode of all K heads gives C_M for every M.

Outputs in <root>/geometry/{tag}/:
    Z_pool.npy, Z_test.npy    posterior means of the pool cells and of the eval cells
    C_M{M}.npy                (N, N) float32 for each M in --M
    G_M{M}.npy                (N, d, d) float64 for each M
    geometry.json             settings, pool definition, timings

    python experiments/lung/geometry.py --root <root> --tag H1_19600_s0 [--cpu]
    python experiments/lung/geometry.py --root <root> --tag H1_19600_s0 --n-pool 64   # quick test
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

POOL_SETS = ("shared", "only_a")          # the 9,000 held-out cells both models encode


def load_heads(path, K):
    """Stack the K heads' weights: W{l} is (K, out, in) and b{l} is (K, out), torch layout."""
    import jax.numpy as jnp
    H = np.load(path)
    P = {}
    for l in range(4):
        P[f"W{l}"] = jnp.asarray(np.stack([H[f"W{k}_{l}"] for k in range(K)]), jnp.float32)
        P[f"b{l}"] = jnp.asarray(np.stack([H[f"b{k}_{l}"] for k in range(K)]), jnp.float32)
    P["theta"] = jnp.asarray(H["theta"], jnp.float32)
    P["l_ref"] = float(H["l_ref"])
    return P


def make_phi(P):
    """phi(z): (..., d) -> (..., K, n_genes), every head at once."""
    import jax
    import jax.numpy as jnp

    def ln(h):                                            # torch LayerNorm, no affine, eps 1e-5
        m = h.mean(-1, keepdims=True)
        v = ((h - m) ** 2).mean(-1, keepdims=True)
        return (h - m) / jnp.sqrt(v + 1e-5)

    sq_theta = jnp.sqrt(P["theta"])

    def phi(z):
        h = jax.nn.relu(ln(jnp.einsum("...d,khd->...kh", z, P["W0"]) + P["b0"]))
        for l in (1, 2):
            h = jax.nn.relu(ln(jnp.einsum("...kh,kgh->...kg", h, P[f"W{l}"]) + P[f"b{l}"]))
        # sqrt(rho) as exp(log_softmax / 2): the same value, with a derivative that stays finite
        # where a gene's softmax underflows to 0
        sqrt_rho = jnp.exp(0.5 * jax.nn.log_softmax(jnp.einsum("...kh,kgh->...kg", h, P["W3"]) + P["b3"], axis=-1))
        return 2.0 * sq_theta * jnp.arcsinh(jnp.sqrt(P["l_ref"]) * sqrt_rho / sq_theta)

    return phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the experiment's root directory")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--M", default="1,8")
    ap.add_argument("--steps", type=int, default=10, help="T, the segment discretisation")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=128)
    ap.add_argument("--n-pool", type=int, default=0, help="first n pool cells only (a quick test)")
    ap.add_argument("--cpu", action="store_true", help="run JAX on the CPU")
    a = ap.parse_args()
    if a.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"                # must precede the first jax import
    import jax
    import jax.numpy as jnp

    root = Path(a.root)
    Ms = sorted(int(m) for m in a.M.split(","))
    if Ms[-1] > a.K:
        raise SystemExit(f"REFUSING: M={Ms[-1]} exceeds the {a.K} heads")
    fit = root / "fits" / a.tag
    out = root / "geometry" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    split = np.load(root / "split.npz")
    pool_idx = np.concatenate([split[s] for s in POOL_SETS])
    if a.n_pool:
        pool_idx = pool_idx[:a.n_pool]
    mu = np.load(fit / "latents.npz")["mu"]
    Z = mu[pool_idx].astype(np.float32)
    np.save(out / "Z_pool.npy", mu[pool_idx])
    np.save(out / "Z_test.npy", mu[split["eval"]])
    N, d = Z.shape
    P = load_heads(fit / "heads.npz", a.K)
    phi = make_phi(P)
    ts = jnp.linspace(0.0, 1.0, a.steps + 1)
    Ms_j = jnp.asarray(Ms)

    @jax.jit
    def block(zr, zc):
        """(R, d), (C, d) -> (len(Ms), R, C) segment lengths under G_M for every M."""
        pts = zr[:, None, None, :] + ts[None, None, :, None] * (zc[None, :, None, :] - zr[:, None, None, :])
        f = phi(pts)                                       # (R, C, T+1, K, G)
        sq = ((f[:, :, 1:] - f[:, :, :-1]) ** 2).sum(-1)   # (R, C, T, K)
        csum = jnp.cumsum(sq, axis=-1)                     # mean over the first M heads
        means = csum[..., Ms_j - 1] / Ms_j                 # (R, C, T, len(Ms))
        return jnp.moveaxis(jnp.sqrt(means + 1e-12).sum(2), -1, 0)

    t0 = time.time()
    C = np.zeros((len(Ms), N, N), np.float32)
    Zj = jnp.asarray(Z)
    R, Cc = a.rows, a.cols
    pad = lambda x, n: jnp.concatenate([x, jnp.repeat(x[-1:], n - len(x), 0)]) if len(x) < n else x
    for r0 in range(0, N, R):
        zr = pad(Zj[r0:r0 + R], R)
        nr = min(R, N - r0)
        for c0 in range(r0 - r0 % Cc, N, Cc):              # upper triangle, whole column chunks
            zc = pad(Zj[c0:c0 + Cc], Cc)
            nc = min(Cc, N - c0)
            C[:, r0:r0 + nr, c0:c0 + nc] = np.asarray(block(zr, zc))[:, :nr, :nc]
        if r0 % (R * 100) == 0:
            print(f"  rows {r0}/{N}  {time.time() - t0:.0f}s", flush=True)
    for i in range(C.shape[0]):                            # mirror the upper triangle
        U = np.triu(C[i])
        C[i] = U + np.triu(U, 1).T
        np.fill_diagonal(C[i], 0.0)
    t_c = time.time() - t0

    t0 = time.time()
    jac = jax.jit(jax.vmap(jax.jacfwd(phi)))               # (B, K, G, d)
    G = np.zeros((len(Ms), N, d, d), np.float64)
    for s in range(0, N, 64):
        J = jac(Zj[s:s + 64])
        g = jnp.einsum("bkgi,bkgj->bkij", J, J)            # per-head J^T J
        cg = jnp.cumsum(g, axis=1)
        for i, M in enumerate(Ms):
            G[i, s:s + 64] = np.asarray(cg[:, M - 1] / M, np.float64)
    t_g = time.time() - t0
    bad = ~np.isfinite(G).all(axis=(2, 3))
    if bad.any():
        raise SystemExit(f"REFUSING: non-finite metric tensors at {int(bad.any(0).sum())} cells")

    for i, M in enumerate(Ms):
        np.save(out / f"C_M{M}.npy", C[i])
        np.save(out / f"G_M{M}.npy", G[i])
    rec = {"tag": a.tag, "N": int(N), "d": int(d), "K": a.K, "M": Ms, "T": a.steps,
           "pool_sets": list(POOL_SETS), "n_pool_cap": a.n_pool, "l_ref": P["l_ref"],
           "secs_C": round(t_c, 1), "secs_G": round(t_g, 1), "device": str(jax.devices()[0])}
    json.dump(rec, open(out / "geometry.json", "w"), indent=1)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
