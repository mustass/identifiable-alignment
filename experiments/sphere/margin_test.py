"""The margin test on the sphere against the number of query points (Fig. 3c).

For a measure P of configs/sphere_measures.yaml, one seed and a reference sample S of N points,
the signature of a query point x is the empirical distribution of its geodesic distances to S
on B radii. The estimated margin at separation t is

    m_hat(t) = min { ||H_x - H_y||_inf : d(x, y) >= t }

over the pairs of query points, and the test compares it with the DKW band 2 eps_N,
eps_N = sqrt(ln(2k / delta) / (2N)), over the k signatures.

The queries are k_max points drawn once and read as nested prefixes of a fixed random order, on
one radius grid fixed from all k_max of them, so a larger k adds pairs and changes nothing else.
Writes <out>/<density>_s<seed>.json: the grid of t and, per k, the band, m_hat(t), the share of
query points whose margin at t exceeds the band, and the number of query pairs at distance >= t.

    PYTHONPATH=src python experiments/sphere/margin_test.py --density mixture_vmf --seed 0 --out results/sphere/margin
"""
import argparse
import json
import os
import time

import numpy as np

from ngma.data.sphere import sample_sphere_measure

DELTA = 0.05


def geodesic(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Great-circle distances between the rows of (m, 3) and (n, 3) unit vectors, float32."""
    G = (A @ B.T).astype(np.float32)
    np.clip(G, -1.0, 1.0, out=G)
    return np.arccos(G, out=G)


def ecdf_rows(Q: np.ndarray, S: np.ndarray, radii: np.ndarray, chunk: int | None = None) -> np.ndarray:
    """H[i, b] = (1/N) #{j : d(Q_i, S_j) <= radii[b]}, radii ascending, without an N x N matrix."""
    k, N = Q.shape[0], S.shape[0]
    B = radii.shape[0]
    if chunk is None:
        chunk = int(max(1, min(k, 4e6 // max(N, 1))))
    H = np.empty((k, B), dtype=np.float64)
    for i0 in range(0, k, chunk):
        i1 = min(i0 + chunk, k)
        D = geodesic(Q[i0:i1], S)
        idx = np.searchsorted(radii, D, side="left")
        for r in range(i1 - i0):
            counts = np.bincount(idx[r], minlength=B + 1)
            H[i0 + r] = np.cumsum(counts)[:B] / float(N)
    return H


def quantile_radii(Q: np.ndarray, S: np.ndarray, B: int, seed: int = 0,
                   n_probe: int = 4000) -> np.ndarray:
    """B ascending radii at the quantiles of the query-to-reference distances, estimated on a
    probe of at most n_probe queries against at most ~2e6 / n_probe reference points."""
    rng = np.random.default_rng(seed)
    probe = Q[rng.choice(Q.shape[0], size=min(n_probe, Q.shape[0]), replace=False)]
    max_d = 2_000_000
    n_s = min(S.shape[0], max(1, max_d // probe.shape[0]))
    Ssub = S if n_s >= S.shape[0] else S[rng.choice(S.shape[0], size=n_s, replace=False)]
    d = geodesic(probe, Ssub).ravel()
    return np.quantile(d, np.linspace(0.0, 1.0, B))


def pairwise_sup_norm(H: np.ndarray, chunk: int = 32) -> np.ndarray:
    """The k x k matrix of sup-norm distances between signature rows, float32."""
    k = H.shape[0]
    Hf = H.astype(np.float32)
    D = np.empty((k, k), dtype=np.float32)
    for i0 in range(0, k, chunk):
        i1 = min(i0 + chunk, k)
        D[i0:i1] = np.max(np.abs(Hf[i0:i1, None, :] - Hf[None, :, :]), axis=2)
    return D


def profile_from_pairs(Cq: np.ndarray, D_sig: np.ndarray, t_grid: np.ndarray,
                       floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """m_hat(t), the share of query points whose margin at t exceeds `floor`, and the number of
    query pairs at distance >= t, on the grid of t. The pairs are sorted by distance once and a
    suffix minimum gives every t."""
    k = Cq.shape[0]
    iu, ju = np.triu_indices(k, 1)
    geo = Cq[iu, ju]
    sig = D_sig[iu, ju].astype(np.float64)
    order = np.argsort(geo)
    geo_s, sig_s = geo[order], sig[order]
    suffmin = np.minimum.accumulate(sig_s[::-1])[::-1]
    pos = np.searchsorted(geo_s, t_grid, side="left")
    n_pairs = geo_s.size - pos
    m_hat = np.where(n_pairs > 0, suffmin[np.minimum(pos, geo_s.size - 1)], np.nan)

    row_geo = Cq + np.diag(np.full(k, -np.inf))
    order_r = np.argsort(row_geo, axis=1)
    geo_r = np.take_along_axis(row_geo, order_r, axis=1)
    sig_r = np.take_along_axis(D_sig.astype(np.float64), order_r, axis=1)
    suffmin_r = np.minimum.accumulate(sig_r[:, ::-1], axis=1)[:, ::-1]
    frac = np.empty_like(t_grid)
    for b, t in enumerate(t_grid):
        idx = np.sum(geo_r < t, axis=1)
        has = idx < k
        margins = suffmin_r[np.arange(k)[has], np.minimum(idx[has], k - 1)]
        frac[b] = float(np.mean(margins > floor)) if has.any() else np.nan
    return m_hat, frac, n_pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[500, 1000, 2000, 3000, 4000, 6000, 8000],
                    help="numbers of query points, read as nested prefixes of one draw")
    ap.add_argument("--N", type=int, default=512_000, help="reference points")
    ap.add_argument("--B", type=int, default=2000, help="radii")
    ap.add_argument("--n-t", type=int, default=129, help="separations t, evenly spaced on [0, pi]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    kmax = max(a.ks)
    S = sample_sphere_measure(a.density, a.N, seed=a.seed)
    Q = sample_sphere_measure(a.density, kmax, seed=10_000 + a.seed)
    Q = Q[np.random.default_rng(a.seed).permutation(kmax)]
    radii = quantile_radii(Q, S, a.B, seed=a.seed)
    H = ecdf_rows(Q, S, radii)
    Cq = geodesic(Q, Q)
    D = pairwise_sup_norm(H)
    t_grid = np.linspace(0.0, np.pi, a.n_t)
    res = {"density": a.density, "seed": a.seed, "N": a.N, "B": a.B, "delta": DELTA,
           "t": t_grid.tolist(), "by_k": {}}
    for k in sorted(a.ks):
        floor = 2.0 * float(np.sqrt(np.log(2.0 * k / DELTA) / (2.0 * a.N)))
        m, frac, npairs = profile_from_pairs(Cq[:k, :k], D[:k, :k], t_grid, floor)
        res["by_k"][k] = {"floor": floor, "m_hat": [None if np.isnan(x) else float(x) for x in m],
                          "frac": [None if np.isnan(x) else float(x) for x in frac],
                          "n_pairs": [int(x) for x in npairs]}
    res["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.density}_s{a.seed}.json"), "w") as fh:
        json.dump(res, fh)
    print(f"{a.density} seed {a.seed}: {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
