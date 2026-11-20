#!/usr/bin/env python
"""RelGeo's relative-geodesic distances for a lung carve, from the scVI models' decoder ensembles.

The lung counterpart of baselines/relgeo_lengths.py, which loads NGMA's own VAE
checkpoints and so cannot take scVI models. Same quantity, same file: for each side, the length
of the straight latent segment from every test cell to every class-centroid anchor under the
model's M-decoder ensemble metric, E = sum_t sqrt(mean_k ||phi_k(z_t) - phi_k(z_{t-1})||^2)
with `--steps` steps (RelGeo's default 30), phi_k as in geometry.py; plus anchor-to-anchor
lengths. Anchors are the class centroids of the carve's training latents, over the classes
present on both sides. Writes the .npz that `run_relgeo.py --geo-mode precomputed --geo-npz`
reads.

    python experiments/lung/relgeo_lengths.py --root <root> <carve> [<carve> ...] [--cpu]
        -> <carve>/relgeo/geo.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carves", nargs="+")
    ap.add_argument("--root", required=True, help="the experiment's root directory (fits/ live there)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp
    import geometry as G

    for carve in a.carves:
        out = os.path.join(carve, "relgeo", "geo.npz")
        if os.path.exists(out):
            print(f"exists: {out}", flush=True)
            continue
        t0 = time.time()
        meta = json.load(open(os.path.join(carve, "subset.meta.json")))
        man = json.load(open(os.path.join(meta["source"], "manifest.meta.json")))
        M = int(man["M"]) if str(man["M"]).isdigit() else 8          # the Euclidean source reuses M = 8
        ld = lambda f: np.load(os.path.join(carve, f))
        la, lb = ld("labels_a.npy"), ld("labels_b.npy")
        classes = np.intersect1d(np.unique(la), np.unique(lb))
        res = {"classes": classes, "steps": a.steps, "geometry": "ensemble_eol_samedec",
               "M": M, "models": np.array([man["model_a"], man["model_b"]])}
        ts = jnp.linspace(0.0, 1.0, a.steps + 1)
        for side, tag, Z, lab, Zt in (("a", man["model_a"], ld("Za.npy"), la, ld("Za_test.npy")),
                                      ("b", man["model_b"], ld("Zb.npy"), lb, ld("Zb_test.npy"))):
            anc = np.stack([Z[lab == c].mean(0) for c in classes]).astype(np.float32)
            phi = G.make_phi(G.load_heads(os.path.join(a.root, "fits", tag, "heads.npz"), M))

            @jax.jit
            def lengths(P, A):
                pts = P[:, None, None, :] + ts[None, None, :, None] * (A[None, :, None, :] - P[:, None, None, :])
                f = phi(pts)                                          # (c, K, T+1, M, G)
                sq = ((f[:, :, 1:] - f[:, :, :-1]) ** 2).sum(-1).mean(-1)
                return jnp.sqrt(sq + 1e-12).sum(-1)                   # (c, K)

            A = jnp.asarray(anc)
            def all_lengths(P):
                rows = []
                for s in range(0, len(P), a.chunk):
                    blk = jnp.asarray(P[s:s + a.chunk], jnp.float32)
                    pad = a.chunk - len(blk)
                    if pad:
                        blk = jnp.concatenate([blk, jnp.repeat(blk[-1:], pad, 0)])
                    rows.append(np.asarray(lengths(blk, A))[:a.chunk - pad])
                return np.concatenate(rows).astype(np.float64)
            res[f"geo_Z{side}"] = all_lengths(Zt)
            res[f"anchor_geo_{side}"] = all_lengths(anc)
            res[f"anchors_{side}"] = anc
            if not np.isfinite(res[f"geo_Z{side}"]).all():
                raise SystemExit(f"REFUSING: non-finite lengths on side {side} of {carve}")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, **res)
        print(f"wrote {out}: {len(classes)} anchors, {res['geo_Za'].shape[0]} test cells, M={M}, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
