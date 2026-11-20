#!/usr/bin/env python
"""Path lengths for the relative geodesics baseline (Yu et al. 2024), from the models' decoders.

For each side, the length of the straight latent segment from every test point to every
class-centroid anchor, measured as the same-decoder ensemble energy that defines NGMA's
`ensemble_eol_samedec` distance: E(i, j) = sum_t sqrt(mean_k ||f_k(x_{t+1}) - f_k(x_t)||^2).
Writes an .npz that `run_relgeo.py --geo-mode precomputed --geo-npz` reads, so the baseline
uses the same models and the same geometry as NGMA and differs only in what it does with
them (labelled class-centroid anchors instead of an unsupervised coupling).

  python baselines/relgeo_lengths.py \\
      --data-dir checkpoints/<root>/seed1/disjoint \\
      --checkpoint-a checkpoints/fmnist_vamp/vae_a/best_model.pickle \\
      --checkpoint-b checkpoints/fmnist_vamp/vae_b/best_model.pickle \\
      --out checkpoints/<root>/alignments_baselines/seed1/disjoint/relgeo_jax/geo.npz
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def class_centroids(Z: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class means in np.unique(labels) order, the order run_relgeo.py uses. With an
    attribute matrix (CelebA), one anchor per attribute in column order."""
    if labels.ndim == 2:
        lo, hi = labels.min(), labels.max(); mid = (lo + hi) / 2.0
        return (np.stack([Z[labels[:, j] > mid].mean(axis=0) for j in range(labels.shape[1])]),
                np.arange(labels.shape[1]))
    classes = np.unique(labels)
    return np.stack([Z[labels == c].mean(axis=0) for c in classes]), classes


def samedec_line_lengths(Zq: np.ndarray, Za: np.ndarray, decode_ensemble, n_steps: int = 30,
                         chunk: int = 64) -> np.ndarray:
    """(n_q, n_a) same-decoder ensemble straight-line lengths from every query to every anchor.

    decode_ensemble: (m, d) -> (m, K, d_x). Chunked over (query, anchor) pairs so the peak is
    chunk * (n_steps+1) * K * d_x decoder outputs.
    """
    import jax
    import jax.numpy as jnp

    Zq = np.asarray(Zq, dtype=np.float32)
    Za = np.asarray(Za, dtype=np.float32)
    n_q, d = Zq.shape
    n_a = Za.shape[0]
    ts = jnp.linspace(0.0, 1.0, n_steps + 1)

    @jax.jit
    def _lengths(z1, z2):                                   # (c, d), (c, d) -> (c,)
        path = z1[:, None, :] + ts[None, :, None] * (z2 - z1)[:, None, :]   # (c, T+1, d)
        x = decode_ensemble(path.reshape(-1, d))            # (c*(T+1), K, d_x)
        x = x.reshape(path.shape[0], n_steps + 1, x.shape[1], -1)
        diff = x[:, 1:] - x[:, :-1]                         # (c, T, K, d_x)
        step = jnp.sqrt((diff ** 2).sum(-1).mean(-1) + 1e-10)   # mean over K, then sqrt
        return step.sum(-1)

    qi, ai = np.meshgrid(np.arange(n_q), np.arange(n_a), indexing="ij")
    qi, ai = qi.ravel(), ai.ravel()
    out = np.empty(n_q * n_a, dtype=np.float64)
    for s in range(0, len(qi), chunk):
        e = min(s + chunk, len(qi))
        out[s:e] = np.asarray(_lengths(jnp.asarray(Zq[qi[s:e]]), jnp.asarray(Za[ai[s:e]])))
    return out.reshape(n_q, n_a)


def _load_model(checkpoint: str, z_dim: int, num_decoders: int, model_class: str):
    import jax
    from flax import nnx
    from ngma.utils import load_obj
    model = load_obj(model_class)(opts={"z_dim": z_dim, "num_decoders": num_decoders},
                                  rngs=nnx.Rngs(jax.random.PRNGKey(0)))
    model.load(checkpoint)      # the checkpoint's own options win (learn_sigma, prior, ...)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="the carve: Za, labels_a, Za_test, ...")
    ap.add_argument("--checkpoint-a", required=True)
    ap.add_argument("--checkpoint-b", required=True)
    ap.add_argument("--out", required=True, help=".npz for run_relgeo.py --geo-npz")
    ap.add_argument("--z-dim", type=int, default=16)
    ap.add_argument("--num-decoders", type=int, default=8)
    ap.add_argument("--model-class", default="ngma.models.vae.VAE",
                    help="ngma.models.vae.VAE (MNIST, FashionMNIST) or "
                         "ngma.models.celeba_vae_ilms.VAE (CelebA, with --z-dim 32)")
    ap.add_argument("--steps", type=int, default=30, help="segment discretisation")
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()

    ld = lambda n: np.load(os.path.join(args.data_dir, n))
    Za, Zb = ld("Za.npy"), ld("Zb.npy")
    la, lb = ld("labels_a.npy"), ld("labels_b.npy")
    Za_test, Zb_test = ld("Za_test.npy"), ld("Zb_test.npy")

    anchors_a, cls_a = class_centroids(Za, la)
    anchors_b, cls_b = class_centroids(Zb, lb)
    if not np.array_equal(cls_a, cls_b):
        raise ValueError(f"class sets differ between sides: {cls_a} vs {cls_b}")

    out = {"classes": cls_a, "steps": args.steps, "geometry": "ensemble_eol_samedec"}
    for side, ckpt, Zt, anc in (("a", args.checkpoint_a, Za_test, anchors_a),
                                ("b", args.checkpoint_b, Zb_test, anchors_b)):
        logging.info(f"side {side}: loading {ckpt}")
        model = _load_model(ckpt, args.z_dim, args.num_decoders, args.model_class)
        logging.info("  side %s: prior=%s z_dim=%d", side,
                     getattr(model.opts, "prior", "standard"), model.opts.z_dim)
        if int(model.opts.z_dim) != int(args.z_dim):
            raise SystemExit(
                f"side {side}: checkpoint has z_dim={model.opts.z_dim} but --z-dim says "
                f"{args.z_dim}. Pass the pair's architecture: CelebA is "
                f"--z-dim 32 --model-class ngma.models.celeba_vae_ilms.VAE.")
        dec = model.decode_ensemble
        logging.info(f"side {side}: {Zt.shape[0]} test x {anc.shape[0]} anchors, {args.steps} steps")
        out[f"geo_Z{side}"] = samedec_line_lengths(Zt, anc, dec, args.steps, args.chunk)
        out[f"anchor_geo_{side}"] = samedec_line_lengths(anc, anc, dec, args.steps, args.chunk)
        out[f"anchors_{side}"] = anc

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    logging.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
