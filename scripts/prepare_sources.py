#!/usr/bin/env python
"""Encode one pool of images with two models and compute each side's geometry: a source.

The pool is a fixed draw from the dataset's validation split (the split is deterministic), the
same images on both sides, so row i of Za and Zb is one image through the two encoders. For
each side the script writes

    Za.npy / Zb.npy                       encoder means of the pool, (N, d)
    labels_a.npy / labels_b.npy           their class labels
    Za_test.npy / Zb_test.npy             encoder means of the whole test split, (n_test, d):
    labels_test.npy                       the true pairs that every alignment is scored on
    C_euclidean_{a,b}.npy                 Euclidean distances between the pool codes
    C_ensemble_eol_samedec_{a,b}.npy      lengths of the straight segments between pool codes under
                                          the decoder ensemble's expected pullback metric: the
                                          segment is cut into 32 steps and each step's length is the
                                          root mean over the decoders of the squared output change
    G_ensemble_eol_samedec_{a,b}.npy      the expected pullback metric tensor (1/K) sum_k J_k^T J_k at
                                          every pool code, (N, d, d)

Carves (carve.py) are index sets into this source. Every array has a `.meta.json` sidecar.

    PYTHONPATH=src python scripts/prepare_sources.py --dataset fmnist --data-root data/fmnist \\
        --vae-a checkpoints/fmnist_gauss/vae_a/best_model.pickle \\
        --vae-b checkpoints/fmnist_gauss/vae_b/best_model.pickle \\
        --n-points 6000 --out sources/fmnist_gauss_ab
"""
import argparse
import functools
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import jax  # noqa: E402
from flax import nnx  # noqa: E402

from ngma import artifacts  # noqa: E402
from ngma.data import get_arrays  # noqa: E402
from ngma.data.utils import subsample  # noqa: E402
from ngma.geometry import compute_distance_matrix  # noqa: E402
from ngma.geometry.metric_tensors import expected_metric_tensors, spd_cholesky  # noqa: E402
from ngma.utils import load_obj, set_seed  # noqa: E402

MODEL_CLASS = {"mnist": "ngma.models.vae.VAE", "fmnist": "ngma.models.vae.VAE",
               "celeba": "ngma.models.celeba_vae_ilms.VAE"}
Z_DIM = {"mnist": 16, "fmnist": 16, "celeba": 32}
ENERGY = {"n_steps": 32, "row_block": 4, "col_chunk": 32}
MODES = ["euclidean", "ensemble_eol_samedec"]
BATCH = 512


def load_model(path, dataset, rng):
    """A model of the dataset's architecture, with the checkpoint's own options and weights."""
    model = load_obj(MODEL_CLASS[dataset])(opts={"z_dim": Z_DIM[dataset], "num_decoders": 8, "learn_sigma": True},
                                           rngs=nnx.Rngs(rng))
    model.load(path)
    model.eval()
    logging.info("  %s: prior=%s z_dim=%d decoders=%d", path, getattr(model.opts, "prior", "standard"),
                 model.opts.z_dim, model.opts.num_decoders)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(MODEL_CLASS))
    ap.add_argument("--data-root", required=True, help="dataset directory (torchvision root, or the CelebA .npy files)")
    ap.add_argument("--vae-a", required=True, help="checkpoint of model a (best_model.pickle)")
    ap.add_argument("--vae-b", required=True, help="checkpoint of model b")
    ap.add_argument("--n-points", type=int, required=True, help="pool size: 6000 on MNIST and FashionMNIST, 15000 on CelebA")
    ap.add_argument("--seed", type=int, default=1, help="the pool draw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--col-chunk", type=int, default=ENERGY["col_chunk"],
                    help="columns per decoder call of the segment-length kernel; lower it if the GPU runs out of memory")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    rng = jax.random.PRNGKey(args.seed)

    logging.info("loading models")
    vae_a = load_model(args.vae_a, args.dataset, rng)
    vae_b = load_model(args.vae_b, args.dataset, jax.random.split(rng)[1])
    artifacts.write_meta(os.path.join(args.out, "source"), {
        "kind": "source", "dataset": args.dataset, "seed": args.seed, "n_points": args.n_points,
        "vae_a": artifacts.file_fingerprint(args.vae_a), "vae_b": artifacts.file_fingerprint(args.vae_b),
        "energy": dict(ENERGY, col_chunk=args.col_chunk), "modes": MODES})

    _, _, val_images, val_labels, test_images, test_labels = get_arrays(args.dataset, args.data_root, val_fraction=0.1)
    imgs, labels = subsample([val_images, val_labels], args.n_points, seed=args.seed * 1000)

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode(model, x):
        _, z_mu, _ = model.encode(x, reparam=False)
        return z_mu

    def encode_all(model, images):
        return np.concatenate([np.array(encode(model, jax.device_put(images[i:i + BATCH])))
                               for i in range(0, len(images), BATCH)])

    def have(name):
        return os.path.exists(os.path.join(args.out, name))

    def save(name, arr, **meta):
        artifacts.save_npy(os.path.join(args.out, name), arr, meta=meta)
        logging.info("  saved %s %s", name, tuple(arr.shape))

    if have("Za.npy") and have("Zb.npy"):
        Za, Zb = np.load(os.path.join(args.out, "Za.npy")), np.load(os.path.join(args.out, "Zb.npy"))
        logging.info("pool codes on disk, loaded")
    else:
        logging.info("encoding the pool (%d images)", args.n_points)
        Za, Zb = encode_all(vae_a, imgs), encode_all(vae_b, imgs)
        save("Za.npy", Za, side="a"); save("Zb.npy", Zb, side="b")
    for name in ("labels_a.npy", "labels_b.npy"):
        if not have(name):
            save(name, labels.astype(np.int32))
    if not (have("Za_test.npy") and have("Zb_test.npy") and have("labels_test.npy")):
        logging.info("encoding the test split (%d images)", len(test_images))
        save("Za_test.npy", encode_all(vae_a, test_images), side="a", split="test")
        save("Zb_test.npy", encode_all(vae_b, test_images), side="b", split="test")
        save("labels_test.npy", test_labels.astype(np.int32), split="test")

    kw = dict(ENERGY, col_chunk=args.col_chunk)
    for mode in MODES:
        for side, Z, vae in (("a", Za, vae_a), ("b", Zb, vae_b)):
            name = f"C_{mode}_{side}.npy"
            if have(name):
                logging.info("%s on disk, skipped", name)
                continue
            t0 = time.time()
            logging.info("computing %s (n=%d)", name, len(Z))
            ck = os.path.join(args.out, f"ckpt_{mode}_{side}.npz") if mode != "euclidean" else None
            C = compute_distance_matrix(Z, mode=mode, model=vae, checkpoint_path=ck, **kw)
            save(name, C, mode=mode, side=side, kind="distance", **(kw if mode != "euclidean" else {}))
            logging.info("  %.0f s", time.time() - t0)

    for side, Z, vae in (("a", Za, vae_a), ("b", Zb, vae_b)):
        name = f"G_ensemble_eol_samedec_{side}.npy"
        if have(name):
            logging.info("%s on disk, skipped", name)
            continue
        t0 = time.time()
        logging.info("computing %s at %d points", name, len(Z))
        G = expected_metric_tensors(vae.decode_single_ensemble, Z)
        _, stats = spd_cholesky(G)
        save(name, G.astype(np.float32), mode="ensemble_eol_samedec", side=side, kind="metric_tensors", stats=stats)
        logging.info("  %.0f s", time.time() - t0)
    logging.info("done: %s", args.out)


if __name__ == "__main__":
    main()
