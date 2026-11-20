#!/usr/bin/env python
"""Fit the scVI models of the lung experiment: two training halves x three sizes x two seeds.

The shipped Tabula Sapiens lung recipe (n_latent 20, three hidden layers of 128, layer norm in
encoder and decoder, gene dispersion, negative binomial, observed library size, 400 epochs),
without a batch covariate. A model is fitted to the cells of one nested training subset
`H{1,2}_{size}` of split.npz and nothing else.

Per model, in <root>/fits/{half}_{size}_s{seed}/:
    model/            scvi-tools save
    latents.npz       posterior means and variances of all 10x cells (pools, eval and train)
    theta.npy         gene dispersions exp(px_r)
    fit.json          ELBO on its training cells and on the eval cells, wall time

    python experiments/lung/fit_scvi.py --root <root> [--only H1_4900_s0,...] [--threads 8]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

HALVES, SEEDS = ("H1", "H2"), (0, 1)


def configs(sizes):
    return [f"{h}_{s}_s{seed}" for s in sizes for h in HALVES for seed in SEEDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the experiment's root directory")
    ap.add_argument("--only", default="", help="comma-separated tags, e.g. H1_4900_s0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max_epochs", type=int, default=400)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import anndata as ad
    import scvi

    root = Path(a.root)
    meta = json.load(open(root / "split_meta.json"))
    split = np.load(root / "split.npz")
    A = ad.read_h5ad(root / "lung10x.h5ad")
    tags = a.only.split(",") if a.only else configs(meta["sizes"])
    for tag in tags:
        half, size, seed = tag.split("_")[0], int(tag.split("_")[1]), int(tag.split("_s")[1])
        d = root / "fits" / tag
        if (d / "fit.json").exists():
            print(f"{tag} cached", flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        tr = A[split[f"{half}_{size}"]].copy()
        scvi.settings.seed = seed
        scvi.model.SCVI.setup_anndata(tr)
        model = scvi.model.SCVI(
            tr, n_latent=20, n_layers=3, n_hidden=128, dropout_rate=0.05,
            dispersion="gene", gene_likelihood="nb", use_observed_lib_size=True,
            use_batch_norm="none", use_layer_norm="both",
        )
        model.train(max_epochs=a.max_epochs, early_stopping=False, train_size=0.9,
                    batch_size=128, accelerator="cpu", devices=1,
                    plan_kwargs=dict(lr=1e-3), enable_progress_bar=False)
        model.to_device("cpu")
        mu, var = model.get_latent_representation(A, give_mean=True, return_dist=True)
        np.savez_compressed(d / "latents.npz", mu=np.asarray(mu, np.float64),
                            s2=np.asarray(var, np.float64))
        np.save(d / "theta.npy", torch.exp(model.module.px_r).detach().cpu().numpy().astype(np.float64))
        model.save(str(d / "model"), overwrite=True)
        ev = A[split["eval"]].copy()
        rec = dict(tag=tag, half=half, size=size, seed=seed,
                   elbo_train=float(model.get_elbo(tr)), elbo_eval=float(model.get_elbo(ev)),
                   secs=round(time.time() - t0, 1))
        json.dump(rec, open(d / "fit.json", "w"), indent=1)
        print(json.dumps(rec), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
