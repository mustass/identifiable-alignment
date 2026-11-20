#!/usr/bin/env python
"""Prepare the Tabula Sapiens lung cells for the scVI experiment.

Inputs (--src): `lung_train.h5ad`, 65,847 lung cells x 3,000 genes of raw counts from four
donors, and `ts_lung_full.h5ad`, the full Tabula Sapiens lung file, from which the labels
(`donor_id`, `method`, `cell_type`, `compartment`) are joined by barcode. The 62,954 10x 3'
v3.1 cells are kept; the Smart-seq2 cells are dropped. No batch covariate is used later, so
the four donors are one data distribution.

Split, by cell with one seed, so every donor appears on both sides and no cell on two:

    eval     10,000   encoded by both models, scored with known pairs
    shared    4,500   pool of cells both models encode
    only_a    4,500   pool of cells only the model in role a encodes
    only_b    4,500   pool of cells only the model in role b encodes (unused)
    train    the remaining cells, cut into halves H1 and H2, each with nested subsets of
             4,900 < 9,800 < 19,600 cells (the training-size axis)

Outputs (--root): `lung10x.h5ad` (the 10x cells with their labels), `split.npz` (the index
sets above), `split_meta.json` (counts, the median library size l_ref) and `labels.npz`
(integer codes and names of cell type, compartment and donor, coded in sorted name order).

    python experiments/lung/prep.py --src <dir with the two h5ad files> --root <root>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

SEED = 0
N_EVAL, N_POOL = 10_000, 4_500
SIZES = (4_900, 9_800, 19_600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding lung_train.h5ad and ts_lung_full.h5ad")
    ap.add_argument("--root", required=True, help="the experiment's root directory (outputs)")
    a = ap.parse_args()
    import anndata as ad
    src, out = Path(a.src), Path(a.root)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    A = ad.read_h5ad(src / "lung_train.h5ad")
    full = ad.read_h5ad(src / "ts_lung_full.h5ad", backed="r")
    obs = full.obs.loc[A.obs_names, ["donor_id", "method", "cell_type", "compartment"]].copy()
    full.file.close()
    for c in obs.columns:
        A.obs[c] = obs[c].astype(str).values
    print(f"loaded and labelled {A.n_obs} cells ({time.time() - t0:.0f}s)", flush=True)

    A = A[A.obs["method"] == "10X"].copy()
    x = A.X.data
    if not np.allclose(x, np.round(x)) or x.min() < 0:
        raise SystemExit("REFUSING: X is not raw counts")
    A.X = A.X.astype(np.float32)
    A.obs = A.obs[["donor_id", "cell_type", "compartment"]]
    n = A.n_obs
    print(f"10x cells: {n}; donors {dict(A.obs['donor_id'].value_counts())}", flush=True)

    perm = np.random.default_rng(SEED).permutation(n)
    cut = np.cumsum([N_EVAL, N_POOL, N_POOL, N_POOL])
    idx = {"eval": perm[:cut[0]], "shared": perm[cut[0]:cut[1]],
           "only_a": perm[cut[1]:cut[2]], "only_b": perm[cut[2]:cut[3]]}
    train = perm[cut[3]:]
    half = len(train) // 2
    if half < SIZES[-1]:
        raise SystemExit(f"REFUSING: halves of {half} cells are smaller than {SIZES[-1]}")
    for h, part in (("H1", train[:half]), ("H2", train[half:2 * half])):
        for s in SIZES:
            idx[f"{h}_{s}"] = part[:s]                      # nested: a prefix of the half
    all_sets = [v for k, v in idx.items() if not k.startswith("H")] + [train[:half], train[half:2 * half]]
    flat = np.concatenate(all_sets)
    assert len(flat) == len(set(flat.tolist())), "pools and halves overlap"

    A.write_h5ad(out / "lung10x.h5ad")
    np.savez(out / "split.npz", **{k: np.sort(v) for k, v in idx.items()})
    labels = {}
    for col, key in (("cell_type", "cell_type"), ("compartment", "compartment"), ("donor_id", "donor")):
        names, codes = np.unique(A.obs[col].astype(str).values, return_inverse=True)
        labels[key], labels[f"{key}_names"] = codes.astype(np.int64), names
    np.savez(out / "labels.npz", **labels)
    lib = np.asarray(A.X.sum(1)).ravel()
    meta = {"n_cells": int(n), "n_genes": int(A.n_vars), "seed": SEED, "sizes": list(SIZES),
            "counts": {k: int(len(v)) for k, v in idx.items()},
            "l_ref_median_library": float(np.median(lib)),
            "donors": {k: int(v) for k, v in A.obs["donor_id"].value_counts().items()},
            "n_cell_types": int(A.obs["cell_type"].nunique())}
    json.dump(meta, open(out / "split_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
