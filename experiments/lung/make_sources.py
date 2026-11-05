#!/usr/bin/env python
"""Turn the per-model geometry into NGMA sources, one per (model pair, M).

A source is what scripts/carve.py cuts regimes from: one pool of cells encoded by both models,
so row i is the same cell on both sides. Every draw is then a choice of two index sets over that
pool, with the costs, metric tensors and labels sliced to match. The pool is the 9,000 held-out
cells of geometry.py; the test split is the 10,000 eval cells.

Pairs, for each training size s in split_meta.json:
    cross   H1_s_s0 vs H2_s_s0, H1_s_s1 vs H2_s_s1   models fitted to different cells
    within  H1_s_s0 vs H1_s_s1, H2_s_s0 vs H2_s_s1   models fitted to the same cells, seeds differ

Each source, <root>/sources/lung{s}_{kind}{r}_M{M}/, holds
    Za, Zb, Za_test, Zb_test           pool and eval codes of the two models
    C_ensemble_eol_samedec_{a,b}       symlinks to geometry/{tag}/C_M{M}.npy
    G_ensemble_eol_samedec_{a,b}       symlinks to geometry/{tag}/G_M{M}.npy
    labels_a, labels_b, labels_test    cell-type codes (labels.npz)
    manifest.meta.json                 sampling "source", the two model tags, M, T, K

M = "euc" is the Euclidean comparator, <root>/sources/lung{s}_{kind}{r}_Meuc/: the costs are
Euclidean distances between the latent codes, C_euclidean_{a,b} (computed once per model into
geometry/{tag}/C_euclidean.npy), and `fit_map.py --geometry euclidean` replaces the metric
tensors by the identity; the G links are there only because its loader opens them. The carve
draws depend on (regime, n, seed, pool size) alone, so its carves hold the same cells as the
pullback carves.

    python experiments/lung/make_sources.py --root <root> [--M 1,8,euc]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

MODE = "ensemble_eol_samedec"


def pairs(sizes):
    for s in sizes:
        yield f"lung{s}_cross1", f"H1_{s}_s0", f"H2_{s}_s0"
        yield f"lung{s}_cross2", f"H1_{s}_s1", f"H2_{s}_s1"
        yield f"lung{s}_within1", f"H1_{s}_s0", f"H1_{s}_s1"
        yield f"lung{s}_within2", f"H2_{s}_s0", f"H2_{s}_s1"


def link(src, dst):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the experiment's root directory")
    ap.add_argument("--M", default="1,8,euc")
    ap.add_argument("--sizes", default="", help="comma-separated subset of the training sizes")
    a = ap.parse_args()
    root = Path(a.root)
    meta = json.load(open(root / "split_meta.json"))
    split = np.load(root / "split.npz")
    lab = np.load(root / "labels.npz")["cell_type"]
    made, missing = 0, []
    sizes = [int(x) for x in a.sizes.split(",")] if a.sizes else meta["sizes"]
    for name, ta, tb in pairs(sizes):
        ga, gb = root / "geometry" / ta, root / "geometry" / tb
        if not ((ga / "geometry.json").exists() and (gb / "geometry.json").exists()):
            missing.append(name)
            continue
        ja, jb = json.load(open(ga / "geometry.json")), json.load(open(gb / "geometry.json"))
        if ja["pool_sets"] != jb["pool_sets"] or ja["N"] != jb["N"] or ja["n_pool_cap"] or jb["n_pool_cap"]:
            raise SystemExit(f"REFUSING: {ta} and {tb} were computed on different or capped pools")
        pool = np.concatenate([split[s] for s in ja["pool_sets"]])[:ja["N"]]
        for M in a.M.split(","):
            out = root / "sources" / f"{name}_M{M}"
            out.mkdir(parents=True, exist_ok=True)
            for side, g in (("a", ga), ("b", gb)):
                if M == "euc":
                    ce = g / "C_euclidean.npy"
                    if not ce.exists():
                        Z = np.load(g / "Z_pool.npy")
                        np.save(ce, cdist(Z, Z).astype(np.float32))
                    link(ce, out / f"C_euclidean_{side}.npy")
                    link(g / "G_M8.npy", out / f"G_{MODE}_{side}.npy")   # opened, then replaced by I
                else:
                    link(g / f"C_M{M}.npy", out / f"C_{MODE}_{side}.npy")
                    link(g / f"G_M{M}.npy", out / f"G_{MODE}_{side}.npy")
            np.save(out / "Za.npy", np.load(ga / "Z_pool.npy"))
            np.save(out / "Zb.npy", np.load(gb / "Z_pool.npy"))
            np.save(out / "Za_test.npy", np.load(ga / "Z_test.npy"))
            np.save(out / "Zb_test.npy", np.load(gb / "Z_test.npy"))
            np.save(out / "labels_a.npy", lab[pool])
            np.save(out / "labels_b.npy", lab[pool])
            np.save(out / "labels_test.npy", lab[split["eval"]])
            man = {"sampling": "source", "n_points": int(len(pool)),
                   "model_a": ta, "model_b": tb, "M": M, "T": ja["T"], "K": ja["K"],
                   "pool_sets": ja["pool_sets"], "geometry": "euclidean" if M == "euc" else MODE}
            json.dump(man, open(out / "manifest.meta.json", "w"), indent=1)
            made += 1
    print(f"{made} sources written to {root / 'sources'}"
          + (f"; geometry missing for {len(missing)} pairs: {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
