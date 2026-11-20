#!/usr/bin/env python
"""Cut one sample pair (a carve) out of a source.

A source holds one pool of images encoded by both models, row i the same image on both sides,
with the pairwise distance matrices and the metric tensors of every pool point (prepare_sources.py).
A carve chooses an index set per side:

    disjoint   the two sides share no image             (the regime of the paper's tables)
    paired     the same images on both sides            (true pairs, for the Procrustes baseline
                                                         and the anchors of LFM)
    frac       exactly --overlap of each side is shared (--overlap 1.0 gives the same images in
                                                         independent orders, the carve the
                                                         supervised ceilings are fitted on)

The draw depends on (regime, n, seed, pool size) only, so two model pairs carved with the same
arguments see the same images. Written to --out:

    Za.npy Zb.npy labels_a.npy labels_b.npy     the carved rows
    C_<mode>_a.npy C_<mode>_b.npy               the carved distance matrices, every mode of the source
    G_ensemble_eol_samedec_{a,b}.npy            the carved metric tensors
    Za_test.npy Zb_test.npy labels_test.npy     the paired test split, copied unchanged
    gt_ab.npy                                   for row i of side a, the row of side b holding the
                                                same image, or -1
    subset.meta.json                            the draw

    PYTHONPATH=src python scripts/carve.py --src sources/fmnist_gauss_ab --out carves/fmnist_gauss_ab/seed1/disjoint \\
        --regime disjoint --n 3000 --seed 1
"""
import argparse
import json
import os
import shutil

import numpy as np


def draw(regime, N, n, seed, frac=None):
    """The two index sets of one regime."""
    if regime == "frac":
        if frac is None or not (0.0 <= frac <= 1.0):
            raise SystemExit("--regime frac needs --overlap in [0, 1]")
        m = int(round(frac * n))                       # images present on both sides
        need = 2 * n - m                               # distinct images required
        if need > N:
            raise SystemExit(f"overlap={frac} at n={n} needs {need} distinct images, the source has {N}")
        rng = np.random.default_rng(seed * 1000 + 7717)
        perm = rng.permutation(N)
        shared, a_only, b_only = perm[:m], perm[m:m + (n - m)], perm[m + (n - m):need]
        a = np.concatenate([shared, a_only])
        b = np.concatenate([shared, b_only])
        # each side is shuffled on its own, so the shared images sit at unrelated positions
        return rng.permutation(a), rng.permutation(b)
    if regime == "paired":
        idx = np.random.default_rng(seed * 1000).choice(N, size=n, replace=False)
        return idx, idx
    if regime == "disjoint":
        if 2 * n > N:
            raise SystemExit(f"disjoint needs 2n={2 * n} points, the source has {N}")
        perm = np.random.default_rng(seed * 1000).permutation(N)
        return perm[:n], perm[n:2 * n]
    raise SystemExit(f"unknown regime {regime!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="a source directory (prepare_sources.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regime", required=True, choices=["disjoint", "paired", "frac"])
    ap.add_argument("--overlap", type=float, default=None, help="--regime frac: shared fraction of each side")
    ap.add_argument("--n", type=int, required=True, help="points per side")
    ap.add_argument("--seed", type=int, default=1, help="the draw")
    args = ap.parse_args()

    Za = np.load(os.path.join(args.src, "Za.npy"))
    Zb = np.load(os.path.join(args.src, "Zb.npy"))
    N = len(Za)
    if len(Zb) != N:
        raise SystemExit(f"Za has {N} rows, Zb has {len(Zb)}: not a source")
    if args.n > N:
        raise SystemExit(f"--n {args.n} exceeds the {N} points of the source")
    modes = sorted(f[2:-6] for f in os.listdir(args.src) if f.startswith("C_") and f.endswith("_a.npy"))
    if not modes:
        raise SystemExit(f"no C_<mode>_a.npy in {args.src}")
    if os.path.exists(os.path.join(args.out, "subset.meta.json")):
        print(f"exists: {args.out}")
        return

    ia, ib = draw(args.regime, N, args.n, args.seed, args.overlap)
    shared = int(np.intersect1d(ia, ib).size)
    os.makedirs(args.out, exist_ok=True)
    print(f"{args.src}: {N} points -> {args.regime} n={args.n} seed={args.seed}, "
          f"{shared} images shared ({100.0 * shared / args.n:.1f}%), modes {', '.join(modes)}")

    pos_in_b = {int(v): j for j, v in enumerate(ib)}
    gt = np.array([pos_in_b.get(int(v), -1) for v in ia], dtype=np.int32)
    np.save(os.path.join(args.out, "gt_ab.npy"), gt)

    np.save(os.path.join(args.out, "Za.npy"), Za[ia])
    np.save(os.path.join(args.out, "Zb.npy"), Zb[ib])
    for name, idx in (("labels_a.npy", ia), ("labels_b.npy", ib)):
        src = os.path.join(args.src, name)
        if os.path.exists(src):
            np.save(os.path.join(args.out, name), np.load(src)[idx])
    for mode in modes:
        for side, idx in (("a", ia), ("b", ib)):
            C = np.load(os.path.join(args.src, f"C_{mode}_{side}.npy"), mmap_mode="r")
            np.save(os.path.join(args.out, f"C_{mode}_{side}.npy"), np.asarray(C[np.ix_(idx, idx)]))
    for side, idx in (("a", ia), ("b", ib)):
        g = os.path.join(args.src, f"G_ensemble_eol_samedec_{side}.npy")
        if os.path.exists(g):
            np.save(os.path.join(args.out, f"G_ensemble_eol_samedec_{side}.npy"), np.load(g)[idx])
    for name in ("Za_test.npy", "Zb_test.npy", "labels_test.npy"):
        src = os.path.join(args.src, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, name))

    json.dump({"source": os.path.abspath(args.src), "regime": args.regime, "overlap_requested": args.overlap,
               "overlap_actual": shared / float(args.n), "n": args.n, "seed": args.seed, "source_n": int(N),
               "modes": modes, "shared_images": shared, "idx_a": ia.tolist(), "idx_b": ib.tolist()},
              open(os.path.join(args.out, "subset.meta.json"), "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
