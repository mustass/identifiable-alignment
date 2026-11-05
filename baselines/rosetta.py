#!/usr/bin/env python
"""Wasserstein Procrustes with geometric initialisation (Schnaus et al. 2026, arXiv 2610.09411),
or the official mini-vec2vec (Dar 2025, arXiv 2510.02348), on one carve, through the authors'
package `unpaired_rosetta` (MIT, github.com/dominik-schnaus/unpaired-rosetta), pinned to one
commit. The package needs torch, numpy, scipy, scikit-learn, pylibmgm and tqdm and never
imports JAX, so it lives in its own environment:

    conda create -n rosetta python=3.12 -y
    $ENV/bin/pip install "git+https://github.com/dominik-schnaus/unpaired-rosetta@fdfad84ea488f29702a61a455be71622c4ff5d49"

Wasserstein Procrustes, with the package's defaults (each one a flag here):
  0. each side is centred on its own fitting mean and unit-normalised; the test vectors get
     the same treatment
  1. S = 30 restarts of { k-means (C = 30 clusters) on b points a side; match the centres by
     the permutation that maximises their linear CKA }; the matched clusterings are averaged
     into one cross-covariance
  2. read-out: one conditional-gradient step on GW from the averaged plan, then Procrustes
  3. R = 100 refinement steps: Hungarian matching of a batch of X W against a batch of Y, then
     Procrustes
With b = 10,000 and carves of 3,000 (CelebA 7,500) points a side, every batch is the whole carve.
`--method mini_vec2vec_official` runs the package's port of mini-vec2vec with its defaults.

Fit on the carve's Za / Zb, scored on Za_test / Zb_test (row i matches row i): R@1 as the
exact-instance argmin, the tie-averaged R@1 and CSLS of `baseline_metrics`, both directions
(W is orthogonal, so b -> a uses W^T), FOSCTTM from the package, 1-NN label accuracy, and the
in-sample orthogonal map on the test pairs under the same preprocessing (the ceiling of this
map class, not a method).

    $ENV/bin/python baselines/rosetta.py --carve checkpoints/<root>/seed1/disjoint \\
        --seed 1 --out results/rosetta_wp/<root>_s1.json
"""
import argparse
import json
import os
import sys
import time
from importlib import metadata

import numpy as np
from scipy.linalg import orthogonal_procrustes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baseline_metrics import retrieval_and_csls  # noqa: E402

METHODS = {"wasserstein_procrustes": "WassersteinProcrustes", "mini_vec2vec_official": "MiniVec2Vec"}


def procrustes(X, Y):
    """The orthogonal W minimising ||X W - Y||_F."""
    return orthogonal_procrustes(np.asarray(X, np.float64), np.asarray(Y, np.float64))[0]


def r_at_1(Q, R, chunk=2048):
    """Exact-instance R@1 on the full split: row i of Q retrieves row i of R, nearest by
    Euclidean distance."""
    rn = (R.astype(np.float64) ** 2).sum(1)
    hits = 0
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk].astype(np.float64)
        hits += int(((rn[None, :] - 2.0 * q @ R.T.astype(np.float64)).argmin(1) == np.arange(i, i + len(q))).sum())
    return hits / len(Q)


def package_record():
    """Name, version and pinned commit of the installed package (a git install records the
    commit in pip's direct_url.json)."""
    import unpaired_rosetta
    dist = metadata.distribution("unpaired-rosetta")
    rec = {"name": "unpaired-rosetta", "version": dist.version, "path": os.path.dirname(unpaired_rosetta.__file__)}
    try:
        rec["commit"] = json.loads(dist.read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id")
    except (FileNotFoundError, json.JSONDecodeError):
        rec["commit"] = None
    return rec


def build(method, a):
    import torch  # noqa: F401  (the package's tensors)
    if method == "wasserstein_procrustes":
        from unpaired_rosetta import WassersteinProcrustes
        return WassersteinProcrustes(num_clusters=a.num_clusters, num_restarts=a.num_restarts,
                                     batch_size=a.batch_size, num_iterations=a.num_iterations,
                                     num_workers=a.num_workers, num_threads=a.num_threads)
    from unpaired_rosetta.baselines.mini_vec2vec import MiniVec2Vec
    return MiniVec2Vec()


def nn_label_acc(P, Y, lab, block=2048):
    """Fraction of rows of P whose nearest row of Y carries the same label (row i pairs with row i)."""
    yy = (Y ** 2).sum(1)
    hit = 0
    for s in range(0, len(P), block):
        D = yy[None] - 2.0 * P[s:s + block] @ Y.T
        hit += int((lab[D.argmin(1)] == lab[s:s + block]).sum())
    return hit / len(P)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carve", required=True, help="the disjoint carve: Za, Zb, Za_test, Zb_test")
    ap.add_argument("--seed", type=int, default=1, help="the draw; also the package's randomness seed")
    ap.add_argument("--method", default="wasserstein_procrustes", choices=sorted(METHODS))
    ap.add_argument("--num-clusters", type=int, default=30, help="C")
    ap.add_argument("--num-restarts", type=int, default=30, help="S")
    ap.add_argument("--batch-size", type=int, default=10_000, help="b")
    ap.add_argument("--num-iterations", type=int, default=100, help="R")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from unpaired_rosetta.evaluation import retrieval_metrics
    from unpaired_rosetta.linalg import center_and_normalize
    from unpaired_rosetta.randomness import Randomness

    g = lambda f: np.load(os.path.join(a.carve, f)).astype(np.float32)
    Za, Zb, Za_te, Zb_te = g("Za.npy"), g("Zb.npy"), g("Za_test.npy"), g("Zb_test.npy")
    T = torch.from_numpy

    t0 = time.time()
    aligner = build(a.method, a).fit(T(Za), T(Zb), randomness=a.seed)
    seconds = round(time.time() - t0, 1)

    with torch.inference_mode():
        Xe = center_and_normalize(T(Za_te), aligner.mean_x).double().numpy()
        Ye = center_and_normalize(T(Zb_te), aligner.mean_y).double().numpy()
        W = aligner.weight.double().numpy()
    fos = retrieval_metrics(aligner, T(Za_te), T(Zb_te), Randomness(a.seed))
    m_ab = {"r_at1": r_at_1(Xe @ W, Ye), **retrieval_and_csls(Xe @ W, Ye), **fos}
    m_ba = {"r_at1": r_at_1(Ye @ W.T, Xe), **retrieval_and_csls(Ye @ W.T, Xe)}
    Wc = procrustes(Xe, Ye)
    ceil = {"r_at1": r_at_1(Xe @ Wc, Ye), **retrieval_and_csls(Xe @ Wc, Ye)}
    lp = os.path.join(a.carve, "labels_test.npy")
    if os.path.exists(lp):
        lab = np.load(lp)
        if lab.ndim == 1:
            m_ab["class_top1"] = nn_label_acc(Xe @ W, Ye, lab)
            m_ba["class_top1"] = nn_label_acc(Ye @ W.T, Xe, lab)
            ceil["class_top1"] = nn_label_acc(Xe @ Wc, Ye, lab)
    stages = {"seconds": seconds,
              "orthogonality_error": float(np.abs(W.T @ W - np.eye(len(W))).max())}

    res = {"carve": os.path.abspath(a.carve), "seed": a.seed, "method": a.method,
           "package": package_record(), "args": vars(a), "eval_n": int(len(Za_te)),
           "pools": {"carve": {"n_per_side": int(len(Za)), "source": "carve (validation images)",
                               "a_to_b": m_ab, "b_to_a": m_ba, "stages": stages,
                               "orthogonal_ceiling_in_sample_NOT_the_method": ceil, "seconds": seconds}}}
    print(f"{a.method} on {a.carve} (n={len(Za)}/side): R@1 a->b {m_ab['r_at1']:.4f}  b->a {m_ba['r_at1']:.4f}"
          f"  FOSCTTM {fos['foscttm']:.4f}"
          f"  | orthogonal ceiling {ceil['r_at1']:.4f} | {seconds:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
