"""Relative geodesic representations (Yu et al. 2024) on one carve.

Each test point is represented by its path lengths to the class-centroid anchors of its own
side, under the decoder's pullback geometry, divided by the spread of the inter-anchor
lengths. The lengths come from `relgeo_lengths.py` (--geo-npz), computed with the same
decoders and the same same-decoder ensemble energy as NGMA's distance. The two sides are then
compared directly: R@1 and R@5 of the true partner, the CSLS block of `baseline_metrics`, and
1-NN label accuracy, in both directions.

Reads from <data-dir>: Za.npy, Zb.npy, labels_a.npy, labels_b.npy, Za_test.npy, Zb_test.npy,
labels_test.npy. Writes <out-dir>/results.csv with one row per direction. Needs numpy and
scikit-learn only.

    python baselines/run_relgeo.py --data-dir checkpoints/<root>/seed1/disjoint \\
        --out-dir checkpoints/<root>/alignments_baselines/seed1/disjoint/relgeo_jax \\
        --geo-mode precomputed --geo-npz checkpoints/<root>/alignments_baselines/seed1/disjoint/relgeo_jax/geo.npz \\
        --seed 1
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_metrics import retrieval_and_csls  # noqa: E402


def _class_centroids(Z: np.ndarray, labels: np.ndarray, classes=None) -> np.ndarray:
    if labels.ndim == 2:                       # multi-attribute labels (CelebA)
        return _attribute_anchors(Z, labels)
    classes = np.unique(labels) if classes is None else classes
    return np.stack([Z[labels == c].mean(axis=0) for c in classes])


def _attribute_anchors(Z, labels):
    """Multi-attribute labels (CelebA): one anchor per attribute, the mean of the images that
    have it, in column order."""
    lo, hi = labels.min(), labels.max(); mid = (lo + hi) / 2.0
    return np.stack([Z[labels[:, j] > mid].mean(axis=0) for j in range(labels.shape[1])])


def _attribute_knn_accuracy(Z_query, Z_ref, labels_query, labels_ref, k):
    """Per-attribute majority vote over the k nearest references, mean agreement with the
    true attribute vector over points and attributes."""
    sq = (Z_query ** 2).sum(1)[:, None] + (Z_ref ** 2).sum(1)[None] - 2.0 * Z_query @ Z_ref.T
    nn = np.argsort(np.maximum(sq, 0.0), axis=1)[:, :k]
    lo, hi = labels_ref.min(), labels_ref.max(); mid = (lo + hi) / 2.0
    vote = np.where(labels_ref[nn].mean(axis=1) >= mid, hi, lo)
    return float((vote == labels_query).mean())


def _instance_retrieval(A: np.ndarray, B: np.ndarray, ks=(1, 5)) -> dict:
    """R@k in the relative representation: row i of A must retrieve row i of B. Ties are
    rank-averaged, so a representation whose rows are all equal scores at chance."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = len(A)
    out = {f"retrieval_at_{k}": 0 for k in ks}
    tied = 0
    step = 256
    for i in range(0, n, step):
        d = ((A[i:i + step, None, :] - B[None]) ** 2).sum(-1)
        true = np.arange(i, min(i + step, n))
        dt = d[np.arange(len(true)), true][:, None]
        lower = (d < dt - 1e-12).sum(1)
        ties = (np.abs(d - dt) <= 1e-12).sum(1) - 1
        rank = lower + ties / 2.0
        tied += int((ties > 0).sum())
        for k in ks:
            out[f"retrieval_at_{k}"] += int((rank < k).sum())
    out = {k: v / n for k, v in out.items()}
    out["retrieval_tie_frac"] = tied / n
    out["retrieval_pool_n"] = float(n)
    return out


def _knn_accuracy(Z_query, Z_ref, labels_query, labels_ref, k):
    if np.asarray(labels_ref).ndim == 2:
        return _attribute_knn_accuracy(Z_query, Z_ref, np.asarray(labels_query), np.asarray(labels_ref), k)
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    clf.fit(Z_ref, labels_ref)
    return float(clf.score(Z_query, labels_query))


def main():
    parser = argparse.ArgumentParser(description="Relative geodesic representations baseline")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--geo-mode", default="precomputed", choices=["precomputed"],
                        help="the path lengths come from relgeo_lengths.py (--geo-npz)")
    parser.add_argument("--geo-npz", required=True, help="geo.npz from relgeo_lengths.py")
    parser.add_argument("--seed", type=int, default=0, help="the draw (the method is deterministic)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    Za = np.load(os.path.join(args.data_dir, "Za.npy")).astype(np.float32)
    Zb = np.load(os.path.join(args.data_dir, "Zb.npy")).astype(np.float32)
    labels_a = np.load(os.path.join(args.data_dir, "labels_a.npy"))
    labels_b = np.load(os.path.join(args.data_dir, "labels_b.npy"))
    Za_test = np.load(os.path.join(args.data_dir, "Za_test.npy")).astype(np.float32)
    Zb_test = np.load(os.path.join(args.data_dir, "Zb_test.npy")).astype(np.float32)
    labels_test = np.load(os.path.join(args.data_dir, "labels_test.npy"))

    d = Za.shape[1]
    # An attribute matrix gives one anchor per attribute, in column order, as relgeo_lengths.py
    # writes them. A class vector gives one anchor per class present on BOTH sides, since an
    # anchor needs both sides (a rare class can be missing from one side of a draw).
    class_order = (np.arange(labels_a.shape[1]) if labels_a.ndim == 2
                   else np.intersect1d(np.unique(labels_a), np.unique(labels_b)))
    n_classes = len(class_order)
    if labels_a.ndim == 1 and n_classes < len(np.union1d(labels_a, labels_b)):
        print(f"anchors on the {n_classes} classes present on both sides "
              f"(of {len(np.union1d(labels_a, labels_b))})")
    print(f"d={d}  classes={n_classes}  geo_mode={args.geo_mode}  "
          f"train: Za={Za.shape} Zb={Zb.shape}  test: {Za_test.shape}")

    print(f"Loading precomputed path lengths from {args.geo_npz} ...")
    npz = np.load(args.geo_npz)
    if not np.array_equal(npz["classes"], class_order):
        raise ValueError("class order in the npz does not match labels_a")
    geo_Za = npz["geo_Za"].astype(np.float32)
    geo_Zb = npz["geo_Zb"].astype(np.float32)
    if geo_Za.shape != (len(Za_test), n_classes) or geo_Zb.shape != (len(Zb_test), n_classes):
        raise ValueError(f"npz shapes {geo_Za.shape}/{geo_Zb.shape} do not match the test split")
    mask = ~np.eye(n_classes, dtype=bool)
    geo_Za /= (npz["anchor_geo_a"][mask].std() + 1e-8)
    geo_Zb /= (npz["anchor_geo_b"][mask].std() + 1e-8)
    geo_mode_str = "geo_" + str(npz["geometry"])

    top1_ab = _knn_accuracy(geo_Za, geo_Zb, labels_test, labels_test, k=1)
    top1_ba = _knn_accuracy(geo_Zb, geo_Za, labels_test, labels_test, k=1)
    ret_ab = _instance_retrieval(np.asarray(geo_Za), np.asarray(geo_Zb))
    ret_ba = _instance_retrieval(np.asarray(geo_Zb), np.asarray(geo_Za))
    ret_ab.update(retrieval_and_csls(np.asarray(geo_Za), np.asarray(geo_Zb)))
    ret_ba.update(retrieval_and_csls(np.asarray(geo_Zb), np.asarray(geo_Za)))
    print(f"A->B  R@1={ret_ab['retrieval_at_1']:.4f}  R@5={ret_ab['retrieval_at_5']:.4f}  "
          f"(pool {int(ret_ab['retrieval_pool_n'])}, tie_frac {ret_ab['retrieval_tie_frac']:.4f})")
    print(f"B->A  R@1={ret_ba['retrieval_at_1']:.4f}  R@5={ret_ba['retrieval_at_5']:.4f}")
    print(f"A->B  CSLS@1={ret_ab['csls_at1']:.4f} (plain {ret_ab['plain_at_1_csls_pool']:.4f} on the "
          f"same {int(ret_ab['csls_pool_n'])} rows, tie_frac {ret_ab['csls_tie_frac']:.4f})")

    print(f"\nA→B  class_acc={top1_ab:.4f}")
    print(f"B→A  class_acc={top1_ba:.4f}")

    results_path = os.path.join(args.out_dir, "results.csv")
    with open(results_path, "w", newline="") as f:
        w = csv.writer(f)
        # top1_accuracy is 1-NN label accuracy; `metric` names it so it is never read as retrieval
        cols = ["direction", "top1_accuracy", "geo_mode", "metric"] + sorted(ret_ab)
        w.writerow(cols)
        for d, t1, ret in (("a_to_b", top1_ab, ret_ab), ("b_to_a", top1_ba, ret_ba)):
            w.writerow([d, t1, geo_mode_str, "knn_class_accuracy"]
                       + [ret[c] for c in sorted(ret_ab)])

    print(f"Results → {results_path}")


if __name__ == "__main__":
    main()
