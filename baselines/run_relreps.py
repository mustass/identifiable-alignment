"""Relative representations (Moschella et al. 2022) on one carve.

Each test representation is re-expressed as its cosine similarities to one anchor per class,
the centroid of that side's own labelled training points (one anchor per attribute when the
labels are an attribute matrix). The two relative spaces share their coordinates, so the
test points are compared directly: R@1 and R@5 of the true partner, the CSLS block of
`baseline_metrics`, and 1-NN label accuracy, in both directions.

Reads from <data-dir>: Za.npy, Zb.npy, labels_a.npy, labels_b.npy, Za_test.npy, Zb_test.npy,
labels_test.npy (row i of the two test files is the same image). Writes
<out-dir>/results.csv with one row per direction. Needs numpy and scikit-learn only.

    python baselines/run_relreps.py --data-dir checkpoints/<root>/seed1/disjoint \\
        --out-dir checkpoints/<root>/alignments_baselines/seed1/disjoint/relreps --seed 1
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_metrics import retrieval_and_csls  # noqa: E402


def _class_centroids(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One centroid per unique label, sorted by label value; one per attribute when the
    labels are a multi-attribute matrix."""
    if labels.ndim == 2:
        return _attribute_anchors(Z, labels)
    classes = np.unique(labels)
    return np.stack([Z[labels == c].mean(axis=0) for c in classes])  # [K, d]


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


def _cosine_rel_reps(Z: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of Z to each anchor. Returns [N, K]."""
    Z_n = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    A_n = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)
    return Z_n @ A_n.T  # [N, K]


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


def _knn_accuracy(Z_query: np.ndarray, Z_ref: np.ndarray,
                  labels_query: np.ndarray, labels_ref: np.ndarray, k: int) -> float:
    if np.asarray(labels_ref).ndim == 2:
        return _attribute_knn_accuracy(Z_query, Z_ref, np.asarray(labels_query), np.asarray(labels_ref), k)
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    clf.fit(Z_ref, labels_ref)
    return float(clf.score(Z_query, labels_query))


def main():
    parser = argparse.ArgumentParser(description="Relative representations baseline")
    parser.add_argument("--data-dir", required=True,
                        help="Dir with Za/Zb/labels_a/labels_b/Za_test/Zb_test/labels_test .npy")
    parser.add_argument("--out-dir", required=True)
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
    n_classes = len(np.unique(labels_a))
    print(f"d={d}  classes={n_classes}  train: Za={Za.shape} Zb={Zb.shape}"
          f"  test: {Za_test.shape}")

    anchors_a = _class_centroids(Za, labels_a)   # [K, d]
    anchors_b = _class_centroids(Zb, labels_b)   # [K, d]
    print(f"Anchors: {anchors_a.shape} per space  (one centroid per class)")

    rel_Za_test = _cosine_rel_reps(Za_test, anchors_a)  # [N_test, K]
    rel_Zb_test = _cosine_rel_reps(Zb_test, anchors_b)  # [N_test, K]

    top1_ab = _knn_accuracy(rel_Za_test, rel_Zb_test, labels_test, labels_test, k=1)
    top1_ba = _knn_accuracy(rel_Zb_test, rel_Za_test, labels_test, labels_test, k=1)
    ret_ab = _instance_retrieval(rel_Za_test, rel_Zb_test)
    ret_ba = _instance_retrieval(rel_Zb_test, rel_Za_test)
    ret_ab.update(retrieval_and_csls(rel_Za_test, rel_Zb_test))
    ret_ba.update(retrieval_and_csls(rel_Zb_test, rel_Za_test))
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
        cols = ["direction", "top1_accuracy", "metric"] + sorted(ret_ab)
        w.writerow(cols)
        for d, t1, ret in (("a_to_b", top1_ab, ret_ab), ("b_to_a", top1_ba, ret_ba)):
            w.writerow([d, t1, "knn_class_accuracy"] + [ret[c] for c in sorted(ret_ab)])

    print(f"Results → {results_path}")


if __name__ == "__main__":
    main()
