"""The Procrustes ceiling and the label- or anchor-informed baselines on a carve's latents.

Reads the latents NGMA aligned, `<cache-dir>/seed<S>/disjoint/` (Za, Zb, labels_a, labels_b,
Za_test, Zb_test, labels_test), and the paired carve of the same draw,
`<cache-dir>/seed<S>/paired/` (the same number of points, row i the same image on both sides),
and scores on the test split, row i of Za_test matching row i of Zb_test:

  Procrustes (n=...)   the orthogonal map with translation, O(d) + t, fitted on the paired
                       carve's true pairs: the paired ceiling of the paper's Table 1
  RelReps-Centroid     relative representations (Moschella et al. 2022) against the class
                       centroids of each side's own labelled points
  LFM-Unsupervised     latent functional maps (Fumero et al. 2024) with heat-kernel-signature
                       descriptors on a kNN graph, no supervision
  LFM-Labels           the same with one-hot class-label descriptors
  LFM-Anchors (k=50)   the same with descriptors from 50 paired anchors of the paired carve

Each row carries top-1 label accuracy of the nearest neighbour, R@1, and the CSLS block of
`baseline_metrics.retrieval_and_csls`. Labels may be a class vector or, for CelebA, an
attribute matrix; the label metrics switch accordingly.

    python baselines/cached_baselines.py --cache-dir checkpoints/fmnist_gauss_ab \\
        --seed 1 --label fmnist_gauss_ab --dataset fmnist --out-csv results/eval_baselines_fmnist_gauss_ab_s1.csv
"""
import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, kneighbors_graph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_metrics import CSLS_FIELDS, retrieval_and_csls  # noqa: E402

logging.basicConfig(format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                    level=logging.INFO, stream=sys.stdout)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _top1(Za_q, Zb_ref, labels_q, labels_ref):
    """Label accuracy of the mapped query in the reference space. A class vector gives 1-NN
    label accuracy; an attribute matrix (CelebA) gives the mean per-attribute accuracy of a
    logistic probe fitted in the reference space."""
    labels_ref = np.asarray(labels_ref)
    labels_q = np.asarray(labels_q)
    if labels_ref.ndim > 1:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.multioutput import MultiOutputClassifier
        scaler = StandardScaler()
        Zb_s = scaler.fit_transform(Zb_ref)
        Za_s = scaler.transform(Za_q)
        clf = MultiOutputClassifier(
            LogisticRegression(max_iter=500, random_state=0), n_jobs=-1)
        clf.fit(Zb_s, labels_ref)
        preds = np.asarray(clf.predict(Za_s))
        return float((preds == labels_q).mean())
    clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    clf.fit(Zb_ref, labels_ref)
    return float(clf.score(Za_q, labels_q))


def _r_at_1(Za_mapped, Zb_test):
    """Fraction of rows whose nearest row of Zb_test is their own index."""
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn.fit(Zb_test)
    _, idx = nn.kneighbors(Za_mapped)
    return float(sum(int(i in idx[i]) for i in range(len(Za_mapped))) / len(Za_mapped))


# ---------------------------------------------------------------------------
# Procrustes
# ---------------------------------------------------------------------------

def _procrustes_map(Za, Zb):
    """The rotation R in SO(d), through the origin, with Za @ R closest to Zb (the final step
    of the LFM map)."""
    if Za.shape[1] != Zb.shape[1]:
        raise ValueError(f"the two sides must have the same dimension ({Za.shape[1]} vs {Zb.shape[1]})")
    H = Za.T @ Zb
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(U.shape[0])
    D[-1, -1] = np.linalg.det(U @ Vt)
    return (U @ D @ Vt).astype(np.float32)


def _procrustes_fit(Za, Zb):
    """The map x -> (x - ma) R + mb with R orthogonal (reflections allowed) and a translation,
    fitted on true pairs: the paired ceiling. Returns (R, ma, mb); apply with `_procrustes_apply`."""
    if Za.shape[1] != Zb.shape[1]:
        raise ValueError(f"the two sides must have the same dimension ({Za.shape[1]} vs {Zb.shape[1]})")
    ma, mb = Za.mean(0), Zb.mean(0)
    U, _, Vt = np.linalg.svd((Za - ma).T @ (Zb - mb), full_matrices=False)
    return (U @ Vt).astype(np.float32), ma.astype(np.float32), mb.astype(np.float32)


def _procrustes_apply(Z, fit):
    R, ma, mb = fit
    return ((Z - ma) @ R + mb).astype(np.float32)


# ---------------------------------------------------------------------------
# Relative representations (Moschella et al. 2022)
# ---------------------------------------------------------------------------

def _cosine_rel(Z, anchors):
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    An = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)
    return (Zn @ An.T).astype(np.float32)


def _class_centroids(Z, labels):
    """One centroid per class. For an attribute matrix, the positive and the negative centroid
    of every attribute (an attribute constant within the sample contributes one side)."""
    labels = np.asarray(labels)
    if labels.ndim > 1:
        cents = []
        for j in range(labels.shape[1]):
            col = labels[:, j]
            for v in (1, 0):
                mask = col == v
                if mask.any():
                    cents.append(Z[mask].mean(0))
        return np.stack(cents)
    return np.stack([Z[labels == c].mean(0) for c in np.unique(labels)])


# ---------------------------------------------------------------------------
# Latent functional maps (Fumero et al. 2024)
# ---------------------------------------------------------------------------

def _knn_graph(Z, k):
    A = kneighbors_graph(Z, n_neighbors=k, mode="distance", include_self=False, n_jobs=-1)
    A = (A + A.T).tocsr()
    sigma = A.data.mean() if A.data.size > 0 else 1.0
    A.data = np.exp(-A.data ** 2 / (sigma ** 2 + 1e-12))
    return A


def _laplacian_eigs(W, n_eigs):
    d = np.asarray(W.sum(axis=1)).ravel()
    d_inv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = sp.diags(d_inv)
    L = sp.eye(W.shape[0], format="csr") - D @ W @ D
    n_eigs = min(n_eigs, W.shape[0] - 2)
    vals, vecs = eigsh(L, k=n_eigs, which="SM")
    order = np.argsort(vals)
    return vals[order].astype(np.float64), vecs[:, order].astype(np.float64)


def _hks(lam, phi, n_times=100):
    nz = lam > 1e-8
    lam_nz, phi_nz = lam[nz], phi[:, nz]
    if len(lam_nz) == 0:
        return np.zeros((phi.shape[0], n_times))
    t_min = 4 * np.log(10) / lam_nz.max()
    t_max = 4 * np.log(10) / lam_nz.min()
    times = np.geomspace(t_min, t_max, n_times)
    return (phi_nz ** 2) @ np.exp(-lam_nz[:, None] * times[None, :])


def _label_desc(labels):
    """One-hot class descriptors; an attribute matrix is its own descriptor."""
    labels = np.asarray(labels)
    if labels.ndim > 1:
        return labels.astype(np.float64)
    classes = np.unique(labels)
    F = np.zeros((len(labels), len(classes)))
    for i, c in enumerate(classes):
        F[labels == c, i] = 1.0
    return F


def _anchor_desc(Phi, Z_train, anchors):
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Z_train)
    _, idx = nn.kneighbors(anchors)
    anch_phi = Phi[idx.ravel()]
    diff = Phi[:, None, :] - anch_phi[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _norm(F, eps=1e-12):
    return F / (np.linalg.norm(F, axis=0, keepdims=True) + eps)


def _solve_fm(Phi_X, Phi_Y, F_X, F_Y):
    Fhat_X = Phi_X.T @ F_X
    Fhat_Y = Phi_Y.T @ F_Y
    U, s, Vt = np.linalg.svd(Fhat_X, full_matrices=False)
    s_inv = np.where(s > s[0] * 1e-4, 1.0 / s, 0.0)
    return Fhat_Y @ (Vt.T @ np.diag(s_inv) @ U.T)


def _lfm_map(Za_tr, Zb_tr, Za_te, mode, labels_a=None, labels_b=None,
             anchor_Za=None, anchor_Zb=None, k=100, n_eigs=50):
    """Functional map between the Laplacian eigenbases of the two kNN graphs, read out as a
    point correspondence, then a rotation fitted on it and applied to the test points."""
    W_X = _knn_graph(Za_tr, k)
    W_Y = _knn_graph(Zb_tr, k)
    lam_X, Phi_X = _laplacian_eigs(W_X, n_eigs)
    lam_Y, Phi_Y = _laplacian_eigs(W_Y, n_eigs)

    if mode == "unsupervised":
        F_X = _norm(_hks(lam_X, Phi_X))
        F_Y = _norm(_hks(lam_Y, Phi_Y))
    elif mode == "labels":
        F_X = _norm(_label_desc(labels_a))
        F_Y = _norm(_label_desc(labels_b))
    elif mode == "anchors":
        F_X = _norm(_anchor_desc(Phi_X, Za_tr, anchor_Za))
        F_Y = _norm(_anchor_desc(Phi_Y, Zb_tr, anchor_Zb))
    else:
        raise ValueError(mode)

    C = _solve_fm(Phi_X, Phi_Y, F_X, F_Y)
    transferred = Phi_X @ C.T
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Phi_Y)
    _, idx = nn.kneighbors(transferred)
    R = _procrustes_map(Za_tr, Zb_tr[idx.ravel()])
    return (Za_te @ R).astype(np.float32)


# ---------------------------------------------------------------------------
# The five rows of one carve
# ---------------------------------------------------------------------------

def evaluate_from_cache(base_dir, seed, regime, args):
    """The five baseline rows on `<base_dir>/seed<seed>/<regime>/`, with the true pairs of
    `<base_dir>/seed<seed>/paired/` for the Procrustes ceiling and the LFM anchors."""
    d = os.path.join(base_dir, f"seed{seed}", regime)
    dp = os.path.join(base_dir, f"seed{seed}", "paired")
    _ld = lambda dd, name: np.load(os.path.join(dd, name))

    Za_tr, Zb_tr = _ld(d, "Za.npy"), _ld(d, "Zb.npy")
    la_tr, lb_tr = _ld(d, "labels_a.npy"), _ld(d, "labels_b.npy")
    Za_te, Zb_te = _ld(d, "Za_test.npy"), _ld(d, "Zb_test.npy")
    test_lbls = _ld(d, "labels_test.npy")
    Za_pair, Zb_pair = _ld(dp, "Za.npy"), _ld(dp, "Zb.npy")

    logging.info(f"\n{'='*60}\n[{args.dataset} {args.label}, seed{seed}/{regime}]")
    logging.info(f"  Za_tr {Za_tr.shape}  Zb_tr {Zb_tr.shape}  test {Za_te.shape}  "
                 f"paired {Za_pair.shape}")

    results = []

    def _row(method, supervision, top1, r1, elapsed, pair):
        row = {"method": method, "supervision": supervision, "regime": regime,
               "top1": top1, "r_at_1": r1, "elapsed_s": round(elapsed, 1)}
        row.update(retrieval_and_csls(*pair))
        results.append(row)
        logging.info(f"  {method:<30} top1={top1:.4f}  R@1={r1:.4f}  CSLS@1={row['csls_at1']:.4f}"
                     f"  ({elapsed:.1f}s)")

    t0 = time.perf_counter()
    Za_proc = _procrustes_apply(Za_te, _procrustes_fit(Za_pair, Zb_pair))
    _row(f"Procrustes (n={len(Za_pair)})", f"{len(Za_pair)} true pairs, O(d)+t",
         _top1(Za_proc, Zb_te, test_lbls, test_lbls),
         _r_at_1(Za_proc, Zb_te), time.perf_counter() - t0, pair=(Za_proc, Zb_te))

    t0 = time.perf_counter()
    cents_a, cents_b = _class_centroids(Za_tr, la_tr), _class_centroids(Zb_tr, lb_tr)
    Ra, Rb = _cosine_rel(Za_te, cents_a), _cosine_rel(Zb_te, cents_b)
    _row("RelReps-Centroid", "class labels",
         _top1(Ra, Rb, test_lbls, test_lbls), _r_at_1(Ra, Rb), time.perf_counter() - t0, pair=(Ra, Rb))

    t0 = time.perf_counter()
    Za_u = _lfm_map(Za_tr, Zb_tr, Za_te, "unsupervised", k=args.k_graph, n_eigs=args.n_eigs)
    _row("LFM-Unsupervised", "none (HKS graph)",
         _top1(Za_u, Zb_te, test_lbls, test_lbls), _r_at_1(Za_u, Zb_te), time.perf_counter() - t0, pair=(Za_u, Zb_te))

    t0 = time.perf_counter()
    Za_l = _lfm_map(Za_tr, Zb_tr, Za_te, "labels", labels_a=la_tr, labels_b=lb_tr,
                    k=args.k_graph, n_eigs=args.n_eigs)
    _row("LFM-Labels", "class labels",
         _top1(Za_l, Zb_te, test_lbls, test_lbls), _r_at_1(Za_l, Zb_te), time.perf_counter() - t0, pair=(Za_l, Zb_te))

    t0 = time.perf_counter()
    n_anch = min(args.n_anchors, len(Za_pair))
    Za_a = _lfm_map(Za_tr, Zb_tr, Za_te, "anchors",
                    anchor_Za=Za_pair[:n_anch], anchor_Zb=Zb_pair[:n_anch],
                    k=args.k_graph, n_eigs=args.n_eigs)
    _row(f"LFM-Anchors (k={n_anch})", f"{n_anch} true pairs",
         _top1(Za_a, Zb_te, test_lbls, test_lbls), _r_at_1(Za_a, Zb_te), time.perf_counter() - t0, pair=(Za_a, Zb_te))

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True,
                    help="the root holding seed<S>/disjoint and seed<S>/paired")
    ap.add_argument("--seed", type=int, required=True, help="the draw S")
    ap.add_argument("--label", required=True, help="the root's name, written to every row")
    ap.add_argument("--dataset", default="fmnist", help="mnist, fmnist or celeba (for the log)")
    ap.add_argument("--z-dim", type=int, default=None,
                    help="latent dimension; when given, checked against the latents")
    ap.add_argument("--regime", default="disjoint", help="the carve under seed<S> to score")
    ap.add_argument("--n-anchors", type=int, default=50)
    ap.add_argument("--k-graph", type=int, default=100)
    ap.add_argument("--n-eigs", type=int, default=50)
    ap.add_argument("--out-csv", required=True,
                    help="rows are appended; the header is written when the file is new")
    args = ap.parse_args()

    if args.z_dim is not None:
        d = np.load(os.path.join(args.cache_dir, f"seed{args.seed}", args.regime, "Za.npy"),
                    mmap_mode="r").shape[1]
        if d != args.z_dim:
            raise SystemExit(f"--z-dim {args.z_dim} but the latents have dimension {d}")

    rows = evaluate_from_cache(args.cache_dir, args.seed, args.regime, args)

    fields = ["seed", "regime", "label", "method", "supervision",
              "top1", "r_at_1", "elapsed_s"] + CSLS_FIELDS
    exists = os.path.exists(args.out_csv)
    dirname = os.path.dirname(args.out_csv)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({"seed": args.seed, "label": args.label, **{k: r[k] for k in fields if k in r}})
    logging.info(f"Appended {len(rows)} rows (seed={args.seed}) to {args.out_csv}")


if __name__ == "__main__":
    main()
