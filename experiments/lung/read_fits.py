#!/usr/bin/env python
"""Read every lung fit into one table, with R@1 inside each cell type alongside plain R@1.

Cells of one type are close to interchangeable among the 10,000 eval cells, so plain R@1 mixes
two things: whether a cell lands in the right type, and whether it finds its partner within
that type. This separates them. From each fit's `<name>_mapped.npz` (the mapped eval cells both
ways and the two directions' weights) it computes, on the 10,000 eval cells:

    r1_ab, r1_ba        plain R@1 per direction
    r1                  plain R@1 of the combined ranking (fit_map.py's own rule: each
                        direction's distances over their median, weighted by the saved weights)
    r1_type             the same combined ranking with candidates restricted to the cell's own
                        type; chance is the mean over cells of 1 / (cells of that type)
    type_acc            the combined ranking's nearest cell has the right type
    comp_acc            the same for the compartment
    foscttm(_ab, _ba)   fraction of candidates closer than the true partner, averaged over cells;
                        also for the Procrustes ceiling (ceil_proc_foscttm_{ab,ba})
    csls_at1_ab         CSLS@1 with baselines/baseline_metrics.retrieval_and_csls, the
                        implementation every baseline's record uses, on the M = 8 disjoint fits
                        (--csls all computes it everywhere, both directions); also for the ceilings

and from the JSON of a disjoint fit (--oracle --oracle-carve): the ceilings, which fit_map.py
scores a -> b, `ceil_map_r1_ab` (the same map class trained on the true pairs) and
`ceil_proc_r1_ab` (the closed-form O(d)+t map on the same pairs), with the Procrustes ceiling's
FOSCTTM, CSLS@1, within-type R@1 and compartment accuracy recomputed from its saved Q and b in
the carve's standardised frame. From the coupling's coupling.json: `coupling_acc`, the
coupling's argmax accuracy on the shared cells (overlap > 0 only).

Writes <root>/results/lung_table.csv (one row per fit) and prints a summary grouped by training
size, pair kind, M and regime, every draw shown.

    python experiments/lung/read_fits.py --root <root> [--csls none|m8-disjoint|all]
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "baselines"))
from baseline_metrics import retrieval_and_csls  # noqa: E402

NAME = re.compile(r"lung(?P<size>\d+)_(?P<pair>(cross|within)\d)_M(?P<M>\w+?)_(?P<regime>disjoint|frac\d+)"
                  r"_s(?P<draw>\d+)$")


def sqdist(A, B):
    A, B = A.astype(np.float32), B.astype(np.float32)
    D = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2.0 * A @ B.T
    return np.sqrt(np.maximum(D, 0.0))


def r1(D):
    return float((D.argmin(1) == np.arange(len(D))).mean())


def r1_within(D, lab):
    hit = 0
    for t in np.unique(lab):
        i = np.flatnonzero(lab == t)
        hit += int((D[np.ix_(i, i)].argmin(1) == np.arange(len(i))).sum())
    return hit / len(lab)


def foscttm(D):
    """Fraction of candidates closer than the true partner (row i's partner is column i), averaged
    over rows: 0 when every partner is the nearest, about 0.5 at random."""
    own = np.diag(D)[:, None]
    return float(((D < own).sum(1) / (D.shape[1] - 1)).mean())


def label_chance(lab):
    """Accuracy of matching each cell to a random cell: the sum of squared label frequencies."""
    _, cnt = np.unique(lab, return_counts=True)
    p = cnt / cnt.sum()
    return float((p ** 2).sum())


def chance_within(lab):
    _, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    return float((1.0 / cnt[inv]).mean())


def read_one(js, comp_eval, csls="m8-disjoint"):
    m = NAME.match(os.path.basename(js)[:-5])
    if not m:
        return None
    R = json.load(open(js))
    mp = js[:-5] + "_mapped.npz"
    row = {k: m.group(k) for k in ("size", "pair", "M", "regime", "draw")}
    row["kind"] = row["pair"][:-1]
    if not os.path.exists(mp):
        row["error"] = "no mapped.npz"
        return row
    Z = np.load(mp)
    lab = Z["labels_te"]
    Dab, Dba = sqdist(Z["mapped_ab"], Z["Zb_te"]), sqdist(Z["mapped_ba"], Z["Za_te"]).T
    wa, wb = float(Z["weights"][0]), float(Z["weights"][1])
    Dc = wa * Dab / np.median(Dab) + wb * Dba / np.median(Dba)
    row.update(r1_ab=r1(Dab), r1_ba=r1(Dba.T), r1=r1(Dc), r1_type=r1_within(Dc, lab),
               chance_type=chance_within(lab), type_acc=float((lab[Dc.argmin(1)] == lab).mean()),
               type_acc_ab=float((lab[Dab.argmin(1)] == lab).mean()),
               type_acc_ba=float((lab[Dba.T.argmin(1)] == lab).mean()),
               chance_type_acc=label_chance(lab),
               foscttm_ab=foscttm(Dab), foscttm_ba=foscttm(Dba.T), foscttm=foscttm(Dc))
    # CSLS on 10,000 x 10,000 is slow, so by default only where the tables use it: M = 8, disjoint, a -> b
    do_csls = csls == "all" or (csls == "m8-disjoint" and row["M"] == "8" and row["regime"] == "disjoint")
    if do_csls:
        row["csls_at1_ab"] = retrieval_and_csls(Z["mapped_ab"], Z["Zb_te"])["csls_at1"]
        if csls == "all":
            row["csls_at1_ba"] = retrieval_and_csls(Z["mapped_ba"], Z["Za_te"])["csls_at1"]
    comp = comp_eval if comp_eval is not None and len(comp_eval) == len(lab) else None
    if comp is not None:
        row.update(comp_acc=float((comp[Dc.argmin(1)] == comp).mean()),
                   comp_acc_ab=float((comp[Dab.argmin(1)] == comp).mean()),
                   comp_acc_ba=float((comp[Dba.T.argmin(1)] == comp).mean()),
                   chance_comp_acc=label_chance(comp))
    c = R["carve"]
    for d, (src, dst, te_src, te_dst) in {"ab": ("Za", "Zb", "Za_te", "Zb_te"),
                                          "ba": ("Zb", "Za", "Zb_te", "Za_te")}.items():
        O = R.get("a_to_b" if d == "ab" else "b_to_a", {}).get("oracle")
        if not O:
            continue                                   # no ceiling on this fit
        row[f"ceil_map_r1_{d}"] = O.get("r_at_1")
        row[f"ceil_map_type_{d}"] = O.get("class_top1")
        row[f"ceil_map_csls_at1_{d}"] = O.get("csls_at1")
        rc = O.get("rigid_closed")
        if not rc:
            continue
        row[f"ceil_proc_r1_{d}"] = rc.get("r_at_1")
        row[f"ceil_proc_type_{d}"] = rc.get("class_top1")
        # the Procrustes ceiling's map in fit_map.py's frame: centre each training cloud, divide
        # by its mean norm, x @ Q.T + b, then undo the target's standardisation
        X, Y = np.load(os.path.join(c, f"{src}.npy")), np.load(os.path.join(c, f"{dst}.npy"))
        mx, my = X.mean(0), Y.mean(0)
        sx, sy = np.linalg.norm(X - mx, axis=1).mean(), np.linalg.norm(Y - my, axis=1).mean()
        P = ((Z[te_src] - mx) / sx) @ np.asarray(rc["Q"]).T + np.asarray(rc["b"])
        Dp = sqdist(P * sy + my, Z[te_dst])
        row[f"ceil_proc_foscttm_{d}"] = foscttm(Dp)
        if do_csls and (d == "ab" or csls == "all"):
            row[f"ceil_proc_csls_at1_{d}"] = retrieval_and_csls(P * sy + my, Z[te_dst])["csls_at1"]
        row[f"ceil_proc_r1_type_{d}"] = r1_within(Dp, lab)
        if comp is not None:
            row[f"ceil_proc_comp_{d}"] = float((comp[Dp.argmin(1)] == comp).mean())
    cj = os.path.join(os.path.dirname(R["pi"]), "coupling.json")
    if os.path.exists(cj):
        row["coupling_acc"] = json.load(open(cj)).get("argmax_acc_on_shared")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the experiment's root directory")
    ap.add_argument("--csls", default="m8-disjoint", choices=["none", "m8-disjoint", "all"],
                    help="where to compute CSLS@1 (slow): the M = 8 disjoint fits a -> b (default), all, none")
    a = ap.parse_args()
    rdir = os.path.join(a.root, "results")
    comp_eval = None
    lp, sp = os.path.join(a.root, "labels.npz"), os.path.join(a.root, "split.npz")
    if os.path.exists(lp) and os.path.exists(sp):
        comp_eval = np.load(lp)["compartment"][np.load(sp)["eval"]]
    files = sorted(glob.glob(os.path.join(rdir, "*.json")))
    rows = []
    for i, j in enumerate(files):
        r = read_one(j, comp_eval, a.csls)
        if r:
            rows.append(r)
        if (i + 1) % 20 == 0:
            print(f"  read {i + 1}/{len(files)} files", flush=True)
    if not rows:
        raise SystemExit(f"no lung result JSONs matching the naming rule in {rdir}")
    cols = ["size", "kind", "pair", "M", "regime", "draw", "r1", "r1_type",
            "chance_type", "type_acc", "r1_ab", "r1_ba",
            "ceil_map_r1_ab", "ceil_map_r1_ba", "ceil_proc_r1_ab", "ceil_proc_r1_ba",
            "ceil_proc_r1_type_ab", "ceil_proc_r1_type_ba",
            "type_acc_ab", "type_acc_ba", "chance_type_acc", "ceil_map_type_ab", "ceil_map_type_ba",
            "ceil_proc_type_ab", "ceil_proc_type_ba",
            "comp_acc", "comp_acc_ab", "comp_acc_ba", "chance_comp_acc", "ceil_proc_comp_ab",
            "ceil_proc_comp_ba", "coupling_acc", "foscttm", "foscttm_ab", "foscttm_ba",
            "ceil_proc_foscttm_ab", "ceil_proc_foscttm_ba", "csls_at1_ab", "csls_at1_ba",
            "ceil_proc_csls_at1_ab", "ceil_proc_csls_at1_ba", "ceil_map_csls_at1_ab", "ceil_map_csls_at1_ba", "error"]
    out = os.path.join(rdir, "lung_table.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: "--" if v is None else f"{v:.3f}"
    groups = defaultdict(list)
    for r in rows:
        groups[(int(r["size"]), r["kind"], r["M"], r["regime"])].append(r)
    print(f"{len(rows)} fits; table in {out}\n")
    print("size   kind    M    regime    | R@1 per draw          | R@1 in type (chance) | type acc "
          "| a->b: method vs ceilings map / Procrustes | coupling acc")
    for k in sorted(groups):
        g = sorted(groups[k], key=lambda r: (r["pair"], r["draw"]))
        if any("error" in r for r in g):
            print(f"{k}: {[r.get('error') for r in g]}")
            continue
        draws = " ".join(f"{r['pair'][-1]}.{r['draw']}:{r['r1']:.3f}" for r in g)
        rt = " ".join(f"{r['r1_type']:.3f}" for r in g)
        ta = " ".join(f"{r['type_acc']:.3f}" for r in g)
        ce = " ".join(f"{r['r1_ab']:.3f} vs {fmt(r.get('ceil_map_r1_ab'))}/{fmt(r.get('ceil_proc_r1_ab'))}"
                      for r in g if r.get("ceil_map_r1_ab") is not None) or "--"
        ca = " ".join(fmt(r.get("coupling_acc")) for r in g if r.get("coupling_acc") is not None) or "--"
        print(f"{k[0]:<6} {k[1]:<7} {k[2]:<4} {k[3]:<9} | {draws} | {rt} ({g[0]['chance_type']:.4f}) | {ta} | {ce} | {ca}")


if __name__ == "__main__":
    main()
