#!/usr/bin/env python
"""Read the baseline suite on the lung disjoint M = 8 carves next to NGMA, pair by pair.

Every number is a -> b on the 10,000 eval cells (GENOT-Q scores only that direction): R@1,
cell-type accuracy, CSLS@1 (each method's record, all from
baselines/baseline_metrics.retrieval_and_csls) and FOSCTTM (NGMA and Procrustes:
read_fits.py; Rosetta and mini-vec2vec: their package; RelGeo: recomputed here from
relgeo/geo.npz; GENOT-Q: its CSV, when the run recorded it). Also:

    R@1         the mapped cell's nearest b-cell is its own partner
    type acc    the mapped cell's nearest b-cell has the right cell type (chance = sum of squared
                type frequencies, as in read_fits.py)

Sources, all under <root>:
    NGMA, Procrustes ceiling   results/lung_table.csv (read_fits.py): r1_ab, type_acc_ab,
                               ceil_proc_r1_ab, ceil_proc_type_ab
    Rosetta, mini-vec2vec      baselines/{rosetta_wp,minivec2vec_official}/<carve root>_s<d>.json
                               (baselines/linear_baselines.py --carve-glob): pools.carve.a_to_b
                               r_at1 and class_top1
    GENOT-Q                    baselines/genot/lung<size>_<pair>_s<d>_{euclidean,precomputed}.csv:
                               r_at_1, top1 (precomputed = the carve's pullback distances)
    RelGeo                     carves/lung<size>_<pair>_M8/seed<d>/disjoint/relgeo/out/results.csv,
                               row a_to_b: retrieval_at_1, top1_accuracy (1-NN class accuracy)

Prints one table per measure, one row per (size, pair), the mean over the draws present with
their count; `--draws` prints every draw instead. Writes baselines/lung_baselines.csv, one row
per (size, pair, draw, method).

    python experiments/lung/read_baselines.py --root <root> [--draws]
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

METHODS = ["NGMA", "Procrustes ceiling", "Rosetta", "mini-vec2vec", "GENOT-Q euc", "GENOT-Q pullback",
           "RelGeo"]


def foscttm(A, B, block=2048):
    """Fraction of rows of B closer to A[i] than B[i], averaged over i (Euclidean)."""
    bb, out = (B ** 2).sum(1), 0.0
    for s in range(0, len(A), block):
        D = bb[None] - 2.0 * A[s:s + block] @ B.T
        own = D[np.arange(len(D)), np.arange(s, s + len(D))][:, None]
        out += float(((D < own).sum(1) / (len(B) - 1)).sum())
    return out / len(A)


def relgeo_foscttm(carve):
    """RelGeo's FOSCTTM from its saved path lengths: run_relgeo.py's representation is geo_Z divided by
    the s.d. of the anchor-to-anchor lengths, retrieved by Euclidean distance."""
    f = os.path.join(carve, "relgeo", "geo.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    mask = ~np.eye(len(z["classes"]), dtype=bool)
    A = z["geo_Za"] / (z["anchor_geo_a"][mask].std() + 1e-8)
    B = z["geo_Zb"] / (z["anchor_geo_b"][mask].std() + 1e-8)
    return foscttm(np.asarray(A, np.float64), np.asarray(B, np.float64))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def scan_carves(R):
    """(size, pair, draw) of every disjoint M = 8 carve on disk."""
    pat = re.compile(r"lung(\d+)_((?:cross|within)\d)_M8/seed(\d+)/disjoint$")
    for c in glob.glob(os.path.join(R, "carves", "lung*_M8", "seed*", "disjoint")):
        m = pat.search(c)
        if m:
            yield m.groups()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the experiment's root directory")
    ap.add_argument("--draws", action="store_true")
    a = ap.parse_args()
    R, B = a.root, os.path.join(a.root, "baselines")
    got = {}                                       # (size, pair, draw, method) -> (r1, type acc, csls, foscttm)

    ngma = os.path.join(R, "results", "lung_table.csv")
    keys = set()
    chance = None
    if os.path.exists(ngma):
        for r in csv.DictReader(open(ngma)):
            if r["M"] != "8" or r["regime"] != "disjoint":
                continue
            k = (int(r["size"]), r["pair"], int(r["draw"]))
            keys.add(k)
            got[k + ("NGMA",)] = (num(r["r1_ab"]), num(r["type_acc_ab"]), num(r.get("csls_at1_ab")),
                                  num(r.get("foscttm_ab")))
            got[k + ("Procrustes ceiling",)] = (num(r["ceil_proc_r1_ab"]), num(r["ceil_proc_type_ab"]),
                                                num(r.get("ceil_proc_csls_at1_ab")), num(r.get("ceil_proc_foscttm_ab")))
            chance = num(r.get("chance_type_acc")) or chance
    else:
        print(f"(no {ngma}; run read_fits.py first for the NGMA columns)")

    for k in sorted(keys | {(int(s), p, int(d)) for s, p, d in scan_carves(R)}):
        s, p, d = k
        keys.add(k)
        root = f"lung{s}_{p}_M8"
        for sub, name in (("rosetta_wp", "Rosetta"), ("minivec2vec_official", "mini-vec2vec")):
            f = os.path.join(B, sub, f"{root}_s{d}.json")
            if os.path.exists(f):
                m = json.load(open(f)).get("pools", {}).get("carve", {}).get("a_to_b", {})
                got[k + (name,)] = (num(m.get("r_at1")), num(m.get("class_top1")), num(m.get("csls_at1")),
                                    num(m.get("foscttm")))   # the package's FOSCTTM, same definition
        for cost, name in (("euclidean", "GENOT-Q euc"), ("precomputed", "GENOT-Q pullback")):
            f = os.path.join(B, "genot", f"lung{s}_{p}_s{d}_{cost}.csv")
            if os.path.exists(f):
                rows = list(csv.DictReader(open(f)))
                if rows:                           # genot_q.py appends; the last row is the latest run
                    got[k + (name,)] = (num(rows[-1]["r_at_1"]), num(rows[-1]["top1"]),
                                        num(rows[-1].get("csls_at1")), num(rows[-1].get("foscttm")))
        f = os.path.join(R, "carves", root, f"seed{d}", "disjoint", "relgeo", "out", "results.csv")
        if os.path.exists(f):
            row = next((r for r in csv.DictReader(open(f)) if r["direction"] == "a_to_b"), None)
            if row:
                got[k + ("RelGeo",)] = (num(row.get("retrieval_at_1")), num(row.get("top1_accuracy")),
                                        num(row.get("csls_at1")),
                                        relgeo_foscttm(os.path.join(R, "carves", root, f"seed{d}", "disjoint")))

    if not got:
        raise SystemExit("nothing to read")
    os.makedirs(B, exist_ok=True)
    out = os.path.join(B, "lung_baselines.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["size", "pair", "draw", "method", "r1_ab", "type_acc_ab", "csls_ab", "foscttm_ab"])
        for (s, p, d, m), v in sorted(got.items()):
            w.writerow([s, p, d, m, *v])
    print(f"{len(got)} (pair, draw, method) entries; table in {out}")
    if chance is not None:
        print(f"type acc chance {chance:.3f}")

    present = [m for m in METHODS if any(k[3] == m for k in got)]
    for j, title in ((0, "R@1 a->b"), (1, "type acc a->b"), (2, "CSLS@1 a->b"), (3, "FOSCTTM a->b")):
        print(f"\n{title}" + ("  (every draw of the pair, in draw order)" if a.draws else "  (mean over draws, n draws)"))
        w = 16 if a.draws else 13
        print(f"{'size':>6} {'pair':<8}" + "".join(f"{m:>{max(w, len(m) + 1)}}" for m in present))
        cells = defaultdict(dict)
        for (s, p, d, m), v in got.items():
            cells[(s, p)][(m, d)] = v[j]
        for (s, p) in sorted(cells):
            line = f"{s:>6} {p:<8}"
            draws = sorted({k[1] for k in cells[(s, p)]})
            for m in present:
                vals = [(d, cells[(s, p)].get((m, d))) for d in draws]
                if all(v is None for _, v in vals):
                    cell = "--"
                elif a.draws:                      # every draw of the pair, . where this method has none
                    cell = " ".join("  . " if v is None else f"{v:.2f}" for _, v in vals)
                else:
                    vals = [(d, v) for d, v in vals if v is not None]
                    cell = f"{np.mean([v for _, v in vals]):.3f} ({len(vals)})"
                line += f"{cell:>{max(w, len(m) + 1)}}"
            print(line)


if __name__ == "__main__":
    main()
