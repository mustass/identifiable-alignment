#!/usr/bin/env python
"""Run NGMA's map fit (scripts/fit_map.py) on several lung carves in one process.

For each carve `<root>/carves/lung{s}_{pair}_M{m}/seed{d}/{regime}` it runs

    fit_map.py <carve> --pi <carve>/alignments/<mode>_kn_<coupling>/pi.npy
        --mode <mode> --geometry <pullback|euclidean> --fit-weight 0.3 --cycle-weight 0.1
        --joint-rounds 3 [--oracle --oracle-carve <the frac100 sibling>]
        --out <root>/results/lung{s}_{pair}_M{m}_{regime}_s{d}.json

with mode `ensemble_eol_samedec`, geometry `pullback` and the fused coupling FGWG_ann for M = 1
and 8, and mode `euclidean`, geometry `euclidean` and the GW coupling for the Euclidean
comparator M = euc. A disjoint carve whose frac100 sibling exists also gets the two ceilings
(the same map class and the Procrustes map, both fitted on the true pairs). A carve without
its coupling, or whose output exists, is skipped; a failed fit is reported and the next one runs.

    PYTHONPATH=src python experiments/lung/fit_maps.py --root <root> <carve> [<carve> ...]
"""
import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PAT = re.compile(r"/(?P<src>lung\d+_(cross|within)\d_M(?P<m>\w+?))/seed(?P<d>\d+)/(?P<reg>disjoint|frac\d+)/?$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carves", nargs="+")
    ap.add_argument("--root", required=True, help="the experiment's root directory (results/ goes there)")
    a = ap.parse_args()
    rdir = os.path.join(a.root, "results")
    os.makedirs(rdir, exist_ok=True)
    done = failed = skipped = 0
    for carve in a.carves:
        m = PAT.search(carve)
        if not m:
            print(f"SKIP, not a lung carve path: {carve}", flush=True)
            skipped += 1
            continue
        euc = m.group("m") == "euc"
        mode, geo, coupling = ("euclidean", "euclidean", "GW") if euc else ("ensemble_eol_samedec", "pullback", "FGWG_ann")
        pi = os.path.join(carve, "alignments", f"{mode}_kn_{coupling}", "pi.npy")
        name = f"{m.group('src')}_{m.group('reg')}_s{m.group('d')}"
        out = os.path.join(rdir, f"{name}.json")
        if os.path.exists(out):
            print(f"exists: {out}", flush=True)
            skipped += 1
            continue
        if not os.path.exists(pi):
            print(f"SKIP, no coupling: {pi}", flush=True)
            skipped += 1
            continue
        cmd = [sys.executable, os.path.join(REPO, "scripts", "fit_map.py"), carve, "--pi", pi,
               "--mode", mode, "--geometry", geo, "--fit-weight", "0.3", "--cycle-weight", "0.1",
               "--joint-rounds", "3"]
        c100 = os.path.join(os.path.dirname(carve.rstrip("/")), "frac100")
        if m.group("reg") == "disjoint" and os.path.exists(os.path.join(c100, "gt_ab.npy")):
            cmd += ["--oracle", "--oracle-carve", c100]
        cmd += ["--out", out]
        t0 = time.time()
        print(f"== {name}\n   {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=REPO)
        print(f"   exit {rc} after {time.time() - t0:.0f}s", flush=True)
        done, failed = (done + 1, failed) if rc == 0 else (done, failed + 1)
    print(f"batch finished: {done} fitted, {failed} failed, {skipped} skipped", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
