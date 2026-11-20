#!/usr/bin/env python
"""Rosetta's Wasserstein Procrustes and the official mini-vec2vec on every carve of a glob.

Both fit an orthogonal map on the carve's latents through the authors' package (`rosetta.py`,
one process per carve and method) and are scored on the test split. Outputs, one JSON per
carve: `<out-root>/rosetta_wp/<root>_s<S>.json` and `<out-root>/minivec2vec_official/<root>_s<S>.json`,
where <root> is the carve's grandparent folder and S its seed<S> folder. A carve whose record
exists and parses is skipped, so a rerun resumes. Each run gets one thread; `--workers` of
them run at once (one run on a 10,000-point test split needs about 3.6 GB).

Run it in the environment that holds the `unpaired_rosetta` package (see rosetta.py):

    $ENV/bin/python baselines/linear_baselines.py \\
        --carve-glob 'checkpoints/*/seed*/disjoint' --out-root results --workers 8
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = {
    "mv_official": ("minivec2vec_official", lambda c, s, o: ["rosetta.py", "--carve", c, "--seed", s,
                                                             "--method", "mini_vec2vec_official", "--out", o]),
    "rosetta": ("rosetta_wp", lambda c, s, o: ["rosetta.py", "--carve", c, "--seed", s, "--out", o]),
}
NEEDED = ("Za.npy", "Zb.npy", "Za_test.npy", "Zb_test.npy")


def complete(path):
    """A record exists and parses; a run killed while writing leaves a truncated file."""
    try:
        with open(path) as fh:
            return "pools" in json.load(fh)
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carve-glob", required=True,
                    help="glob of carve directories, e.g. 'checkpoints/*/seed*/disjoint'")
    ap.add_argument("--methods", default="mv_official,rosetta")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--dry", action="store_true", help="list the runs, start nothing")
    a = ap.parse_args()

    found, missing = [], []
    for d in sorted(glob.glob(a.carve_glob)):
        m = re.search(r"seed(\d+)", d)
        entry = (os.path.basename(os.path.dirname(os.path.dirname(d.rstrip("/")))), m.group(1) if m else "1", d)
        (found if all(os.path.exists(os.path.join(d, n)) for n in NEEDED) else missing).append(entry)
    print(f"{len(found)} carves with latents present, {len(missing)} without them"
          + (f" (first: {missing[0][2]})" if missing else ""))
    jobs = []
    for name in [m for m in METHODS if m in a.methods.split(",")]:
        sub, argv = METHODS[name]
        for root, seed, d in found:
            out = os.path.join(a.out_root, sub, f"{root}_s{seed}.json")
            if not complete(out):
                jobs.append((name, f"{root}_s{seed}", out, argv(d, seed, out)))
    print(f"{len(jobs)} runs to do: " + ", ".join(f"{m} {sum(j[0] == m for j in jobs)}" for m in METHODS))
    if a.dry or not jobs:
        for j in jobs[:10]:
            print("  would run", j[0], j[1])
        return

    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "PYTHONWARNINGS": "ignore"}
    # the workers must not all bind their one thread to the same core
    env.pop("OMP_PLACES", None)
    env["OMP_PROC_BIND"] = "false"

    def run(job):
        name, tag, out, argv = job
        log = os.path.join(os.path.dirname(out), "logs", f"{tag}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.run([sys.executable, os.path.join(HERE, argv[0]), *argv[1:]],
                                stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        line = next((l.strip() for l in open(log) if "R@1 a->b" in l), "")
        return name, tag, rc, time.time() - t0, line, log

    failed = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(run, j) for j in jobs]), 1):
            name, tag, rc, sec, line, log = f.result()
            failed += rc != 0
            r = re.search(r"R@1 a->b [0-9.]+", line)
            print(f"[{i}/{len(jobs)}] {name:11s} {tag:34s} "
                  + (r.group(0) if rc == 0 and r else f"FAILED (rc {rc}), see {log}") + f"  {sec:.0f}s", flush=True)
    print(f"done, {failed} failed")


if __name__ == "__main__":
    sys.exit(main())
