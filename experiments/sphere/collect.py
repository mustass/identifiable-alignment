"""The figure inputs of the sphere runs, from their results.pkl files.

Reads <root>/<density>/round/<mode>/seed<S>/results.pkl, written by experiments/sphere/run.py,
and writes into --out-dir:

  sphere_err_raw.json    {run: {density, geometry, sample_mode, seed, errs}} with
                         errs["results/<k>/dopt/<coupling>/err_deg"], the landmark cases
  sphere_full_raw.json   the same with "results/full/all/<coupling>/err_deg" and n_points, the
                         full-cloud case
  sphere_runs.csv        one row per run, k and coupling: the error, the GW stages (iterations,
                         converged, seconds), the LP status, the pairing accuracy in the
                         permuted regime and the D-opt switch index

The couplings are named as the runs name them: ldd_lp (the exact LP), gw (annealed GW alone) and
fgw (NGMA's coupling). The GW columns of sphere_runs.csv also record the fused coupling's stages.

    PYTHONPATH=src python experiments/sphere/collect.py --root results/sphere --out-dir results/sphere
"""
import argparse
import csv
import glob
import json
import os
import pickle


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join("results", "sphere"))
    ap.add_argument("--out-dir", default=None, help="default: --root")
    a = ap.parse_args()
    out_dir = a.out_dir or a.root

    paths = sorted(glob.glob(os.path.join(a.root, "*", "round", "*", "seed*", "results.pkl")))
    if not paths:
        raise SystemExit(f"no results.pkl under {a.root}")
    raw, full, rows = {}, {}, []
    for p in paths:
        R = pickle.load(open(p, "rb"))
        name = f"{R['density']}_{R['sample_mode']}_s{R['seed']}"
        head = {"density": R["density"], "geometry": R.get("geometry", "round"),
                "sample_mode": R["sample_mode"], "seed": str(R["seed"])}
        errs, ferrs = {}, {}
        sw = R.get("dopt_switch") or {}
        for (k, method), case in R["cases"].items():
            for coup, cr in case["coupling_results"].items():
                if k == "full":
                    ferrs[f"results/full/all/{coup}/err_deg"] = cr["err_deg"]
                else:
                    errs[f"results/{k}/{method}/{coup}/err_deg"] = cr["err_deg"]
                d = cr.get("coupling_diag", {})
                st = d.get("gw_stages") or []
                rows.append({
                    **head, "k": R["n"] if k == "full" else k, "method": method, "coupling": coup,
                    "err_deg": cr["err_deg"], "t_coupling": cr.get("t_coupling"),
                    "coupling_valid": d.get("is_valid"),
                    "max_row_err": d.get("max_row_err"), "max_col_err": d.get("max_col_err"),
                    "lp_result_code": d.get("lp_result_code"), "lp_warning": d.get("lp_warning"),
                    "pair_acc": d.get("pair_acc"), "pair_n": d.get("pair_n"),
                    "gw_budget_limited": d.get("gw_budget_limited"),
                    "gw_iters": "/".join(str(s["iters"]) for s in st),
                    "gw_converged": "/".join("y" if s["converged"] else "n" for s in st),
                    "gw_seconds": "/".join(f"{s['seconds']:.1f}" for s in st),
                    "dopt_switch_a": sw.get("a"), "dopt_switch_b": sw.get("b"),
                    "measures_sha": (R.get("provenance") or {}).get("sphere_measures_sha"),
                    "source": p})
        raw[name] = {**head, "errs": errs}
        full[name] = {**head, "n_points": R["n"], "errs": ferrs}

    os.makedirs(out_dir, exist_ok=True)
    for fn, obj in (("sphere_err_raw.json", raw), ("sphere_full_raw.json", full)):
        with open(os.path.join(out_dir, fn), "w") as fh:
            json.dump(obj, fh)
    fn = os.path.join(out_dir, "sphere_runs.csv")
    with open(fn, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    shas = {r["measures_sha"] for r in rows}
    print(f"{len(paths)} runs, {len(rows)} rows -> {out_dir}/sphere_err_raw.json, "
          f"sphere_full_raw.json, sphere_runs.csv")
    if len(shas) > 1:
        print(f"  the runs come from more than one sphere_measures.yaml: {sorted(shas)}")


if __name__ == "__main__":
    main()
