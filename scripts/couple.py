#!/usr/bin/env python
"""The coupling between the two sides of a carve.

NGMA's coupling is entropic Gromov-Wasserstein with a linear term, annealed in four stages.
Both distance matrices are divided by one scale, the mean of their two mean entries. At
every Frank-Wolfe step the inner Sinkhorn solve runs on

    (1 - alpha_k) L_T + alpha_k M,

where L_T is entropic GW's linearised cost at the current plan and M is the ground cost of
Memoli's third lower bound on the same normalised distances (the squared 2-Wasserstein
distance between the two points' distance distributions). The stage weights are
alpha = 1, 2/3, 1/3, 0 and the temperatures interpolate geometrically between GW's schedule
1, 0.1, 0.01, 0.001 and the lower bound's own temperature, 3e-4 of max(M), so the last stage
is plain entropic GW at epsilon 1e-3. Each stage starts from the previous plan, the first
from the independent coupling; the inner Sinkhorn runs to convergence with its dual
potentials carried from one step to the next. Solved in float64 on JAX, at most 500
Frank-Wolfe steps per stage, stopping when the plan's relative change is below 1e-9.

The directory name of this coupling is FGWG_ann. With --coupling GW the plan is plain
annealed entropic GW on the same schedule (used for the Euclidean comparator of the lung
experiment, with --mode euclidean).

Input: a carve directory holding C_<mode>_a.npy and C_<mode>_b.npy (and gt_ab.npy, read only
to report the argmax accuracy on the images both sides share). Output:

    <carve>/alignments/<mode>_kn_<coupling>/pi.npy        the plan, float64, uniform marginals
    <carve>/alignments/<mode>_kn_<coupling>/coupling.json  settings and the solver record

A carve whose pi.npy exists is skipped.

    PYTHONPATH=src python scripts/couple.py <carve> [<carve> ...] [--mode ensemble_eol_samedec]
        [--coupling FGWG_ann]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import jax  # noqa: E402

from ngma.alignment.couplings import fgw_alpha_tag, fgw_coupling  # noqa: E402
from ngma.alignment.gw import gw_entropic_jax  # noqa: E402
from ngma.alignment.schedule import build_eps_schedule  # noqa: E402

SETTINGS = {"dtype": "float64", "tol": 1e-9, "normalize": "shared", "eps_start": 1.0,
            "eps_final": 1e-3, "n_stages": 4, "max_iter": 500}
FUSED = {"alpha": "ann", "eps_rule": "gmix", "warm_duals": True}
COUPLINGS = {fgw_alpha_tag(FUSED["alpha"], FUSED["eps_rule"]): "FGW", "GW": "GW"}


def schedule():
    return build_eps_schedule(SETTINGS["eps_start"], SETTINGS["eps_final"], n_steps=SETTINGS["n_stages"])


def fused_coupling(Ca, Cb):
    """The plan and the solver record of NGMA's coupling on two distance matrices."""
    pi, info = fgw_coupling(Ca, Cb, FUSED["alpha"], schedule(), max_iter=SETTINGS["max_iter"],
                            tol=SETTINGS["tol"], dtype=SETTINGS["dtype"], normalize=SETTINGS["normalize"],
                            eps_rule=FUSED["eps_rule"], warm_duals=FUSED["warm_duals"])
    if not info["finite"]:
        raise FloatingPointError(f"the plan is not finite: {info}")
    return np.asarray(pi, np.float64), info


def gw_coupling(Ca, Cb):
    """Plain annealed entropic GW on the same schedule, as a comparator."""
    scale = 0.5 * (float(Ca.mean()) + float(Cb.mean()))
    Ca_n, Cb_n = (Ca / scale).astype(np.float32), (Cb / scale).astype(np.float32)
    pi, stages = None, []
    for eps in schedule():
        pi, log = gw_entropic_jax(Ca_n, Cb_n, epsilon=eps, init_coupling=pi, max_iter=SETTINGS["max_iter"],
                                  tol=SETTINGS["tol"], sink_tol=SETTINGS["tol"], dtype=SETTINGS["dtype"])
        stages.append({"eps": float(eps), "iters": int(log.get("iters_done", -1)),
                       "converged": bool(log.get("converged", False)),
                       "final_rel_change": float(log.get("final_rel_change", float("nan")))})
    pi64 = np.asarray(pi, np.float64)
    a, b = pi64.sum(1), pi64.sum(0)
    Ca64, Cb64 = Ca_n.astype(np.float64), Cb_n.astype(np.float64)
    cost = float(a @ (Ca64 ** 2) @ a + b @ (Cb64 ** 2) @ b - 2.0 * np.sum(pi64 * (Ca64 @ pi64 @ Cb64.T)))
    return pi64, {"stages": stages, "gw_cost": cost, "scale": scale}


def couple(carve, mode, coupling):
    out = os.path.join(carve, "alignments", f"{mode}_kn_{coupling}")
    if os.path.exists(os.path.join(out, "pi.npy")):
        print(f"exists: {out}/pi.npy", flush=True)
        return
    Ca = np.load(os.path.join(carve, f"C_{mode}_a.npy"))
    Cb = np.load(os.path.join(carve, f"C_{mode}_b.npy"))
    t0 = time.time()
    pi, info = fused_coupling(Ca, Cb) if COUPLINGS[coupling] == "FGW" else gw_coupling(Ca, Cb)
    rec = {"carve": os.path.abspath(carve), "mode": mode, "coupling": coupling, "n": int(len(Ca)),
           "settings": {**SETTINGS, **(FUSED if COUPLINGS[coupling] == "FGW" else {}),
                        "schedule": schedule()},
           "solver": info, "secs": round(time.time() - t0, 1)}
    gt_path = os.path.join(carve, "gt_ab.npy")
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
        shared = gt >= 0
        rec["n_shared"] = int(shared.sum())
        rec["argmax_acc_on_shared"] = (float((pi.argmax(1)[shared] == gt[shared]).mean())
                                       if shared.any() else None)
    os.makedirs(out, exist_ok=True)
    json.dump(rec, open(os.path.join(out, "coupling.json"), "w"), indent=1, default=str)
    np.save(os.path.join(out, "pi.npy"), pi)      # written last: pi.npy means the coupling is complete
    print(f"{out}: n={len(Ca)} {rec['secs']}s"
          + (f" argmax accuracy on {rec['n_shared']} shared points {rec['argmax_acc_on_shared']:.3f}"
             if rec.get("n_shared") else ""), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carves", nargs="+", help="carve directories, solved in turn")
    ap.add_argument("--mode", default="ensemble_eol_samedec", choices=["ensemble_eol_samedec", "euclidean"],
                    help="which distance matrices to couple")
    ap.add_argument("--coupling", default="FGWG_ann", choices=sorted(COUPLINGS),
                    help="FGWG_ann is NGMA's coupling; GW is plain annealed entropic GW")
    a = ap.parse_args()
    jax.config.update("jax_enable_x64", True)
    for carve in a.carves:
        couple(carve, a.mode, a.coupling)


if __name__ == "__main__":
    main()
