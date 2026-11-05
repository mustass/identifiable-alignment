#!/usr/bin/env python
"""Alignment of two samples on the sphere S^2 related by a known rotation (Fig. 3a, b).

One run is one (measure, sampling regime, seed). It draws the two samples, builds their geodesic
distance matrices and LDD signatures, orders landmarks on each side by D-optimal design, and for
every landmark count k (and for the full cloud, k = n) solves the LDD-LP, and on the mixtures
also NGMA's annealed entropic GW, on the k landmarks. Each coupling is turned into a rotation by
weighted Procrustes over SO(3), and the error is the angle between that rotation and the true one.

Measures: `configs/sphere_measures.yaml` (uniform, vmf, mixture_vmf, mixture_vmf_eq) with the
ground-truth rotation. Solver settings: `configs/sphere.yaml`.

Regimes. `independent`: the two samples are drawn independently from P and R#P. `permuted`: the
second sample is the image of the first under R in a random order, z^b_{sigma(i)} = R z^a_i, and
C^b is C^a with rows and columns permuted by sigma.

Writes <out>/<density>/round/<mode>/seed<S>/results.pkl, which experiments/sphere/collect.py
reads, and the couplings under <out>/.../couplings/ when `save_couplings` is on.

    PYTHONPATH=src python experiments/sphere/run.py --density mixture_vmf --mode independent \\
        --seed 10 --out results/sphere
"""
import argparse
import logging
import os
import pickle
import time

import numpy as np
import yaml

from ngma.alignment.closed_form import fit_orthogonal
from ngma.alignment.gw import gw_entropic_jax
from ngma.alignment.schedule import build_eps_schedule
from ngma.data.sphere import (load_sphere_measures, make_sphere_experiment, sphere_measure,
                              sphere_measures_hash)
from ngma.geometry.ldd import compute_ldd_signatures, greedy_dopt_selection

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DENSITIES = ("uniform", "vmf", "mixture_vmf", "mixture_vmf_eq")
MODES = ("independent", "permuted")


def sphere_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Great-circle distances between the rows of an (n, 3) array of unit vectors."""
    pts = np.asarray(points, dtype=np.float32)
    cos_sim = np.clip(pts @ pts.T, -1.0, 1.0)
    return np.arccos(cos_sim).astype(np.float32)


def rotation_error(R_pred: np.ndarray, R_star: np.ndarray) -> float:
    """The angle, in radians, of the residual rotation R_pred^T R_star."""
    cos_angle = (np.trace(R_pred.T @ R_star) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def coupling_diagnostics(pi: np.ndarray) -> dict:
    """Total mass, the largest deviation of the row and column sums from 1/k, and whether the
    plan is a valid coupling of two uniform measures (finite, mass near 1, marginals within 10%)."""
    k = pi.shape[0]
    target = 1.0 / k
    pi64 = pi.astype(np.float64)
    total_mass = float(pi64.sum())
    max_row_err = float(np.abs(pi64.sum(axis=1) - target).max())
    max_col_err = float(np.abs(pi64.sum(axis=0) - target).max())
    has_nan = bool(np.any(np.isnan(pi64)))
    has_inf = bool(np.any(np.isinf(pi64)))
    is_valid = (not has_nan and not has_inf and total_mass > 0.5
                and max_row_err < 0.1 * target and max_col_err < 0.1 * target)
    return dict(total_mass=total_mass, max_row_err=max_row_err, max_col_err=max_col_err,
                has_nan=has_nan, has_inf=has_inf, is_valid=is_valid)


def ldd_lp_cost(Ha: np.ndarray, Hb: np.ndarray, dr: float) -> np.ndarray:
    """The cost of the LDD-LP: the squared L2 distance between the signature vectors of a point
    on each side, times the spacing dr of the shared radius grid, in float64."""
    A, B = Ha.astype(np.float64), Hb.astype(np.float64)
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(sq, 0.0) * float(dr)


def ldd_lp(M: np.ndarray) -> tuple[np.ndarray, dict]:
    """The LDD-LP on uniform marginals, solved exactly by network simplex."""
    import ot
    k_a, k_b = M.shape
    pi, log = ot.emd(np.full(k_a, 1.0 / k_a), np.full(k_b, 1.0 / k_b), M,
                     numItermax=10_000_000, log=True)
    return np.asarray(pi, dtype=np.float64), {
        "lp_solver": "emd", "lp_result_code": int(log.get("result_code", -1)),
        "lp_warning": str(log.get("warning") or ""), "lp_cost": float(log["cost"]),
        "lp_cost_mean": float(M.mean())}


def gw_annealed(Ca_k: np.ndarray, Cb_k: np.ndarray, schedule: list, max_iter: int,
                tol: float) -> tuple[np.ndarray, float, list, float]:
    """NGMA's entropic GW: both distance matrices divided by the average of their means, epsilon
    annealed over `schedule` with each stage started from the previous plan, float64, outer and
    inner tolerance `tol`, at most `max_iter` Frank-Wolfe steps per stage.

    Returns (plan, GW distance in the original units, one record per stage, the scale)."""
    scale = 0.5 * (float(Ca_k.mean()) + float(Cb_k.mean()))
    Ca_n = (Ca_k / scale).astype(np.float64)
    Cb_n = (Cb_k / scale).astype(np.float64)
    pi, stages = None, []
    for eps in schedule:
        pi, log = gw_entropic_jax(Ca_n, Cb_n, epsilon=float(eps), init_coupling=pi,
                                  max_iter=int(max_iter), tol=float(tol), sink_tol=float(tol),
                                  dtype="float64")
        stages.append({"eps": float(eps), "iters": int(log.get("iters_done", -1)),
                       "converged": bool(log.get("converged", False)),
                       "final_rel_change": float(log.get("final_rel_change", float("nan"))),
                       "seconds": float(log.get("seconds", float("nan")))})
    pi = np.asarray(pi, dtype=np.float64)
    M = ((Ca_n ** 2) @ pi.sum(1))[:, None] + ((Cb_n ** 2) @ pi.sum(0))[None, :] \
        - 2.0 * (Ca_n @ pi @ Cb_n.T)
    return pi, float(np.sum(pi * M)) * scale ** 2, stages, scale


def pair_accuracy(pi: np.ndarray, idx_a, idx_b, inv_sigma: np.ndarray) -> tuple[float, int]:
    """Permuted regime: the share of a-landmarks whose heaviest b-column is their true image,
    over the a-landmarks whose image is among the b-landmarks. inv_sigma[i] is the b-index of
    the image of a-point i."""
    n = len(inv_sigma)
    ia = np.arange(n) if idx_a is None else np.asarray(idx_a)
    ib = np.arange(n) if idx_b is None else np.asarray(idx_b)
    pos_b = {int(j): c for c, j in enumerate(ib)}
    target = np.array([pos_b.get(int(inv_sigma[i]), -1) for i in ia])
    has = target >= 0
    if not has.any():
        return float("nan"), 0
    return float((pi.argmax(1)[has] == target[has]).mean()), int(has.sum())


def run(density: str, sample_mode: str, seed: int, out_root: str, cfg: dict,
        n: int, landmark_ks: list, gw_densities: list, save_couplings: bool) -> str:
    import jax
    jax.config.update("jax_enable_x64", True)
    np.random.seed(0)

    seed_a, seed_b = 2 * seed, 2 * seed + 1
    ldd_cfg, gw_cfg = cfg["ldd"], cfg["gw_anneal"]
    couplings = ["ldd_lp"] + (["gw"] if density in gw_densities else [])
    gw_schedule = build_eps_schedule(float(gw_cfg["eps_start"]), float(gw_cfg["eps_final"]),
                                     n_steps=int(gw_cfg["n_steps"]))
    r_max = np.pi if str(ldd_cfg["r_max"]) == "pi" else float(ldd_cfg["r_max"])
    logging.info(f"density={density} mode={sample_mode} seed={seed} n={n} couplings={couplings} "
                 f"gw_schedule={gw_schedule} measures_sha={sphere_measures_hash()}")

    # the two samples and the rotation that relates them
    rot = load_sphere_measures()["rotation"]
    rotation_axis = np.array(rot["axis"], dtype=float)
    rotation_angle = float(np.radians(float(rot["angle_deg"])))
    density_kwargs = sphere_measure(density)
    Za, Zb, R_star = make_sphere_experiment(n=n, density=density, rotation_axis=rotation_axis,
                                            rotation_angle=rotation_angle, seed_a=seed_a,
                                            seed_b=seed_b, **density_kwargs)
    sigma = inv_sigma = None
    if sample_mode == "permuted":
        sigma = np.random.default_rng(seed_b).permutation(n)
        inv_sigma = np.argsort(sigma)
        Zb = (Za @ R_star.T)[sigma].astype(np.float32)

    # geodesic distances and signatures on one radius grid shared by both sides
    Ca = sphere_distance_matrix(Za)
    Cb = Ca[np.ix_(sigma, sigma)] if sample_mode == "permuted" else sphere_distance_matrix(Zb)
    Ha, radii_a = compute_ldd_signatures(Ca, n_bins=int(ldd_cfg["n_bins"]), use_quantiles=False,
                                         r_max=r_max)
    Hb, _ = compute_ldd_signatures(Cb, n_bins=int(ldd_cfg["n_bins"]), use_quantiles=False,
                                   r_max=r_max)
    dr = float(radii_a[1] - radii_a[0])

    # one D-opt ordering per side; the landmarks of every k are a prefix of it
    k_max = max([k for k in landmark_ks if k < n] or [1])
    t0 = time.time()
    order_a, switch_a = greedy_dopt_selection(Ha, k_max, fill="maxmin", return_switch=True)
    order_b, switch_b = greedy_dopt_selection(Hb, k_max, fill="maxmin", return_switch=True)
    logging.info(f"D-opt orderings to k={k_max} in {time.time() - t0:.1f}s; farthest-point "
                 f"sampling after {switch_a} (a) and {switch_b} (b) landmarks")

    ckpt_root = os.path.join(out_root, density, "round", sample_mode, f"seed{seed}")
    os.makedirs(ckpt_root, exist_ok=True)
    results = {
        "density": density, "geometry": "round", "sample_mode": sample_mode, "seed": seed,
        "n": n, "R_star": R_star, "sigma": sigma, "density_kwargs": density_kwargs,
        "couplings": couplings, "ldd_lp_solver": "emd", "ldd_lp_cost": "cdf",
        "gw_schedule": gw_schedule, "gw_anneal": dict(gw_cfg), "map_fit": "closed_form",
        "ldd_grid": {"n_bins": int(ldd_cfg["n_bins"]), "use_quantiles": False, "r_max": r_max,
                     "dr": dr},
        "dopt_fill": "maxmin",
        "dopt_switch": {"a": int(switch_a), "b": int(switch_b), "k_max": int(k_max)},
        "provenance": {"sphere_measures_sha": sphere_measures_hash()},
        "cases": {},
    }

    cases = [("full", "all")] + [(k, "dopt") for k in landmark_ks]
    for ci, (k_label, method) in enumerate(cases):
        logging.info(f"[{ci + 1}/{len(cases)}] k={k_label}")
        if k_label == "full":
            idx_a = idx_b = None
            Za_k, Zb_k, Ca_k, Cb_k, Ha_k, Hb_k = Za, Zb, Ca, Cb, Ha, Hb
        else:
            idx_a, idx_b = order_a[:k_label], order_b[:k_label]
            Za_k, Zb_k = Za[idx_a], Zb[idx_b]
            Ca_k, Cb_k = Ca[np.ix_(idx_a, idx_a)], Cb[np.ix_(idx_b, idx_b)]
            Ha_k, Hb_k = Ha[idx_a], Hb[idx_b]
        case = {"k": k_label, "method": method, "idx_a": idx_a, "idx_b": idx_b,
                "coupling_results": {}}

        for coup in couplings:
            t0 = time.time()
            if coup == "ldd_lp":
                pi_k, diag = ldd_lp(ldd_lp_cost(Ha_k, Hb_k, dr))
                dist = diag["lp_cost"]
                diag.update(coupling_diagnostics(pi_k), timed_out=False)
            else:
                pi_k, dist, stages, scale = gw_annealed(Ca_k, Cb_k, gw_schedule,
                                                        int(gw_cfg["max_iter"]),
                                                        float(gw_cfg["tol"]))
                diag = coupling_diagnostics(pi_k)
                diag.update(timed_out=False, gw_stages=stages, gw_scale=scale,
                            gw_budget_limited=any(not st["converged"] for st in stages))
                logging.info("    GW stages: " + "; ".join(
                    f"eps={st['eps']:.3g} {st['iters']} it {'conv' if st['converged'] else 'BUDGET'}"
                    f" {st['seconds']:.1f}s" for st in stages))
            t_coup = time.time() - t0
            if sigma is not None:
                diag["pair_acc"], diag["pair_n"] = pair_accuracy(pi_k, idx_a, idx_b, inv_sigma)
            dist_key = "lh2_dist" if coup == "ldd_lp" else "gw2_dist"
            if not diag["is_valid"]:
                logging.warning(f"  [INVALID COUPLING] {coup} k={k_label}: "
                                f"mass={diag['total_mass']:.4g} row_err={diag['max_row_err']:.4g} "
                                f"col_err={diag['max_col_err']:.4g}")

            if save_couplings:
                coup_dir = os.path.join(ckpt_root, "couplings", f"{k_label}_{method}_{coup}")
                os.makedirs(coup_dir, exist_ok=True)
                if coup == "ldd_lp":            # an exact LP plan is a permutation: keep its support
                    r_, c_ = np.nonzero(pi_k)
                    np.savez(os.path.join(coup_dir, "coupling_sparse.npz"), rows=r_, cols=c_,
                             vals=pi_k[r_, c_], shape=np.array(pi_k.shape))
                else:
                    np.save(os.path.join(coup_dir, "coupling.npy"), pi_k.astype(np.float32))

            # the rotation: argmin over SO(3) of sum_ij pi_ij ||Q x_i - y_j||^2, in closed form
            t0 = time.time()
            R_pred = np.asarray(fit_orthogonal(pi_k, Za_k, Zb_k))
            t_map = time.time() - t0
            err_rad = rotation_error(R_pred, R_star)
            err_deg = float(np.degrees(err_rad))
            case["coupling_results"][coup] = {
                "err_rad": err_rad, "err_deg": err_deg, dist_key: dist,
                "coupling_valid": diag["is_valid"], "coupling_diag": diag, "timed_out": False,
                "t_coupling": t_coup, "t_map": t_map, "R_pred": R_pred}
            flags = ("" if diag["is_valid"] else " [INVALID]") \
                + (" [BUDGET-LIMITED]" if diag.get("gw_budget_limited") else "") \
                + (f" pair_acc={diag['pair_acc']:.3f}" if "pair_acc" in diag else "")
            logging.info(f"    {coup}: err={err_deg:.2f} deg  {dist_key}={dist:.4g}  "
                         f"t_coup={t_coup:.1f}s{flags}")
        results["cases"][(k_label, method)] = case

    path = os.path.join(ckpt_root, "results.pkl")
    with open(path, "wb") as fh:
        pickle.dump(results, fh, protocol=4)
    logging.info(f"wrote {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", required=True, choices=DENSITIES)
    ap.add_argument("--mode", required=True, choices=MODES, help="the sampling regime")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="results root: <out>/<density>/round/<mode>/seed<S>/")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "sphere.yaml"))
    ap.add_argument("--n-points", type=int, default=None, help="override the config (smoke tests)")
    ap.add_argument("--ks", type=int, nargs="+", default=None, help="override the config's landmark_ks")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(a.config) as fh:
        cfg = yaml.safe_load(fh)
    n = int(a.n_points if a.n_points is not None else cfg["n_points"])
    ks = list(a.ks if a.ks is not None else cfg["landmark_ks"])
    run(a.density, a.mode, a.seed, a.out, cfg, n, ks, list(cfg["gw_densities"]),
        bool(cfg.get("save_couplings", True)))


if __name__ == "__main__":
    main()
