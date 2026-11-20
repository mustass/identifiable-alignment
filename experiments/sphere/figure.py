#!/usr/bin/env python
"""Fig. 3: data on the sphere.

  (a) the rotation error of the LDD-LP on independent samples against the number of landmarks k,
      for the four measures: the median over seeds with a band from the best to the worst seed;
      with permuted samples the error is zero for every measure (one black line);
  (b) on P_mix, the LDD-LP, NGMA's coupling and annealed entropic GW alone;
  (c) the margin test: xi_hat(t) = m_hat(t) / 2 eps_N at `--margin-k` query points, mean and s.d.
      over seeds, up to the largest t at which every measure keeps `--min-pairs` query pairs.

Inputs: sphere_err_raw.json and sphere_full_raw.json from experiments/sphere/collect.py (the
full-cloud runs end the k axis at k = n) and the directory of experiments/sphere/margin_test.py.

    PYTHONPATH=src python experiments/sphere/figure.py --raw results/sphere/sphere_err_raw.json \\
        --full results/sphere/sphere_full_raw.json --margin-sweep results/sphere/margin \\
        --out figures/sphere_main.pdf
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles

# the four measures: label, colour, dash pattern, marker
MEASURES = [("uniform", r"$P_{\mathrm{unif}}$, $\mathrm{Aut}=O(3)$", "#5c5c5c", (None, None), "o"),
            ("vmf", r"$P_{\mathrm{vMF}}$, $\mathrm{Aut}\cong O(2)$", "#a3a3a3", (4, 1.4), "v"),
            ("mixture_vmf_eq", r"$P_{\mathrm{mix}}$, equal parameters", "#e87ba4", (1.6, 1.4), "s"),
            ("mixture_vmf", r"$P_{\mathrm{mix}}$", "#4a3aa7", (5, 1.4, 1.4, 1.4), "D")]
# the three couplings of panel (b): label, key, colour, marker, dash pattern (None: solid). Entropic
# GW alone is dashed with hollow markers, so it stays visible where it lands on NGMA's coupling.
COUPLINGS = [("LDD-LP", "ldd_lp", "#eb6834", "s", None),
             ("NGMA's coupling", "fgw", "#2a78d6", "o", None),
             ("entropic GW", "gw", "#5c5c5c", "o", (4, 1.6))]


def style(ncols):
    return bundles.neurips2024(usetex=False, ncols=ncols, nrows=1, family="serif")


def _axes(ax):
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def load_landmarks(path, coupling, method="dopt"):
    """(density, mode) -> {k: [err_deg per seed]} from sphere_err_raw.json."""
    raw = json.load(open(path))
    pat = re.compile(rf"results/(\d+)/{re.escape(method)}/{re.escape(coupling)}/err_deg$")
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for v in raw.values():
        if v.get("geometry") != "round" or not v.get("density"):
            continue
        for key, err in (v.get("errs") or {}).items():
            m = pat.match(key)
            if m and err is not None:
                out[(v["density"], v["sample_mode"])][int(m.group(1))].append(float(err))
    return out


def load_full(path, coupling):
    """(density, mode) -> {n_points: [err_deg per seed]} from sphere_full_raw.json: the full
    cloud, every point a landmark, so k = n."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for v in json.load(open(path)).values():
        if v.get("geometry") != "round" or not v.get("n_points"):
            continue
        e = (v.get("errs") or {}).get(f"results/full/all/{coupling}/err_deg")
        if e is not None:
            out[(v["density"], v["sample_mode"])][int(v["n_points"])].append(float(e))
    return out


def load_margin_sweep(path, k):
    """One row per (density, seed, t) at k query points, from the margin test's JSON files."""
    rows = []
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        r = json.load(open(f))
        v = r["by_k"].get(str(k))
        if v is None:
            raise SystemExit(f"{f} has no k = {k}")
        rows += [{"density": r["density"], "seed": r["seed"], "t": t, "m_hat": m, "floor": v["floor"],
                  "n_pairs": n} for t, m, n in zip(r["t"], v["m_hat"], v["n_pairs"]) if m is not None]
    return rows


def sphere_figure(raw, full_raw, margin_sweep, out, margin_k=2000, min_pairs=1000):
    series_of = {key: load_landmarks(raw, key) for _, key, *_ in COUPLINGS}
    ldd, gw = series_of["ldd_lp"], series_of["gw"]
    for coupling, series in series_of.items():
        for key, by_n in load_full(full_raw, coupling).items():
            if len(by_n) != 1 or any(n in series[key] for n in by_n):
                raise SystemExit(f"full-cloud runs of {key} at n = {sorted(by_n)} clash with the k grid")
            series[key].update(by_n)
    rows = load_margin_sweep(margin_sweep, margin_k)
    with plt.rc_context(style(3)):
        fig, (a, c, b) = plt.subplots(1, 3, figsize=(5.5, 1.65))
        for dens, label, colour, dash, marker in MEASURES:
            s = ldd.get((dens, "independent"), {})
            ks = sorted(k for k in s if s[k])
            a.fill_between(ks, [min(s[k]) for k in ks], [max(s[k]) for k in ks],
                           color=colour, alpha=0.15, linewidth=0)
            a.plot(ks, [st.median(s[k]) for k in ks], color=colour, dashes=dash, marker=marker,
                   markersize=2.5, linewidth=1.1, label=label)
        # permuted samples: every measure, k and seed at 0, so one line, the worst seed of all four
        perm = [ldd.get((d, "permuted"), {}) for d, *_ in MEASURES]
        kp = sorted(set.intersection(*[set(p) for p in perm]))
        if kp:
            a.plot(kp, [max(max(p[k]) for p in perm) for k in kp], color="black", linewidth=0.9,
                   marker="o", markersize=1.8)
            a.text(kp[len(kp) // 2], 0.1, "permuted samples, all four", fontsize=6, ha="center",
                   va="bottom")
        for name, key, colour, marker, dash in COUPLINGS:
            s = series_of[key].get(("mixture_vmf", "independent"), {})
            ks = sorted(k for k in s if s[k])
            med = [st.median(s[k]) for k in ks]
            lo, hi = [min(s[k]) for k in ks], [max(s[k]) for k in ks]
            c.fill_between(ks, lo, hi, color=colour, alpha=0.18, linewidth=0)
            c.plot(ks, med, color=colour, marker=marker, markersize=2.5, linewidth=1.1, label=name,
                   **({"dashes": dash, "markerfacecolor": "white"} if dash else {}))
        for ax, title in ((a, "(a) LDD-LP, four measures"), (c, r"(b) $P_{\mathrm{mix}}$")):
            ax.axhline(5.0, color="black", linewidth=0.6, linestyle=":")
            ax.set_xscale("log", base=2)
            last = max(k for s in (ldd, gw) for d in s.values() for k in d)
            ticks = [4, 16, 64, 256] + ([last] if last > 1024 else [1024])
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.minorticks_off()
            # linear on [0, 1] so an exact 0 has a place, logarithmic above
            ax.set_yscale("symlog", linthresh=1.0, linscale=0.35)
            ax.set_ylim(-0.15, 200)
            ax.set_yticks([0, 1, 10, 100])
            ax.set_yticklabels(["0", "1", "10", "100"])
            ax.set_xlabel("landmarks $k$")
            ax.set_title(title, loc="left", fontsize=8)
            _axes(ax)
        a.set_ylabel("rotation error (°)")
        # lower left: below 5 degrees at small k nothing is drawn, while GW fills the upper right
        c.legend(frameon=False, fontsize=6, loc="lower left", handlelength=1.3, labelspacing=0.2,
                 handletextpad=0.4, borderaxespad=0.1, borderpad=0.1)

        by = collections.defaultdict(lambda: collections.defaultdict(list))
        npairs = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            by[r["density"]][float(r["t"])].append(float(r["m_hat"]) / float(r["floor"]))
            npairs[r["density"]][float(r["t"])].append(int(r["n_pairs"]))
        t_max = min(max(t for t in npairs[d] if t > 0 and min(npairs[d][t]) >= min_pairs)
                    for d, *_ in MEASURES)
        for dens, label, colour, dash, marker in MEASURES:
            ts = sorted(t for t in by[dens] if 0 < t <= t_max)
            mean = np.array([np.mean(by[dens][t]) for t in ts])
            sd = np.array([np.std(by[dens][t], ddof=1) if len(by[dens][t]) > 1 else 0 for t in ts])
            b.plot(ts, mean, color=colour, dashes=dash, linewidth=1.1)
            b.fill_between(ts, np.clip(mean - sd, 1e-3, None), mean + sd, color=colour,
                           alpha=0.15, linewidth=0)
        b.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
        b.text(0.42, 0.64, "fails to reject below", fontsize=6, ha="center", va="center",
               transform=b.get_yaxis_transform())
        b.set_yscale("log")
        b.set_xlabel("separation $t$")
        b.set_ylabel(r"$\hat\xi(t)$")
        b.set_title("(c) margin test", loc="left", fontsize=8)
        _axes(b)
        h, l = a.get_legend_handles_labels()
        fig.legend(h, l, loc="outside upper center", ncol=4, frameon=False, fontsize=6.5)
        fig.savefig(out)
        plt.close(fig)
    return {"t_max": t_max, **{key: {k: (st.median(v), max(v)) for k, v in
                                     sorted(series_of[key].get(("mixture_vmf", "independent"), {}).items())}
                               for _, key, *_ in COUPLINGS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="sphere_err_raw.json")
    ap.add_argument("--full", required=True, help="sphere_full_raw.json")
    ap.add_argument("--margin-sweep", required=True, help="the margin test's output directory")
    ap.add_argument("--out", required=True, help="the figure, e.g. sphere_main.pdf")
    ap.add_argument("--margin-k", type=int, default=2000, help="query points shown in (c)")
    ap.add_argument("--min-pairs", type=int, default=1000,
                    help="(c) ends at the largest t where every measure keeps this many query pairs")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    info = sphere_figure(a.raw, a.full, a.margin_sweep, a.out, a.margin_k, a.min_pairs)
    print(f"wrote {a.out}; (c) up to t = {info['t_max']:.3f}")
    for name, key, *_ in COUPLINGS:
        print(f"  {name} on P_mix, independent samples: " + ", ".join(
            f"k={k}: median {m:.2f} / worst {w:.2f}" for k, (m, w) in info[key].items()))


if __name__ == "__main__":
    main()
