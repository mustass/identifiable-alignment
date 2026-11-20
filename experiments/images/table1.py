#!/usr/bin/env python
"""Table 1 of the paper from the result files of the image experiments.

Reads, for every root `<dataset>_<prior>_<pairing>` (dataset mnist | fmnist | celeba, prior gauss |
vamp | cross, pairing two letters naming the two models) and every draw `s<S>`:

    results/fits/<root>_s<S>.json                        NGMA: combined.r_at_1 (fit_map.py)
    results/ceilings/<root>_s<S>.json                    NGMA fitted on the true pairs: a_to_b.oracle
                                                         and b_to_a.oracle (fit_map.py --ceiling)
    results/baselines/cached/<root>_s<S>.csv             Procrustes on the true pairs, RelReps-Centroid,
                                                         LFM-Unsupervised / -Labels / -Anchors
    results/baselines/relreps/<root>_s<S>/results.csv    RelReps (a -> b row)
    results/baselines/relgeo/<root>_s<S>/results.csv     RelGeo (a -> b row)
    results/baselines/genot/<root>_s<S>_{euclidean,pullback}.csv   GENOT-Q, each cost
    results/baselines/rosetta_wp/<root>_s<S>.json        Rosetta
    results/baselines/minivec2vec_official/<root>_s<S>.json        mini-vec2vec

Every number is R@1 on the paired test split. A cell is (dataset, prior). Within a cell, a method's
values are averaged per draw (the two oracle directions of a ceiling), then per model pairing over
its draws, and the cell mean is the mean over pairings; the printed spread is the s.d. over all
draws of the cell. Draws with R@1 below 0.3 are counted as failed, and a cell where any draw failed
prints the successes out of the draws. The best unpaired baseline of a cell is the highest mean
among LFM-Unsupervised, GENOT-Q (both costs), Rosetta and mini-vec2vec; the best baseline given
labels or anchors is the highest among RelReps-Centroid, RelReps, RelGeo, LFM-Labels and
LFM-Anchors.

Writes results/tables/Headline_summary_disjoint.csv (one row per cell) and
results/tables/main_results.tex (the LaTeX table).

    PYTHONPATH=src python experiments/images/table1.py [--results results] [--out results/tables]
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics as st

ROOT_RE = re.compile(r"^(mnist|fmnist|celeba)_(gauss|vamp|cross)_([a-d]{2})$")
DRAW_RE = re.compile(r"^(.+)_s(\d+)(?:_(euclidean|pullback))?$")
FAIL_R1 = 0.3
CELL_ORDER = ["mnist gauss", "mnist vamp", "mnist cross", "fmnist gauss", "fmnist vamp",
              "fmnist cross", "celeba gauss", "celeba vamp", "celeba cross"]
UNPAIRED = ["LFM-Unsupervised", "GENOT-Q, Euclidean cost", "GENOT-Q, pullback cost", "Rosetta", "mini-vec2vec"]
WEAK = ["RelReps-Centroid", "RelReps", "LFM-Anchors", "RelGeo", "LFM-Labels"]
SUPERVISED = "NGMA (supervised)"
DATASET = {"mnist": "MNIST", "fmnist": "FashionMNIST", "celeba": "CelebA"}
PRIOR = {"gauss": "Gaussian", "vamp": "VampPrior", "cross": "cross-prior"}
DATASET_ROWS = {"fmnist": ["Fashion-", "MNIST"]}
SYMBOLS = [r"\dag", r"\ddag", r"\S", r"\P", r"\|"]
OWN = [("Rosetta", "Rosetta"), ("mini_vec2vec", "mini-vec2vec")]


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def parse(name):
    """`<root>_s<S>[_cost]` -> (cell, pairing, seed, cost) or None."""
    m = DRAW_RE.match(name)
    if not m:
        return None
    r = ROOT_RE.match(m.group(1))
    if not r:
        return None
    return f"{r.group(1)} {r.group(2)}", r.group(3), int(m.group(2)), m.group(3)


def collect(results):
    """{(cell, method): {(pairing, seed): [values]}} from every result file."""
    rows = collections.defaultdict(lambda: collections.defaultdict(list))

    def add(name, method, value):
        key = parse(name)
        if key is None or value is None:
            return
        cell, pairing, seed, _ = key
        rows[(cell, method)][(pairing, seed)].append(value)

    for path in glob.glob(os.path.join(results, "fits", "*.json")):
        R = json.load(open(path))
        if os.path.basename(os.path.normpath(R.get("carve", ""))) != "disjoint":
            continue
        add(os.path.basename(path)[:-5], "NGMA", num((R.get("combined") or {}).get("r_at_1")))
    for path in glob.glob(os.path.join(results, "ceilings", "*.json")):
        R = json.load(open(path))
        for key in ("a_to_b", "b_to_a"):
            o = (R.get(key) or {}).get("oracle") or {}
            if (o.get("supervision") or {}).get("kind") == "gt_ab permutation":
                add(os.path.basename(path)[:-5], SUPERVISED, num(o.get("r_at_1")))
    for path in glob.glob(os.path.join(results, "baselines", "cached", "*.csv")):
        for r in csv.DictReader(open(path)):
            add(os.path.basename(path)[:-4], r["method"].split(" (")[0], num(r.get("r_at_1")))
    for sub, method in (("relreps", "RelReps"), ("relgeo", "RelGeo")):
        for path in glob.glob(os.path.join(results, "baselines", sub, "*", "results.csv")):
            for r in csv.DictReader(open(path)):
                if r.get("direction") == "a_to_b":
                    add(os.path.basename(os.path.dirname(path)), method, num(r.get("retrieval_at_1")))
    for path in glob.glob(os.path.join(results, "baselines", "genot", "*.csv")):
        name = os.path.basename(path)[:-4]
        cost = "pullback" if name.endswith("_pullback") else "Euclidean"
        for r in csv.DictReader(open(path)):
            add(name, f"GENOT-Q, {cost} cost", num(r.get("r_at_1")))
    for sub, method in (("rosetta_wp", "Rosetta"), ("minivec2vec_official", "mini-vec2vec")):
        for path in glob.glob(os.path.join(results, "baselines", sub, "*.json")):
            try:
                v = json.load(open(path))["pools"]["carve"]["a_to_b"]["r_at1"]
            except (KeyError, ValueError):
                continue
            add(os.path.basename(path)[:-5], method, num(v))
    return rows


def aggregate(per_draw):
    """mean over pairings of the per-pairing mean over draws; sd and draw list over all draws."""
    by_pairing = collections.defaultdict(list)
    draws = []
    for (pairing, seed), vs in sorted(per_draw.items()):
        v = st.mean(vs)
        by_pairing[pairing].append(v)
        draws.append(v)
    means = [st.mean(v) for v in by_pairing.values()]
    return {"mean": st.mean(means), "sd": st.stdev(draws) if len(draws) > 1 else None,
            "n_pairings": len(means), "draws": draws}


def pm(a):
    """`0.828±0.004`; a single-pairing cell prints its draws instead of an s.d."""
    if a is None:
        return ""
    if a["n_pairings"] < 2:
        if len(a["draws"]) > 1:
            return f"{a['mean']:.3f} (" + " / ".join(f"{v:.3f}" for v in a["draws"]) + ")"
        return f"{a['mean']:.3f}"
    return f"{a['mean']:.3f}" if a["sd"] is None else f"{a['mean']:.3f}±{a['sd']:.3f}"


def ok(a):
    return "" if a is None else f"{sum(v >= FAIL_R1 for v in a['draws'])}/{len(a['draws'])}"


def summary(rows):
    agg = {k: aggregate(v) for k, v in rows.items()}
    cells = [c for c in CELL_ORDER if any(k[0] == c for k in agg)]
    out = []
    for cell in cells:
        get = lambda m: agg.get((cell, m))  # noqa: E731
        best = lambda tier: max((m for m in tier if get(m)), key=lambda m: get(m)["mean"], default=None)  # noqa: E731
        unp, weak = best(UNPAIRED), best(WEAK)
        row = {"cell": cell, "NGMA": pm(get("NGMA")), "NGMA_ok": ok(get("NGMA")),
               "NGMA_supervised": pm(get(SUPERVISED)), "Procrustes_true_pairs": pm(get("Procrustes"))}
        for method, key in (("Rosetta", "Rosetta"), ("mini-vec2vec", "mini_vec2vec")):
            row[key], row[f"{key}_ok"] = pm(get(method)), ok(get(method))
        row.update({"best_unpaired": pm(get(unp)) if unp else "", "best_unpaired_method": unp or "",
                    "best_unpaired_ok": ok(get(unp)) if unp else "",
                    "best_weak": pm(get(weak)) if weak else "", "best_weak_method": weak or "",
                    "best_weak_supervision": ("class labels" if weak and weak.startswith(("RelReps", "RelGeo"))
                                              else "paired anchors" if weak == "LFM-Anchors" else
                                              "class labels" if weak else "")})
        out.append(row)
    return out


# --------------------------------------------------------------------------- LaTeX
NUM = r"(\d\.\d{3})"


def _parse(s):
    """'0.963±0.025' -> ('0.963', '0.025'); '0.677 (0.090 / ...)' or '0.677' -> ('0.677', None)."""
    s = (s or "").strip()
    m = re.match(NUM + r"±" + NUM + r"$", s)
    if m:
        return m.group(1), m.group(2)
    m = re.match(NUM + r"(?: \((.*)\))?$", s)
    if m:
        return m.group(1), None
    raise ValueError(f"cannot parse {s!r}")


def _num(s, mark=""):
    mean, sd = _parse(s)
    tex = f"{mean}{{\\scriptsize$\\pm${sd}}}" if sd else mean
    return tex + (f"\\textsuperscript{{{mark}}}" if mark else "")


def _count(ok_):
    k, _, n = (ok_ or "").partition("/")
    return f"\\,{{\\scriptsize({k}/{n})}}" if n and k != n else ""


def _winners(rows, col):
    names = [rows[c][col] for c in CELL_ORDER if c in rows]
    n = collections.Counter(names)
    usual = max(names, key=lambda m: (n[m], -names.index(m)))
    return usual, [m for m in dict.fromkeys(names) if m != usual]


def _legend(usual, others, sym):
    if not others:
        return usual
    return (f"{usual}, or " + " and ".join(f"{m} where marked \\textsuperscript{{{sym[m]}}}" for m in others))


def latex(rows_list, out):
    rows = {r["cell"]: r for r in rows_list}
    u_usual, u_other = _winners(rows, "best_unpaired_method")
    w_usual, w_other = _winners(rows, "best_weak_method")
    sym = dict(zip(u_other + w_other, SYMBOLS))
    lines = []
    cells = [c for c in CELL_ORDER if c in rows]
    for i, cell in enumerate(cells):
        r = rows[cell]
        ds, prior = cell.split()
        if i % 3 == 0 and i:
            lines.append(r"\midrule")
        label = DATASET_ROWS.get(ds, [DATASET[ds]])
        lines.append(f"{label[i % 3] if i % 3 < len(label) else ''} & "
                     f"{PRIOR[prior]} & {_num(r['NGMA'])}{_count(r['NGMA_ok'])} & "
                     f"{_num(r['best_unpaired'], sym.get(r['best_unpaired_method'], ''))}"
                     f"{_count(r['best_unpaired_ok'])} & "
                     f"{_num(r['best_weak'], sym.get(r['best_weak_method'], ''))} & "
                     f"{_num(r['NGMA_supervised'])} & {_num(r['Procrustes_true_pairs'])} \\\\")
    caption = (
        "Instance retrieval (R@1) among the $10{,}000$ test points ($19{,}962$ on CelebA) between "
        "independently trained VAEs, fitted on disjoint samples ($n=3000$ per side, $7500$ on CelebA), "
        "as mean $\\pm$ s.d.\\ over model pairings and draws (Appendix~\\ref{app:exp-setup}). "
        "NGMA uses no pairs and no labels. "
        f"The best unpaired baseline is {_legend(u_usual, u_other, sym)}. "
        f"The best baseline given class labels or anchor pairs is {_legend(w_usual, w_other, sym)}. "
        "The paired ceilings fit NGMA's map and a Procrustes rotation on the true pairs. "
        f"Where any draw has R@1 below ${FAIL_R1}$, the parentheses count the draws that reach it.")
    tex = "\n".join([
        "% Generated by experiments/images/table1.py from results/tables/Headline_summary_disjoint.csv. "
        "Do not edit by hand; rebuild.",
        r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{tab:alignment}",
        r"\vspace{4pt}", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{ll ccccc}", r"\toprule",
        r" & & & \multicolumn{2}{c}{Best baseline} & \multicolumn{2}{c}{Paired ceiling} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & Prior & NGMA & unpaired & weak corr. & NGMA & Procrustes \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    open(out, "w").write(tex)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="default: <results>/tables")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "tables")
    os.makedirs(out, exist_ok=True)
    rows = summary(collect(a.results))
    if not rows:
        raise SystemExit(f"no result files under {a.results}")
    cols = list(rows[0].keys())
    with open(os.path.join(out, "Headline_summary_disjoint.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    latex(rows, os.path.join(out, "main_results.tex"))
    for r in rows:
        print(f"{r['cell']:14s} NGMA {r['NGMA']:14s} ({r['NGMA_ok']})  unpaired {r['best_unpaired']:14s} {r['best_unpaired_method']:22s} "
              f"weak {r['best_weak']:14s} {r['best_weak_method']:16s} ceiling {r['NGMA_supervised']:14s} Procrustes {r['Procrustes_true_pairs']}")
    print(f"wrote {out}/Headline_summary_disjoint.csv and main_results.tex")


if __name__ == "__main__":
    main()
