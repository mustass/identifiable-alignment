#!/usr/bin/env python
"""NGMA's map: two networks fitted from the geometry, with the coupling as the start.

Inputs are one carve directory (the two aligned clouds `Za`, `Zb`, their cost matrices
`C_<mode>_{a,b}`, the metric tensors `G_ensemble_eol_samedec_{a,b}`, and the paired test clouds
`Za_test`, `Zb_test`, `labels_test`) and the coupling `pi.npy` solved on it.

The fit, in order:

  1. Each cloud is centred and divided by its mean norm; the metric tensors are rescaled to
     match and the two cost matrices are divided by the mean of their two means.
  2. The coupling's targets: for every source point, the target with the largest coupling mass.
  3. The rigid start: the coupling-weighted Procrustes map over O(d) plus a translation, in
     closed form. The reverse map starts at its inverse.
  4. Two maps, phi(x) = W x + b + f(x) from a to b and psi(y) = W' y + b' + f'(y) from b to a,
     with f and f' two-hidden-layer GELU networks of width `--hidden` whose last layers start
     at zero, so training starts at the rigid pair. All parameters train; none are shared.
  5. The loss on a batch of a points and a batch of b points:
         iso_a(phi) + iso_b(psi) + fit_weight * [fit_a(phi) + fit_b(psi)]
         + cycle_weight * [|psi(phi(x)) - x|^2 + |phi(psi(y)) - y|^2]
     where iso is the pullback-isometry residual |J^T G_b(snap phi(x)) J - G_a(x)|^2 / |G_a|^2
     per point (the target metric read at the nearest target point of the mapped point) and
     fit is the squared distance to the coupling's target. Adam, `--steps` steps of `--batch`
     points per side.
  6. Selection without pairs: `--holdout` a rows (and the b rows whose targets they are) enter
     no loss. Every `--eval-every` steps the pair is scored by the mean of the two held-out
     mutual-nearest-neighbour fractions and the best checkpoint is kept; the rigid start is a
     candidate too.
  7. Joint refinement: up to `--joint-rounds` rounds of 1000 steps. The two directions are
     combined into one ranking on the aligned clouds, its mutual nearest neighbours become the
     fit targets of their rows and fill half of every batch, and the pair is trained on. The
     round with the largest held-out combined mutual-NN fraction is kept, the state before the
     first round included.

The reported number is the combined ranking on the test clouds: each direction's distances
divided by their median, weighted by the squared held-out mutual-NN fraction of that direction;
`combined.r_at_1` is the fraction of test points whose true partner ranks first.

The oracle (`--oracle --oracle-carve <carve>`) trains the same network on the true pairs of a
carve that holds the same images on both sides, starting at the Procrustes map fitted on those
pairs, and scores it on the full test set; that Procrustes map is scored too, as `rigid_closed`.
Inside a fit it is run for the a -> b direction and never enters the unsupervised fit.
`--ceiling --oracle-carve <carve>` trains no unsupervised map at all: it evaluates the rigid
start and the oracle in both directions (b -> a on the swapped clouds), which is the ceiling of
the map class that the paper's main table reports.

Outputs: the JSON at `--out` (`a_to_b`, `b_to_a`, `joint`, `joint_round0`, `combined`, `kept`,
`args`) and, unless `--no-save-mapped` or `--ceiling`, `<out>_mapped.npz` with the mapped test
clouds, the target clouds, the labels and the combination weights.

    fit_map.py <carve> --pi <carve>/alignments/ensemble_eol_samedec_kn_FGWG_ann/pi.npy \\
        [--oracle --oracle-carve <frac100 carve>] --out results/<name>.json
    fit_map.py <carve> --pi <pi> --ceiling --oracle-carve <frac100 carve> --out results/oracle_<name>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from ngma.alignment.closed_form import fit_orthogonal, orientation_pair  # noqa: E402
from ngma.evaluation.coupling_metrics import csls_retrieval, retrieval_from_D  # noqa: E402
from ngma.geometry.metric_tensors import whiten_apply, whiten_metric, whiten_unapply  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

#: Loss weights (gm, iso, fit) of the two candidates. `gm` is the soft Gromov-Monge term, which
#: the method does not train on; it is evaluated on the held-out rows and by the symmetry test.
VARIANTS = {"rigid": (0.0, 0.0, 0.0),      # the start, no training
            "iso+fit": (0.0, 1.0, 0.3)}
#: The weight of the cycle term, set from --cycle-weight before anything is traced.
CYCLE = {"w": 0.0}
#: Temperature of the soft snap, annealed geometrically from TAU0 to TAU1 over the steps.
TAU0, TAU1 = 1.0, 0.05
#: Steps of each joint round, and the growth of the anchor set below which the rounds stop.
ALT_STEPS, ANCHOR_STOP = 1000, 0.02
#: Random rotations of the start in the symmetry test, and its reference gap.
SYMMETRY_ROTATIONS, SYMMETRY_REF = 8, 1.0


# --------------------------------------------------------------------------- #
# the map
# --------------------------------------------------------------------------- #
def init_params(key, d, hidden, W0, b0):
    """W x + b + f(x) with f a two-hidden-layer GELU network whose last layer is zero."""
    k1, k2 = jax.random.split(key)
    return {"W": jnp.asarray(W0, jnp.float32), "b": jnp.asarray(b0, jnp.float32),
            "f1": jax.random.normal(k1, (d, hidden)) / np.sqrt(d), "c1": jnp.zeros((hidden,)),
            "f2": jax.random.normal(k2, (hidden, hidden)) / np.sqrt(hidden), "c2": jnp.zeros((hidden,)),
            "f3": jnp.zeros((hidden, d)), "c3": jnp.zeros((d,))}


def linear(p, x):
    return x @ p["W"].T + p["b"]


def residual(p, x):
    h = jax.nn.gelu(x @ p["f1"] + p["c1"])
    h = jax.nn.gelu(h @ p["f2"] + p["c2"])
    return h @ p["f3"] + p["c3"]


def phi(p, x):
    return linear(p, x) + residual(p, x)


def phi_inv(p, y):
    """The reverse map psi, the second network of the pair, held in p["rev"]."""
    return linear(p["rev"], y) + residual(p["rev"], y)


def residual_inv(p, y):
    return residual(p["rev"], y)


# --------------------------------------------------------------------------- #
# the loss
# --------------------------------------------------------------------------- #
def make_loss(Zb, Cb, Gb, lam, map_fn=None, full=False):
    """The losses of one direction against the target cloud Zb with costs Cb and metrics Gb.

    lam = (gm, iso, fit). `full` evaluates every term whatever its weight (the held-out
    evaluation); the training step skips a term of weight zero. map_fn is phi, or phi_inv for
    the reverse direction."""
    map_fn = phi if map_fn is None else map_fn
    compute_gm = full or lam[0] > 0
    lam_gm, lam_iso, lam_fit = lam
    Zb = jnp.asarray(Zb, jnp.float32); Cb = jnp.asarray(Cb, jnp.float32)
    Gb = jnp.asarray(Gb, jnp.float32)
    jac = jax.vmap(jax.jacfwd(lambda p, x: map_fn(p, x[None])[0], argnums=1), in_axes=(None, 0))

    def loss(p, X, CaB, GaB, T, tau, lam_fit_rt=None):
        P = map_fn(p, X)                                                 # (B, d)
        D2 = (P ** 2).sum(1)[:, None] + (Zb ** 2).sum(1)[None] - 2.0 * P @ Zb.T
        w = jax.nn.softmax(-D2 / tau, axis=1)                            # (B, n_b) soft snap
        out = {}
        if compute_gm:
            Chat = w @ Cb @ w.T                                          # (B, B)
            mask = 1.0 - jnp.eye(X.shape[0])
            out["gm"] = (((Chat - CaB) ** 2) * mask).sum() / mask.sum()
        else:
            out["gm"] = jnp.float32(0.0)
        snap = jnp.argmax(w, axis=1)                                     # no gradient through the choice
        J = jac(p, X)                                                    # (B, d, d)
        pulled = jnp.einsum("bji,bjk,bkl->bil", J, Gb[snap], J)
        num = ((pulled - GaB) ** 2).sum((1, 2)); den = (GaB ** 2).sum((1, 2)) + 1e-9
        out["iso"] = (num / den).mean()
        out["fit"] = ((P - T) ** 2).sum(1).mean()
        lf = lam_fit if lam_fit_rt is None else lam_fit_rt
        total = lam_gm * out["gm"] + lam_iso * out["iso"] + lf * out["fit"]
        return total, out
    return loss


def _step_sym(p, st, X, CaB, GaB, T, Xb, CbB, GbB, Tb, tau, lf, loss_fn, loss_inv, opt):
    """One step of the pair: the a-side losses of phi on an a-batch, the b-side losses of psi
    on a b-batch, and the cycle term on both batches."""
    def both(p):
        la, _ = loss_fn(p, X, CaB, GaB, T, tau, lf)
        lb, _ = loss_inv(p, Xb, CbB, GbB, Tb, tau, lf)
        tot = la + lb
        if CYCLE["w"] > 0:
            cyc = ((phi_inv(p, phi(p, X)) - X) ** 2).sum(1).mean() \
                + ((phi(p, phi_inv(p, Xb)) - Xb) ** 2).sum(1).mean()
            tot = tot + CYCLE["w"] * cyc
        return tot, None
    (l, _), grads = jax.value_and_grad(both, has_aux=True)(p)
    upd, st = opt.update(grads, st, p)
    return optax.apply_updates(p, upd), st, l


def hard(pi):
    """Row argmax as a coupling: one target per source, mass 1/n."""
    n = pi.shape[0]
    h = np.zeros_like(pi, dtype=np.float64)
    h[np.arange(n), np.asarray(pi).argmax(1)] = 1.0 / n
    return h


def cdist(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return np.sqrt(np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :]
                              - 2.0 * A @ B.T, 0.0))


def score(mapped, Zb, labels=None):
    """Tie-averaged R@1 and R@5, CSLS@1 and CSLS@5 with the tie fraction, and the class accuracy
    of the nearest target, for mapped points against the target test cloud."""
    D = cdist(mapped, Zb)
    r = retrieval_from_D(D, ks=(1, 5))
    c = csls_retrieval(D, ks=(1, 5))
    out = {"r_at_1": r["at1"], "r_at_5": r["at5"], "csls_at1": c["at1"], "csls_at5": c["at5"],
           "csls_tie_frac": c["tie_frac"], "pool_n": r["n"]}
    if labels is not None:
        out["class_top1"] = float((labels[D.argmin(1)] == labels).mean())
    return out


def mnn_frac(p, Xs, Ys, rows=None, map_fn=None):
    """Fraction of sources that are mutual nearest neighbours with a target under the map,
    counted over `rows` (the held-out rows) or over every row. Every source is mapped and
    every target competes; only the rows averaged over change."""
    Pv = np.asarray((phi if map_fn is None else map_fn)(p, jnp.asarray(Xs, jnp.float32)), np.float64)
    Dv = (Pv ** 2).sum(1)[:, None] + (Ys ** 2).sum(1)[None] - 2.0 * Pv @ Ys.T
    av, bv = Dv.argmin(1), Dv.argmin(0)
    ok = bv[av] == np.arange(len(Xs))
    return float(ok.mean() if rows is None else ok[rows].mean())


# --------------------------------------------------------------------------- #
# the symmetry test
# --------------------------------------------------------------------------- #
def symmetry_gap(Q0, b0, Xs, Ys, Cb_n, Gb_s, Ga_s, hb, ca_sub, d):
    """Geometry losses of the rigid start against those of the start composed with random
    rotations of the target space, on the held-out rows: gap = median L(R phi0) / L(phi0) - 1.
    Near 0, the geometry cannot tell rotations apart and the fit weight is raised."""
    loss_fn = make_loss(Ys, Cb_n, Gb_s, (1.0, 1.0, 0.0))
    ev = jax.jit(lambda p, X, CaB, GaB, T: loss_fn(p, X, CaB, GaB, T, TAU1, jnp.float32(0.0))[1])
    args_hb = (jnp.asarray(Xs[hb], jnp.float32), jnp.asarray(ca_sub(hb), jnp.float32),
               jnp.asarray(Ga_s[hb], jnp.float32), jnp.zeros((len(hb), d), jnp.float32))
    rng = np.random.default_rng(12345)
    base = {k: float(v) for k, v in ev(init_params(jax.random.PRNGKey(0), d, 8, Q0, b0), *args_hb).items()}
    gm, iso = [], []
    for _ in range(SYMMETRY_ROTATIONS):
        Rm, _ = np.linalg.qr(rng.standard_normal((d, d)))
        if np.linalg.det(Rm) < 0:
            Rm[:, 0] = -Rm[:, 0]
        h = {k: float(v) for k, v in ev(init_params(jax.random.PRNGKey(0), d, 8, Rm @ Q0, Rm @ b0), *args_hb).items()}
        gm.append(h["gm"] / max(base["gm"], 1e-12)); iso.append(h["iso"] / max(base["iso"], 1e-12))
    return {"gm_start": base["gm"], "iso_start": base["iso"],
            "gap_gm": float(np.median(gm)) - 1.0, "gap_iso": float(np.median(iso)) - 1.0,
            "gap_gm_min": float(np.min(gm)) - 1.0}


# --------------------------------------------------------------------------- #
# the oracle: the same network on the true pairs
# --------------------------------------------------------------------------- #
def oracle_map(args, key, d, Xs_te, hb_args, Ys, Cb_n, Gb_s, Xs, unstd, Zb_te, lab, sup, sup_meta):
    """phi fitted on the known pairs `sup` (the oracle carve, in this run's coordinates) and
    scored on the full test set. It starts at the Procrustes map fitted on the same pairs,
    which is scored too as `rigid_closed`. Never a candidate for selection."""
    Xsup, Ysup = np.asarray(sup[0], np.float64), np.asarray(sup[1], np.float64)
    n_sup = len(Xsup)
    Qs, bs = fit_orthogonal(np.eye(n_sup) / n_sup, Xsup, Ysup, translation=True,
                            allow_reflection=True, scale=False)
    p = init_params(key, d, args.hidden, Qs, bs)
    Xtr, Ytr = jnp.asarray(Xsup, jnp.float32), jnp.asarray(Ysup, jnp.float32)
    info = {"kind": "gt_ab permutation", "train_pairs": int(n_sup), **(sup_meta or {})}
    fit = jax.jit(jax.value_and_grad(lambda p: ((phi(p, Xtr) - Ytr) ** 2).sum(1).mean()))
    opt = optax.adam(args.lr); st = opt.init(p)
    for _ in range(args.oracle_steps):
        _, g = fit(p); upd, st = opt.update(g, st, p); p = optax.apply_updates(p, upd)
    loss_fn = make_loss(Ys, Cb_n, Gb_s, (1.0, 1.0, 0.0))
    evf = jax.jit(lambda p, X, CaB, GaB, T: loss_fn(p, X, CaB, GaB, T, TAU1, jnp.float32(0.0))[1])
    held = {k: float(v) for k, v in evf(p, *hb_args).items()}
    Pv = np.asarray(phi(p, jnp.asarray(Xs, jnp.float32)), np.float64)
    Dv = (Pv ** 2).sum(1)[:, None] + (Ys ** 2).sum(1)[None] - 2.0 * Pv @ Ys.T
    av, bv = Dv.argmin(1), Dv.argmin(0)
    held["mutual_nn_frac"] = float((bv[av] == np.arange(len(Xs))).mean())
    Xe = jnp.asarray(Xs_te, jnp.float32)
    m = score(unstd(np.asarray(phi(p, Xe))), Zb_te, lab)
    fn = float(np.linalg.norm(np.asarray(residual(p, Xe)), axis=1).mean()
               / max(np.linalg.norm(np.asarray(phi(p, Xe)), axis=1).mean(), 1e-9))
    out = {"held": held, "residual_frac": fn, "train_pairs": info["train_pairs"],
           "supervision": info, **m}
    out["rigid_closed"] = score(unstd(np.asarray(Xs_te) @ Qs.T + bs), Zb_te, lab)
    out["rigid_closed"]["Q"] = np.asarray(Qs, np.float64).tolist()
    out["rigid_closed"]["b"] = np.asarray(bs, np.float64).tolist()
    return out


# --------------------------------------------------------------------------- #
# one direction: the rigid start, the pair's fit, the oracle
# --------------------------------------------------------------------------- #
def run_direction(args, S, tag, rng, key, train_pair):
    """From the carve's clouds S: standardise, build the rigid start and the held-out split,
    run the symmetry test, evaluate the rigid start, and with `train_pair` train the pair
    (phi and psi) and select its checkpoint. With `--oracle` the oracle is fitted too.
    Returns the record, the mapped test cloud and, when the pair was trained, the state the
    joint rounds continue from."""
    Za, Zb, Za_te, Zb_te, lab = S["Za"], S["Zb"], S["Za_te"], S["Zb_te"], S["lab"]
    Ca, Cb, Ga, Gb, pi_soft = S["Ca"], S["Cb"], S["Ga"], S["Gb"], S["pi_soft"]
    n, d = Za.shape
    pi = hard(pi_soft)
    R_targets = {"mode": "argmax",
                 "argmax_distinct_frac": float(len(np.unique(np.asarray(pi_soft).argmax(1))) / len(pi_soft))}

    # standardised coordinates; costs divided by the mean of their two means
    mu_a, mu_b = Za.mean(0), Zb.mean(0)
    Xc, Yc = whiten_apply(Za - mu_a, None), whiten_apply(Zb - mu_b, None)
    sa = float(np.linalg.norm(Xc, axis=1).mean()); sb = float(np.linalg.norm(Yc, axis=1).mean())
    Xs, Ys = Xc / sa, Yc / sb
    Xs_te = whiten_apply(Za_te - mu_a, None) / sa
    unstd = lambda P: whiten_unapply(P * sb, None) + mu_b
    sc = 0.5 * (Ca.mean() + Cb.mean()); Ca_n, Cb_n = Ca / sc, Cb / sc
    Ga_s, Gb_s = whiten_metric(Ga, None, sa), whiten_metric(Gb, None, sb)
    rows = np.arange(n)
    Ys_snap, Gb_snap = Ys[rows], Gb_s[rows]

    # the rigid start: coupling-weighted Procrustes over O(d) plus a translation; the
    # orientation is the one Procrustes returns, the mirror is recorded beside it
    (Qd, bd), (Qm, bm), orient = orientation_pair(pi_soft, Xs, Ys, translation=True, scale=False)
    Q0, b0 = Qd, bd
    s0 = float(np.linalg.norm(np.asarray(Q0, np.float64), 2))
    T = pi @ Ys / pi.sum(1, keepdims=True)

    # the held-out rows: at most a sixth of the cloud
    h = min(args.holdout, max(1, n // 6))
    if h != args.holdout:
        print(f"  holdout capped: asked {args.holdout}, using {h} of {n} rows (at most a sixth)")
    perm = rng.permutation(rows); hold, train = perm[:h], perm[h:]
    hb = hold[: min(len(hold), 512)]
    mnn_rows = hold

    ca_sub = lambda idx: Ca_n[np.ix_(np.asarray(idx, int), np.asarray(idx, int))]
    sym = symmetry_gap(Q0, b0, Xs, Ys_snap, Cb_n, Gb_snap, Ga_s, hb, ca_sub, d)
    fit_scale = 1.0
    if np.isfinite(sym["gap_gm"]):
        fit_scale = max(1.0, SYMMETRY_REF / max(sym["gap_gm"], 1e-6))
    print(f"  symmetry test: a random rotation of the start multiplies the gm loss by "
          f"{1 + sym['gap_gm']:.2f} (iso by {1 + sym['gap_iso']:.2f}; median of {SYMMETRY_ROTATIONS}) "
          f"-> gap g = {sym['gap_gm']:.2f}; fit weight x{fit_scale:.1f}")
    hb_args = (jnp.asarray(Xs[hb], jnp.float32), jnp.asarray(ca_sub(hb), jnp.float32),
               jnp.asarray(Ga_s[hb], jnp.float32), jnp.asarray(T[hb], jnp.float32))

    if train_pair:
        # the b side of the pair: psi is read on b rows and snapped onto the a training cloud
        # (the held-out a rows never enter a loss), with targets from the transposed coupling.
        # The b rows whose target is a held-out a row are never trained on either; they are
        # the held-out rows of the b -> a mutual-NN fraction.
        pi_t = hard(pi_soft.T)
        Tb0 = pi_t @ Xs / np.maximum(pi_t.sum(1, keepdims=True), 1e-300)
        partner = pi_t.argmax(1)
        b_train = np.where(np.isin(partner, train))[0]
        hold_b = np.where(np.isin(partner, hold))[0]
        tr = np.asarray(train, int)
        cb_sub = lambda idx: Cb_n[np.ix_(np.asarray(idx, int), np.asarray(idx, int))]
        symb = {"Tb": Tb0.copy(), "fit_rows": None}      # the b-side targets the joint rounds refresh

        def set_b_targets(Tb_new, fit_rows):
            symb["Tb"] = np.asarray(Tb_new, np.float64)
            fr = np.asarray(fit_rows, int); symb["fit_rows"] = fr[np.isin(fr, b_train)]

        # what selects: the mean of the two held-out mutual-NN fractions
        sel_stat = lambda p: 0.5 * (mnn_frac(p, Xs, Ys, mnn_rows) + mnn_frac(p, Ys, Xs, hold_b, map_fn=phi_inv))

        def sym_step(loss_fn, opt, lam):
            loss_inv = make_loss(Xs[tr], Ca_n[np.ix_(tr, tr)], Ga_s[tr], lam, map_fn=phi_inv)
            core = jax.jit(lambda p, st, X, CaB, GaB, T, Xb, CbB, GbB, Tb_, tau, lf:
                           _step_sym(p, st, X, CaB, GaB, T, Xb, CbB, GbB, Tb_, tau, lf, loss_fn, loss_inv, opt))

            def step(p, st, X, CaB, GaB, T, tau, lf=None):
                # half the b-batch from the anchor rows once the rounds have set them, the rest
                # from the b training rows
                fr = symb["fit_rows"]; nb = len(X)
                if fr is not None and len(fr):
                    ja = rng.choice(fr, size=min(nb // 2, len(fr)), replace=False)
                    jb = np.concatenate([ja, rng.choice(b_train, size=min(nb - len(ja), len(b_train)), replace=False)]).astype(int)
                else:
                    jb = rng.choice(b_train, size=min(nb, len(b_train)), replace=False)
                return core(p, st, X, CaB, GaB, T, jnp.asarray(Ys[jb], jnp.float32),
                            jnp.asarray(cb_sub(jb), jnp.float32), jnp.asarray(Gb_s[jb], jnp.float32),
                            jnp.asarray(symb["Tb"][jb], jnp.float32), tau,
                            jnp.float32(lam[2]) if lf is None else lf)
            return step

    R = {"direction": tag, "n": int(n), "d": int(d), "fit_targets": R_targets,
         "rigid_init": {"s": s0, **orient, "kept": "det", "Q": np.asarray(Q0, np.float64).tolist(),
                        "b": np.asarray(b0, np.float64).tolist()},
         "variants": {}, "symmetry": {**sym, "fit_scale": fit_scale},
         "holdout": {"requested": int(args.holdout), "used": int(h), "rows": int(n), "train": int(len(train))}}
    print(f"\n[{tag}] init: closed-form rigid on the coupling (s={s0:.3f}); "
          f"{len(train)} train / {len(hold)} held-out rows"
          + (f"; steps {args.steps}, batch {args.batch}, tau {TAU0} -> {TAU1}" if train_pair else ""))
    print(f"  {'variant':<12} | {'gm held':>8} {'iso held':>8} | {'mNN':>5} | "
          f"{'R@1':>6} {'R@5':>6} {'CSLS':>6} {'class':>6} | {'|f|/|phi|':>9} {'kept@':>6} {'time':>5}")

    params = {}
    for name in (("rigid", "iso+fit") if train_pair else ("rigid",)):
        lam = VARIANTS[name]
        lam = (lam[0], lam[1], (args.fit_weight if lam[2] > 0 else 0.0) * fit_scale)
        t0 = time.time()
        p = init_params(key, d, args.hidden, Q0, b0)
        if train_pair:
            # the pair's second map starts at the rigid start's inverse
            Qi = np.linalg.inv(np.asarray(Q0, np.float64))
            p = dict(p, rev=init_params(jax.random.PRNGKey(args.seed + 29), d, args.hidden,
                                        Qi, -Qi @ np.asarray(b0, np.float64)))
        loss_fn = make_loss(Ys_snap, Cb_n, Gb_snap, lam)
        loss_ev = make_loss(Ys_snap, Cb_n, Gb_snap, lam, full=True)
        eval_fn = jax.jit(lambda p, X, CaB, GaB, T: loss_ev(p, X, CaB, GaB, T, TAU1, jnp.float32(lam[2]))[1])
        # the start is a candidate: a trained checkpoint is kept only if it beats it
        best_p, best_it = p, 0
        best_c = float("inf") if name == "rigid" else -sel_stat(p)
        if name != "rigid":
            opt = optax.adam(args.lr); st = opt.init(p)
            step = sym_step(loss_fn, opt, lam)
            for it in range(args.steps):
                idx = rng.choice(train, size=min(args.batch, len(train)), replace=False)
                frac = it / max(args.steps - 1, 1)
                tau = TAU0 * (TAU1 / TAU0) ** frac
                p, st, _ = step(p, st, jnp.asarray(Xs[idx], jnp.float32), jnp.asarray(ca_sub(idx), jnp.float32),
                                jnp.asarray(Ga_s[idx], jnp.float32), jnp.asarray(T[idx], jnp.float32),
                                jnp.float32(tau), jnp.float32(lam[2]))
                if (it + 1) % args.eval_every == 0 or it == args.steps - 1:
                    c = -sel_stat(p)                                     # larger fraction = better
                    if c < best_c:
                        best_p, best_c, best_it = p, c, it + 1
            p = best_p
        held = {k: float(v) for k, v in eval_fn(p, *hb_args).items()}; held["best_step"] = best_it
        held["mutual_nn_frac"] = mnn_frac(p, Xs, Ys, mnn_rows)
        held["mutual_nn_frac_all"] = mnn_frac(p, Xs, Ys)
        mapped = unstd(np.asarray(phi(p, jnp.asarray(Xs_te, jnp.float32))))
        m = score(mapped, Zb_te, lab)
        fn = float(np.linalg.norm(np.asarray(residual(p, jnp.asarray(Xs_te, jnp.float32))), axis=1).mean()
                   / max(np.linalg.norm(np.asarray(phi(p, jnp.asarray(Xs_te, jnp.float32))), axis=1).mean(), 1e-9))
        R["variants"][name] = {"lam": lam, "held": held, "residual_frac": fn, **m, "seconds": time.time() - t0}
        params[name] = p
        print(f"  {name:<12} | {held['gm']:>8.4f} {held['iso']:>8.3f} | {held['mutual_nn_frac']:>5.3f} | "
              f"{m['r_at_1']:>6.3f} {m['r_at_5']:>6.3f} {m['csls_at1']:>6.3f} {m['class_top1']:>6.3f} | "
              f"{fn:>9.3f} {best_it:>6} {time.time() - t0:>4.0f}s")

    if args.oracle_carve:
        # the oracle carve in THIS run's coordinates; `moment_shift` records how far its means
        # sit from the run's own
        sup = ((S["sup_a"] - mu_a) / sa, (S["sup_b"] - mu_b) / sb)
        sup_meta = dict(S["sup_meta"])
        sup_meta["moment_shift"] = {"a": float(np.linalg.norm(S["sup_a"].mean(0) - mu_a) / sa),
                                    "b": float(np.linalg.norm(S["sup_b"].mean(0) - mu_b) / sb)}
        O = oracle_map(args, jax.random.PRNGKey(7), d, Xs_te, hb_args, Ys_snap, Cb_n, Gb_snap, Xs, unstd,
                       Zb_te, lab, sup, sup_meta)
        R["oracle"] = O
        Or = O["rigid_closed"]
        print(f"  {'ORACLE rigid':<12} | {'':>8} {'':>8} | {'':>5} | {Or['r_at_1']:>6.3f} {Or['r_at_5']:>6.3f} "
              f"{Or['csls_at1']:>6.3f} {Or['class_top1']:>6.3f} |   [Procrustes on the true pairs]")
        print(f"  {'ORACLE':<12} | {O['held']['gm']:>8.4f} {O['held']['iso']:>8.3f} | "
              f"{O['held']['mutual_nn_frac']:>5.3f} | {O['r_at_1']:>6.3f} {O['r_at_5']:>6.3f} "
              f"{O['csls_at1']:>6.3f} {O['class_top1']:>6.3f} | {O['residual_frac']:>9.3f} {'':>6}"
              f"   [the map class on {O['train_pairs']} true pairs, scored on {O.get('pool_n', '?')}; not a candidate]")

    best = max(R["variants"], key=lambda k: R["variants"][k]["held"]["mutual_nn_frac"])
    R["selected"] = best; R["selection"] = "mnn"
    R["selected_by_held_gm_plus_iso"] = best
    if train_pair:
        print(f"  selected by the largest held-out mutual-NN fraction over start, variants and checkpoints: "
              f"{best} (mNN {R['variants'][best]['held']['mutual_nn_frac']:.3f}) -> R@1 "
              f"{R['variants'][best]['r_at_1']:.3f}  (rigid init {R['variants']['rigid']['r_at_1']:.3f})")
    p = params[best]

    P = np.asarray(phi(p, jnp.asarray(Xs, jnp.float32)), np.float64)
    D2 = (P ** 2).sum(1)[:, None] + (Ys ** 2).sum(1)[None] - 2.0 * P @ Ys.T
    a2b = D2.argmin(1); b2a = D2.argmin(0)
    ok_rows = b2a[a2b] == np.arange(n)
    R["mutual_nn_frac_all"] = float(ok_rows.mean())
    R["mutual_nn_frac"] = float(ok_rows[mnn_rows].mean())
    R["mutual_nn_rows"] = "hold"
    mapped_te = unstd(np.asarray(phi(p, jnp.asarray(Xs_te, jnp.float32))))
    if not train_pair:
        return R, mapped_te, {"p": p, "Q0": np.asarray(Q0, np.float64)}
    # the joint rounds train the kept pair on under the trained candidate's loss
    lam_j = VARIANTS["iso+fit"]
    lam_j = (lam_j[0], lam_j[1], max(lam_j[2], args.fit_weight) * fit_scale)
    loss_j = make_loss(Ys_snap, Cb_n, Gb_snap, lam_j)
    opt_j = optax.adam(args.lr)
    state = {"p": p, "opt": opt_j, "st": opt_j.init(p), "step": sym_step(loss_j, opt_j, lam_j),
             "set_b_targets": set_b_targets,
             "Xs": Xs, "Ys": Ys, "Xs_te": Xs_te, "unstd": unstd, "T": T,
             "train": train, "ca_sub": ca_sub, "Ga_s": Ga_s, "n": n, "hold": hold, "mnn_rows": mnn_rows,
             "Q0": np.asarray(Q0, np.float64), "b0": np.asarray(b0, np.float64),
             "map": phi, "res": residual, "d": int(d), "lam": lam_j,
             "Ca_n": Ca_n, "Cb_n": Cb_n, "Gb_s": Gb_s,
             "std": {"mu_a": mu_a, "sa": sa, "mu_b": mu_b, "sb": sb}}
    return R, mapped_te, state


# --------------------------------------------------------------------------- #
# the b -> a direction: the pair's second network, read out
# --------------------------------------------------------------------------- #
def reverse_direction(args, A, S, rng):
    """The b -> a direction is psi, trained jointly with phi in run_direction. This scores it,
    with its own held-out split on the b rows and targets from the transposed coupling."""
    p = A["p"]; n, d = A["n"], A["d"]; std = A["std"]; lam = A["lam"]
    Xs, Ys = A["Ys"], A["Xs"]                        # the b cloud is the source now, a the target
    Xs_te = whiten_apply(np.asarray(S["Zb_te"]) - std["mu_b"], None) / std["sb"]
    unstd = lambda P: whiten_unapply(P * std["sa"], None) + std["mu_a"]
    pi_t = hard(S["pi_soft"].T)
    T = pi_t @ Ys / np.maximum(pi_t.sum(1, keepdims=True), 1e-300)
    h = len(A["hold"]); perm = rng.permutation(n); hold, train = perm[:h], perm[h:]
    hb = hold[: min(len(hold), 512)]
    mnn_rows = hold
    ca_sub = lambda idx: A["Cb_n"][np.ix_(np.asarray(idx, int), np.asarray(idx, int))]
    Ga_s = A["Gb_s"]
    loss_ev = make_loss(Ys, A["Ca_n"], A["Ga_s"], lam, map_fn=phi_inv, full=True)
    eval_fn = jax.jit(lambda p, X, CaB, GaB, T: loss_ev(p, X, CaB, GaB, T, TAU1, jnp.float32(lam[2]))[1])
    hb_args = (jnp.asarray(Xs[hb], jnp.float32), jnp.asarray(ca_sub(hb), jnp.float32),
               jnp.asarray(Ga_s[hb], jnp.float32), jnp.asarray(T[hb], jnp.float32))

    # the cycle residual on the forward map's own source rows
    probe = jnp.asarray(Ys[:256], jnp.float32)
    cyc_err = float(jnp.abs(phi_inv(p, phi(p, probe)) - probe).max())
    P = np.asarray(phi_inv(p, jnp.asarray(Xs, jnp.float32)), np.float64)
    D2 = (P ** 2).sum(1)[:, None] + (Ys ** 2).sum(1)[None] - 2.0 * P @ Ys.T
    a2b = D2.argmin(1); b2a = D2.argmin(0)
    ok_rows = b2a[a2b] == np.arange(n)
    mapped_te = unstd(np.asarray(phi_inv(p, jnp.asarray(Xs_te, jnp.float32))))
    m = score(mapped_te, S["Za_te"], S["lab"])
    held = {k: float(v) for k, v in eval_fn(p, *hb_args).items()}
    held["best_step"] = None
    held["mutual_nn_frac"] = float(ok_rows[mnn_rows].mean())
    R = {"direction": "b->a", "pair_of": "a_to_b", "n": int(n), "d": int(d),
         "holdout": {"n_hold": int(len(hold)), "n_train": int(len(train)), "rows": "hold"},
         "selected": "psi", "selection": "the second network of the pair, trained jointly with a->b",
         "variants": {"psi": {"held": held, **m}},
         "cycle_max_abs_error": cyc_err,
         "mutual_nn_frac_all": float(ok_rows.mean()), "mutual_nn_frac": held["mutual_nn_frac"],
         "mutual_nn_rows": "hold"}
    print(f"  b->a is the pair's second network (max |psi(phi(x)) - x| = {cyc_err:.2e}): "
          f"R@1 {m['r_at_1']:.3f}, held-out mutual-NN {held['mutual_nn_frac']:.3f}")
    state = {"p": p, "Xs": Xs, "Ys": Ys, "Xs_te": Xs_te, "unstd": unstd, "T": T, "train": train,
             "n": n, "hold": hold, "mnn_rows": mnn_rows, "map": phi_inv, "res": residual_inv, "d": int(d),
             "Q0": A["Q0"], "b0": A["b0"]}
    return R, mapped_te, state


# --------------------------------------------------------------------------- #
# joint rounds: anchors from the combined ranking, the pair refitted
# --------------------------------------------------------------------------- #
def _combined_train(A, B):
    """Distances from each direction on the two aligned clouds and the combined matrix, weighted
    by the squared mutual-NN fractions as at test. Both matrices are (n_a, n_b)."""
    Pa = np.asarray(A["map"](A["p"], jnp.asarray(A["Xs"], jnp.float32)), np.float64)
    Pb = np.asarray(B["map"](B["p"], jnp.asarray(B["Xs"], jnp.float32)), np.float64)
    Dab = cdist(Pa, A["Ys"]); Dba = cdist(Pb, B["Ys"]).T
    fa = float((Dab.argmin(0)[Dab.argmin(1)] == np.arange(Dab.shape[0])).mean())
    fb = float((Dba.argmin(1)[Dba.argmin(0)] == np.arange(Dba.shape[1])).mean())
    wa, wb = fa ** 2 + 1e-6, fb ** 2 + 1e-6; wa, wb = wa / (wa + wb), wb / (wa + wb)
    return wa * Dab / np.median(Dab) + wb * Dba / np.median(Dba), (fa, fb), (wa, wb)


def _mnn_of(Dc, rows=None):
    """Mutual-NN fraction of the combined ranking, over `rows` (the held-out rows) or all."""
    a2b, b2a = Dc.argmin(1), Dc.argmin(0)
    ok = b2a[a2b] == np.arange(Dc.shape[0])
    return float(ok.mean() if rows is None else ok[np.asarray(rows, int)].mean())


def _refit(D, Tcur, fit_rows, steps, batch, rng, tau):
    """Continue training the pair on targets Tcur, half of every a-batch from fit_rows."""
    p, st = D["p"], D["st"]
    fit_rows = np.asarray(fit_rows, int)
    fit_rows = fit_rows[np.isin(fit_rows, D["train"])]      # held-out rows are never fit targets
    for _ in range(steps):
        ia = rng.choice(fit_rows, size=min(batch // 2, len(fit_rows)), replace=False) if len(fit_rows) else np.empty(0, int)
        ib = rng.choice(D["train"], size=min(batch - len(ia), len(D["train"])), replace=False)
        idx = np.concatenate([ia, ib]).astype(int)
        p, st, _ = D["step"](p, st, jnp.asarray(D["Xs"][idx], jnp.float32), jnp.asarray(D["ca_sub"](idx), jnp.float32),
                             jnp.asarray(D["Ga_s"][idx], jnp.float32), jnp.asarray(Tcur[idx], jnp.float32), jnp.float32(tau))
    D["p"], D["st"] = p, st


def _pair_scores(A, B, S):
    """Test scores of the pair as it stands: each direction and the combined ranking."""
    _, (fa, fb), (wa, wb) = _combined_train(A, B)
    mab = A["unstd"](np.asarray(A["map"](A["p"], jnp.asarray(A["Xs_te"], jnp.float32))))
    mba = B["unstd"](np.asarray(B["map"](B["p"], jnp.asarray(B["Xs_te"], jnp.float32))))
    ra = score(mab, S["Zb_te"], S["lab"]); rb = score(mba, S["Za_te"], S["lab"])
    Dab = cdist(mab, S["Zb_te"]); Dba = cdist(mba, S["Za_te"]).T
    Dt = wa * Dab / np.median(Dab) + wb * Dba / np.median(Dba)
    r = retrieval_from_D(Dt, ks=(1, 5)); c = csls_retrieval(Dt, ks=(1, 5))
    return {"mutual_nn_frac": [fa, fb], "weights": [wa, wb],
            "a_to_b_r_at_1": ra["r_at_1"], "b_to_a_r_at_1": rb["r_at_1"],
            "combined": {"r_at_1": r["at1"], "r_at_5": r["at5"], "csls_at1": c["at1"],
                         "csls_at5": c["at5"],
                         "class_top1": float((S["lab"][Dt.argmin(1)] == S["lab"]).mean())}}


def joint_rounds(args, A, B, S, R, rng):
    """Refit the pair on the mutual nearest neighbours of the combined ranking, and keep the
    round with the largest held-out combined mutual-NN fraction (round 0 = the pair as it
    entered the rounds)."""
    n = A["n"]
    R["joint"] = []; prev = 0
    snap = lambda: (jax.tree_util.tree_map(lambda x: x, A["p"]),
                    jax.tree_util.tree_map(lambda x: x, B["p"]))
    best_sel, best_p, best_rnd = -1.0, snap(), 0
    sel_rows = A["mnn_rows"]
    R["joint_round0"] = _pair_scores(A, B, S)
    print(f"\n  joint rounds: anchors from the combined ranking on the aligned clouds, the pair refitted")
    print(f"  {'round':>5} {'anchors':>8} {'of':>5} {'w a/b':>11} | {'a->b R@1':>8} {'b->a R@1':>8} | {'comb R@1':>8} {'R@5':>6} {'CSLS':>6} {'time':>5}")
    for rnd in range(1, args.joint_rounds + 1):
        t0 = time.time()
        Dc, (fa, fb), (wa, wb) = _combined_train(A, B)
        a2b = Dc.argmin(1); b2a = Dc.argmin(0)
        anchors = np.where(b2a[a2b] == np.arange(n))[0]
        if prev and len(anchors) < prev * (1.0 + ANCHOR_STOP):
            print(f"  {rnd:>5} {len(anchors):>8} {n:>5} | anchors grew by less than {100 * ANCHOR_STOP:.0f}%: stop")
            R["joint_stopped_at"] = rnd; break
        prev = len(anchors)
        sel = _mnn_of(Dc, sel_rows)                  # the state entering this round
        if sel > best_sel:
            best_sel, best_p, best_rnd = sel, snap(), rnd - 1
        TA = A["T"].copy(); TA[anchors] = A["Ys"][a2b[anchors]]
        TB = B["T"].copy(); TB[a2b[anchors]] = B["Ys"][anchors]
        A["set_b_targets"](TB, a2b[anchors])
        _refit(A, TA, anchors, ALT_STEPS, args.batch, rng, TAU1)
        B["p"] = A["p"]                              # one parameter set holds both networks
        mab = A["unstd"](np.asarray(A["map"](A["p"], jnp.asarray(A["Xs_te"], jnp.float32))))
        mba = B["unstd"](np.asarray(B["map"](B["p"], jnp.asarray(B["Xs_te"], jnp.float32))))
        ra = score(mab, S["Zb_te"], S["lab"]); rb = score(mba, S["Za_te"], S["lab"])
        Dab = cdist(mab, S["Zb_te"]); Dba = cdist(mba, S["Za_te"]).T
        Dt = wa * Dab / np.median(Dab) + wb * Dba / np.median(Dba)
        r = retrieval_from_D(Dt, ks=(1, 5)); c = csls_retrieval(Dt, ks=(1, 5))
        R["joint"].append({"round": rnd, "anchors": int(len(anchors)), "mutual_nn_frac": [fa, fb],
                           "selection_mnn_entering": sel,
                           "weights": [wa, wb], "a_to_b_r_at_1": ra["r_at_1"], "b_to_a_r_at_1": rb["r_at_1"],
                           "combined": {"r_at_1": r["at1"], "r_at_5": r["at5"], "csls_at1": c["at1"],
                                        "csls_at5": c["at5"],
                                        "class_top1": float((S["lab"][Dt.argmin(1)] == S["lab"]).mean())}})
        print(f"  {rnd:>5} {len(anchors):>8} {n:>5} {wa:>5.2f}/{wb:<5.2f} | {ra['r_at_1']:>8.3f} {rb['r_at_1']:>8.3f} | "
              f"{r['at1']:>8.3f} {r['at5']:>6.3f} {c['at1']:>6.3f} {time.time() - t0:>4.0f}s")
    Dc, _, _ = _combined_train(A, B)
    sel_final = _mnn_of(Dc, sel_rows)
    if sel_final > best_sel:
        best_sel, best_rnd = sel_final, len(R["joint"])
    else:
        A["p"], B["p"] = best_p
    R["joint_selected_round"] = int(best_rnd)
    R["joint_selected_mnn"] = float(best_sel)
    R["joint_selection_rows"] = "hold"
    print(f"  kept round {best_rnd} of {len(R['joint'])} (largest combined held-out mutual-NN {best_sel:.3f}; "
          f"0 = no round improved on the pair that entered them)")
    return (A["unstd"](np.asarray(A["map"](A["p"], jnp.asarray(A["Xs_te"], jnp.float32)))),
            B["unstd"](np.asarray(B["map"](B["p"], jnp.asarray(B["Xs_te"], jnp.float32)))))


# --------------------------------------------------------------------------- #
def load_side(carve, mode, pi_path, pullback):
    g = lambda f: np.load(os.path.join(carve, f))
    S = {"Za": g("Za.npy").astype(np.float64), "Zb": g("Zb.npy").astype(np.float64),
         "Za_te": g("Za_test.npy"), "Zb_te": g("Zb_test.npy"), "lab": g("labels_test.npy"),
         "Ca": g(f"C_{mode}_a.npy").astype(np.float64), "Cb": g(f"C_{mode}_b.npy").astype(np.float64),
         "pi_soft": np.load(pi_path).astype(np.float64)}
    n, d = S["Za"].shape
    if S["Ca"].shape != (n, n) or S["Cb"].shape != (len(S["Zb"]), len(S["Zb"])):
        raise SystemExit(f"C_{mode}_* must be n x n over the carve's points")
    if pullback:
        S["Ga"] = g("G_ensemble_eol_samedec_a.npy").astype(np.float64)
        S["Gb"] = g("G_ensemble_eol_samedec_b.npy").astype(np.float64)
    else:
        S["Ga"] = np.broadcast_to(np.eye(d), (n, d, d)).copy()
        S["Gb"] = np.broadcast_to(np.eye(d), (len(S["Zb"]), d, d)).copy()
    return S


def load_supervision(carve, run_carve):
    """The true pairs for the oracle, from a carve that holds the same images on both sides:
    (Za, Zb[gt_ab], meta). Refuses a carve whose gt_ab is not a permutation, and records whether
    the two carves were cut from the same source (then they share the same test split)."""
    g = lambda f: np.load(os.path.join(carve, f))
    gt = g("gt_ab.npy")
    n_paired = int((gt >= 0).sum())
    if n_paired != len(gt):
        raise SystemExit(f"REFUSING: --oracle-carve {carve} has {n_paired}/{len(gt)} points with a "
                         f"partner; the oracle needs a carve with every image on both sides.")
    if len(set(gt.tolist())) != len(gt):
        raise SystemExit(f"REFUSING: --oracle-carve {carve} gt_ab is not injective")
    meta = {"carve": carve, "n_pairs": int(len(gt))}
    src = {}
    for tag, d in (("sup", carve), ("run", run_carve)):
        p = os.path.join(d, "subset.meta.json")
        src[tag] = json.load(open(p)).get("source") if os.path.exists(p) else None
    meta["same_phase_a_source"] = bool(src["sup"]) and src["sup"] == src["run"]
    meta["source"] = src["sup"]
    if not meta["same_phase_a_source"]:
        print(f"  WARNING: the oracle carve and the run's carve name different sources "
              f"({src['sup']} vs {src['run']}); their test splits may differ")
    return g("Za.npy").astype(np.float64), g("Zb.npy").astype(np.float64)[gt], meta


def swap(S):
    """The carve seen from side b: the reverse direction of the ceiling run."""
    return {"Za": S["Zb"], "Zb": S["Za"], "Za_te": S["Zb_te"], "Zb_te": S["Za_te"], "lab": S["lab"],
            "Ca": S["Cb"], "Cb": S["Ca"], "Ga": S["Gb"], "Gb": S["Ga"], "pi_soft": S["pi_soft"].T,
            "sup_a": S["sup_b"], "sup_b": S["sup_a"], "sup_meta": S["sup_meta"]}


def combined_ranking(mapped_ab, mapped_ba, S, wa, wb):
    """Each direction's test distances over their median, weighted by wa and wb."""
    Dab = cdist(mapped_ab, S["Zb_te"]); Dba = cdist(mapped_ba, S["Za_te"]).T
    Dc = wa * Dab / np.median(Dab) + wb * Dba / np.median(Dba)
    r = retrieval_from_D(Dc, ks=(1, 5)); c = csls_retrieval(Dc, ks=(1, 5))
    return {"r_at_1": r["at1"], "r_at_5": r["at5"], "csls_at1": c["at1"],
            "csls_at5": c["at5"], "csls_tie_frac": c["tie_frac"],
            "class_top1": float((S["lab"][Dc.argmin(1)] == S["lab"]).mean()),
            "pool_n": r["n"], "weights": {"a_to_b": wa, "b_to_a": wb}}


def linear_record(W, bvec, Q0):
    """The linear part of a network against the rigid start it began at."""
    W = np.asarray(W, np.float64); bvec = np.asarray(bvec, np.float64)
    sv = np.linalg.svd(W, compute_uv=False)
    return {"W": W.tolist(), "b": bvec.tolist(), "singular_values": sv.tolist(),
            "orth_gap": float(np.linalg.norm(W.T @ W - np.eye(W.shape[0]))),
            "drift_from_start": float(np.linalg.norm(W - Q0)),
            "det_sign": float(np.sign(np.linalg.det(W)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carve")
    ap.add_argument("--pi", required=True, help="the coupling over the carve, pi.npy")
    ap.add_argument("--mode", default="ensemble_eol_samedec", choices=["ensemble_eol_samedec", "euclidean"],
                    help="the cost matrices C_<mode>_{a,b} the losses read")
    ap.add_argument("--geometry", choices=["pullback", "euclidean"], default="pullback",
                    help="pullback: the metric tensors G; euclidean: G = I and the Euclidean costs")
    ap.add_argument("--fit-weight", type=float, default=0.3, help="weight of the coupling-fit term")
    ap.add_argument("--cycle-weight", type=float, default=0.1, help="weight of the cycle term")
    ap.add_argument("--joint-rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--holdout", type=int, default=500, help="held-out rows (at most a sixth of the carve)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--oracle", action="store_true", help="also fit the map class on the true pairs of --oracle-carve")
    ap.add_argument("--ceiling", action="store_true",
                    help="train no unsupervised map: the rigid start and the oracle in both directions")
    ap.add_argument("--oracle-carve", default=None, help="a carve with every image on both sides (gt_ab a permutation)")
    ap.add_argument("--oracle-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    ap.add_argument("--no-save-mapped", action="store_true", help="do not write the mapped test clouds")
    args = ap.parse_args()
    CYCLE["w"] = float(args.cycle_weight)
    if (args.oracle or args.ceiling) != bool(args.oracle_carve):
        raise SystemExit("--oracle or --ceiling needs --oracle-carve, and the carve needs one of them")
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"REFUSING: {args.out} exists. Pass --overwrite to replace it.")

    if args.geometry == "euclidean":
        args.mode = "euclidean"
    S = load_side(args.carve, args.mode, args.pi, args.geometry == "pullback")
    S["sup_a"] = S["sup_b"] = S["sup_meta"] = None
    if args.oracle_carve:
        S["sup_a"], S["sup_b"], S["sup_meta"] = load_supervision(args.oracle_carve, args.carve)
    rng = np.random.default_rng(args.seed); key = jax.random.PRNGKey(args.seed)
    print(f"NGMA map on {args.carve}\n  cost {args.mode}, coupling {args.pi}")
    print(f"  jax {jax.__version__} on {jax.default_backend()}")
    # the settings every record carries, the fixed ones included
    R = {"carve": args.carve, "pi": args.pi, "mode": args.mode, "geometry": args.geometry,
         "args": {**vars(args), "select": "mnn", "res_weight": 0.0, "alternate": 0, "mnn_gate": 1.0,
                  "both_directions": True,
                  **({"variants": "rigid", "inverse": "fit", "joint_rounds": 0} if args.ceiling else
                     {"variants": "rigid,iso+fit", "inverse": "cycle", "cycle_sides": "both", "fit_b_scale": 1.0})}}

    if args.ceiling:
        # the ceiling: the rigid start and the oracle, in both directions, nothing trained
        R["a_to_b"], mapped_ab, A = run_direction(args, S, "a->b", rng, key, train_pair=False)
        R["b_to_a"], mapped_ba, B = run_direction(args, swap(S), "b->a", rng, jax.random.PRNGKey(args.seed + 1),
                                                  train_pair=False)
        wa, wb = R["a_to_b"]["mutual_nn_frac"] ** 2 + 1e-6, R["b_to_a"]["mutual_nn_frac"] ** 2 + 1e-6
        wa, wb = wa / (wa + wb), wb / (wa + wb)
        R["combined"] = combined_ranking(mapped_ab, mapped_ba, S, wa, wb)
        R["kept"] = {d: {"variant": "rigid", "best_step": 0, "round": None, "mutual_nn_frac": R[d]["mutual_nn_frac"],
                         "linear": linear_record(St["p"]["W"], St["p"]["b"], St["Q0"])}
                     for d, St in (("a_to_b", A), ("b_to_a", B))}
        R["kept"]["joint_rounds_run"] = None
        R["kept"]["reported"] = "oracle"
        for d in ("a_to_b", "b_to_a"):
            print(f"  ceiling [{d}]: map class {R[d]['oracle']['r_at_1']:.3f}, "
                  f"Procrustes {R[d]['oracle']['rigid_closed']['r_at_1']:.3f}")
    else:
        R["a_to_b"], mapped_ab, A = run_direction(args, S, "a->b", rng, key, train_pair=True)
        R["b_to_a"], mapped_ba, B = reverse_direction(args, A, S, rng)
        if args.joint_rounds > 0:
            mapped_ab, mapped_ba = joint_rounds(args, A, B, S, R, rng)
        # the combined ranking: each direction's distances over their median, weighted by the
        # squared held-out mutual-NN fraction of that direction
        wa, wb = R["a_to_b"]["mutual_nn_frac"] ** 2 + 1e-6, R["b_to_a"]["mutual_nn_frac"] ** 2 + 1e-6
        wa, wb = wa / (wa + wb), wb / (wa + wb)
        R["combined"] = combined_ranking(mapped_ab, mapped_ba, S, wa, wb)
        print(f"\n  both directions combined (weights {wa:.2f} / {wb:.2f} from the mutual-NN fractions "
              f"{R['a_to_b']['mutual_nn_frac']:.2f} / {R['b_to_a']['mutual_nn_frac']:.2f}): "
              f"R@1 {R['combined']['r_at_1']:.3f}  R@5 {R['combined']['r_at_5']:.3f}  "
              f"CSLS@1 {R['combined']['csls_at1']:.3f}")
        # which map produced the reported number: the kept candidate and checkpoint per
        # direction, the kept joint round, and the linear part of each network against its start
        R["kept"] = {d: {"variant": R[d]["selected"],
                         "best_step": R[d]["variants"][R[d]["selected"]]["held"].get("best_step"),
                         "round": None, "mutual_nn_frac": R[d]["mutual_nn_frac"]}
                     for d in ("a_to_b", "b_to_a")}
        R["kept"]["joint_rounds_run"] = len(R.get("joint", [])) or None
        R["kept"]["joint_selected_round"] = R.get("joint_selected_round")
        R["kept"]["reported"] = "combined"
        R["kept"]["a_to_b"]["linear"] = linear_record(A["p"]["W"], A["p"]["b"], A["Q0"])
        R["kept"]["b_to_a"]["linear"] = linear_record(A["p"]["rev"]["W"], A["p"]["rev"]["b"], np.linalg.inv(A["Q0"]))
        for d in ("a_to_b", "b_to_a"):
            k = R["kept"][d]
            print(f"  kept [{d}]: {k['variant']} @ step {k['best_step']}, mutual-NN {k['mutual_nn_frac']:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"  wrote {args.out}")
    if not args.no_save_mapped and not args.ceiling:
        npz = args.out[:-5] + "_mapped.npz"
        np.savez_compressed(npz, mapped_ab=np.asarray(mapped_ab, np.float32),
                            mapped_ba=np.asarray(mapped_ba, np.float32),
                            Zb_te=np.asarray(S["Zb_te"], np.float32), Za_te=np.asarray(S["Za_te"], np.float32),
                            labels_te=np.asarray(S["lab"]),
                            weights=np.array([wa, wb], np.float64))
        print(f"  wrote {npz}")


if __name__ == "__main__":
    main()
