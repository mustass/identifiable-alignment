#!/usr/bin/env python
"""GENOT-Q (Klein et al. 2024), the quadratic variant of GENOT, on one carve.

An entropic GW coupling is solved on the two intra-domain cost matrices (NGMA's log-domain
solver, `ngma.alignment.gw.gw_entropic_jax`), pairs are sampled from it, and one conditional
flow-matching network (rectified flow, flax + optax) is trained to transport Gaussian noise
to the target conditioned on the source. The batch is the whole carve, so one coupling is
solved once and reused by every training step. The mapped test points are the average over
`--n-transport-samples` noise draws through the flow.

Costs: `--cost euclidean` (latent distances) or `--cost precomputed --cx --cy` (the carve's
pullback distance matrices). Both are raised to `--cost-power` and scaled to unit mean.

Reads <latent-dir>/{Za,Zb,Za_test,Zb_test,labels_a,labels_b,labels_test}.npy and appends one
row to `--out` (top-1 label accuracy, R@1, FOSCTTM, the CSLS block, and the run's settings).

  python baselines/genot_q.py --latent-dir checkpoints/<root>/seed1/disjoint \\
      --dataset fmnist --regime disjoint --seed 1 --batch-size 3000 --cost euclidean \\
      --out results/genot_q_<root>_s1_euclidean.csv
  python baselines/genot_q.py ... --cost precomputed \\
      --cx checkpoints/<root>/seed1/disjoint/C_ensemble_eol_samedec_a.npy \\
      --cy checkpoints/<root>/seed1/disjoint/C_ensemble_eol_samedec_b.npy \\
      --out results/genot_q_<root>_s1_samedec.csv
"""
import argparse
import csv
import functools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_metrics import retrieval_and_csls  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from ngma.alignment.gw import gw_entropic_jax


def _solve_gw(Ca, Cb, eps, max_iter):
    """Entropic GW coupling by the log-domain solver, in float32 (the flow stays float32)."""
    T, _ = gw_entropic_jax(Ca, Cb, eps, max_iter=max_iter, dtype="float32")
    return np.asarray(T, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Retrieval metrics (pure numpy)
# --------------------------------------------------------------------------- #
def _cdist(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return np.sqrt(np.maximum(
        (A * A).sum(1, keepdims=True) + (B * B).sum(1)[None, :] - 2.0 * A @ B.T, 0.0))


def _top1(Zq, Zref, lq, lref):
    """1-NN label accuracy for a class vector; for an attribute matrix (CelebA) the mean
    per-attribute accuracy of a logistic probe fitted in the reference space."""
    lref = np.asarray(lref); lq = np.asarray(lq)
    if lref.ndim > 1:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.multioutput import MultiOutputClassifier
        sc = StandardScaler()
        clf = MultiOutputClassifier(
            LogisticRegression(max_iter=500, random_state=0), n_jobs=-1)
        clf.fit(sc.fit_transform(Zref), lref)
        return float((np.asarray(clf.predict(sc.transform(Zq))) == lq).mean())
    return float(np.mean(lref[_cdist(Zq, Zref).argmin(1)] == lq))


def _r_at_1(Zmapped, Zref):
    nn_idx = _cdist(Zmapped, Zref).argmin(1)
    return float(np.mean(nn_idx == np.arange(len(Zmapped))))


# --------------------------------------------------------------------------- #
# Conditional flow-matching vector field  v_theta(y_t, t | x)
# --------------------------------------------------------------------------- #
class VectorField(nn.Module):
    hidden: int
    depth: int
    out_dim: int

    @nn.compact
    def __call__(self, y, t, x):
        # sinusoidal time features, concatenated with the state y and the condition x
        freqs = 2.0 ** jnp.arange(6, dtype=jnp.float32) * jnp.pi
        te = jnp.concatenate([jnp.sin(t * freqs), jnp.cos(t * freqs)], -1)
        h = jnp.concatenate([y, te, x], -1)
        for _ in range(self.depth):
            h = nn.gelu(nn.Dense(self.hidden)(h))
        return nn.Dense(self.out_dim)(h)


def make_train_state(key, p_dim, q_dim, hidden, depth, lr):
    net = VectorField(hidden=hidden, depth=depth, out_dim=q_dim)
    params = net.init(key, jnp.zeros((1, q_dim)), jnp.zeros((1, 1)), jnp.zeros((1, p_dim)))
    opt = optax.adamw(lr, weight_decay=1e-10)
    return net, params, opt, opt.init(params)


@functools.partial(jax.jit, static_argnums=(0, 4))
def _fm_step(net, params, opt_state, batch, opt):
    X, Y, key = batch["X"], batch["Y"], batch["key"]
    kz, kt = jax.random.split(key)
    Z = jax.random.normal(kz, Y.shape)                      # rho = N(0, I_q)
    t = jax.random.uniform(kt, (Y.shape[0], 1))
    Yt = (1.0 - t) * Z + t * Y                              # interpolant
    target = Y - Z                                          # rectified-flow velocity

    def loss_fn(p):
        pred = net.apply(p, Yt, t, X)
        return jnp.mean(jnp.sum((pred - target) ** 2, axis=-1))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


@functools.partial(jax.jit, static_argnums=(0, 4))
def _transport(net, params, X, Z0, n_steps):
    # integrate dy/dt = v(y, t | x) from y(0) = Z0 ~ N(0, I) to y(1), fixed Heun steps
    dt = 1.0 / n_steps

    def body(y, i):
        t = (i * dt) * jnp.ones((X.shape[0], 1))
        k1 = net.apply(params, y, t, X)
        k2 = net.apply(params, y + dt * k1, t + dt, X)
        return y + 0.5 * dt * (k1 + k2), None

    yT, _ = jax.lax.scan(body, Z0, jnp.arange(n_steps))
    return yT


# --------------------------------------------------------------------------- #
def load_latents(d):
    g = lambda f: np.load(os.path.join(d, f))
    return (g("Za.npy").astype(np.float32), g("Zb.npy").astype(np.float32),
            g("Za_test.npy").astype(np.float32), g("Zb_test.npy").astype(np.float32),
            g("labels_a.npy"), g("labels_b.npy"), g("labels_test.npy"))


def euclidean_cost(Z):
    return _cdist(Z, Z).astype(np.float32)


def sample_coupled_pairs(P, ix, iy, k, rng):
    # P: (B, B) GW coupling; returns source/target global indices of the sampled pairs
    row = np.maximum(np.asarray(P, np.float64), 0.0)
    rs = row.sum(1, keepdims=True)
    row = np.where(rs < 1e-12, 1.0, row)          # rows of (nearly) zero mass -> uniform
    row = row / row.sum(1, keepdims=True)
    m = len(iy)
    src, tgt = [], []
    for r in range(len(ix)):
        p = row[r] / row[r].sum()
        j = rng.choice(m, size=k, p=p)
        src.extend([ix[r]] * k)
        tgt.extend(iy[j])
    return np.asarray(src), np.asarray(tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-dir", required=True)
    ap.add_argument("--dataset", default="fmnist")
    ap.add_argument("--regime", default="disjoint")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cost", choices=["euclidean", "precomputed"], default="euclidean")
    ap.add_argument("--cx", default=None, help="precomputed: source (n_a, n_a) cost .npy")
    ap.add_argument("--cy", default=None, help="precomputed: target (n_b, n_b) cost .npy")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=2000,
                    help="pass the carve size, so the coupling is solved once on the whole carve")
    ap.add_argument("--gw-epsilon", type=float, default=1e-2)
    ap.add_argument("--gw-max-iter", type=int, default=100)
    ap.add_argument("--cost-power", type=float, default=2.0, help="the intra-domain costs are raised to this power")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--k-noise-per-x", type=int, default=1)
    ap.add_argument("--ode-steps", type=int, default=100)
    ap.add_argument("--n-transport-samples", type=int, default=8)
    ap.add_argument("--map-seed", type=int, default=0, help="seed of the transport noise draws")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Za, Zb, Za_te, Zb_te, la, lb, lte = load_latents(args.latent_dir)
    na, nb = len(Za), len(Zb)
    B = min(args.batch_size, na, nb)
    print(f"[genot-q] {args.dataset} {args.regime} s{args.seed} cost={args.cost} "
          f"| Za{Za.shape} Zb{Zb.shape} batch={B}", flush=True)

    if args.cost == "euclidean":
        C_X, C_Y = euclidean_cost(Za), euclidean_cost(Zb)
    else:
        if not (args.cx and args.cy):
            raise SystemExit("--cost precomputed needs --cx and --cy")
        C_X = np.load(args.cx).astype(np.float32)
        C_Y = np.load(args.cy).astype(np.float32)
        assert C_X.shape == (na, na) and C_Y.shape == (nb, nb), (C_X.shape, C_Y.shape)
    # intra-domain costs raised to cost_power and scaled to unit mean
    C_X = np.power(C_X, args.cost_power); C_Y = np.power(C_Y, args.cost_power)
    C_X = C_X / (C_X[C_X > 0].mean() + 1e-12)
    C_Y = C_Y / (C_Y[C_Y > 0].mean() + 1e-12)

    key = jax.random.PRNGKey(args.seed)
    net, params, opt, opt_state = make_train_state(
        key, Za.shape[1], Zb.shape[1], args.hidden_dim, args.depth, args.lr)

    rng = np.random.default_rng(args.seed)
    full_batch = B >= min(na, nb)
    if full_batch:
        # every batch is the same set of points, so the GW problem is identical each
        # iteration: solve it once and reuse the coupling
        print(f"[genot-q] full-batch: solving GW ONCE on the {na}x{nb} clouds ...", flush=True)
        _t = time.time()
        P_full = _solve_gw(C_X, C_Y, args.gw_epsilon, args.gw_max_iter)
        print(f"[genot-q] GW done in {time.time()-_t:.0f}s", flush=True)
        ix_full, iy_full = np.arange(na), np.arange(nb)
    t0 = time.time()
    running = 0.0
    for it in range(args.iterations):
        if full_batch:
            ix, iy, P = ix_full, iy_full, P_full
        else:
            ix = rng.choice(na, size=B, replace=False)
            iy = rng.choice(nb, size=B, replace=False)
            P = _solve_gw(C_X[np.ix_(ix, ix)], C_Y[np.ix_(iy, iy)],
                          args.gw_epsilon, args.gw_max_iter)
        src, tgt = sample_coupled_pairs(P, ix, iy, args.k_noise_per_x, rng)
        key, sk = jax.random.split(key)
        batch = {"X": jnp.asarray(Za[src]), "Y": jnp.asarray(Zb[tgt]), "key": sk}
        params, opt_state, loss = _fm_step(net, params, opt_state, batch, opt)
        running += float(loss)
        if (it + 1) % max(1, args.iterations // 10) == 0:
            print(f"[genot-q] it {it+1}/{args.iterations}  fm_loss={running/(args.iterations//10):.4f}",
                  flush=True)
            running = 0.0
    t_train = time.time() - t0

    # transport: average n_transport_samples noise draws through the flow
    samples = []
    for s in range(args.n_transport_samples):
        z0 = jax.random.normal(jax.random.PRNGKey(args.map_seed + s), (len(Za_te), Zb.shape[1]))
        samples.append(np.asarray(_transport(net, params, jnp.asarray(Za_te), z0, args.ode_steps)))
    mapped = np.mean(samples, axis=0).astype(np.float32)

    support_gap = float(np.linalg.norm(mapped.mean(0) - Zb_te.mean(0)) /
                        (np.linalg.norm(Zb_te.mean(0)) + 1e-8))
    scale_ratio = float(mapped.std(0).mean() / (Zb_te.std(0).mean() + 1e-8))
    top1, r1 = _top1(mapped, Zb_te, lte, lte), _r_at_1(mapped, Zb_te)
    bb, fos = (Zb_te.astype(np.float64) ** 2).sum(1), 0.0       # fraction closer than the true partner
    for s0 in range(0, len(mapped), 2048):
        Dm = bb[None] - 2.0 * mapped[s0:s0 + 2048].astype(np.float64) @ Zb_te.astype(np.float64).T
        own = Dm[np.arange(len(Dm)), np.arange(s0, s0 + len(Dm))][:, None]
        fos += float(((Dm < own).sum(1) / (len(Zb_te) - 1)).sum())
    fos /= len(mapped)
    print(f"[genot-q] top1={top1:.4f}  R@1={r1:.4f}  | train {t_train:.0f}s "
          f"support_gap={support_gap:.3f} scale_ratio={scale_ratio:.3f}", flush=True)

    row = dict(method="GENOT-Q", supervision="unsup", dataset=args.dataset,
               regime=args.regime, seed=args.seed, cost=args.cost,
               top1=round(top1, 4), r_at_1=round(r1, 4), foscttm=round(fos, 5),
               **{k: round(v, 4) for k, v in retrieval_and_csls(mapped, Zb_te).items()},
               iterations=args.iterations, batch_size=B, gw_epsilon=args.gw_epsilon,
               train_s=round(t_train, 1), support_gap=round(support_gap, 4),
               scale_ratio=round(scale_ratio, 4))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[genot-q] appended → {args.out}", flush=True)


if __name__ == "__main__":
    main()
