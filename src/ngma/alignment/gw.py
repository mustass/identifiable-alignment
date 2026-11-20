"""Entropic Gromov-Wasserstein solvers in JAX.

Both solvers run the Frank-Wolfe iteration of entropic GW with the square loss: at every
step the linearised GW cost at the current plan is handed to a log-domain Sinkhorn solve
with uniform marginals. `gw_entropic_jax` is plain entropic GW; `fgw_entropic_jax` adds a
fixed linear term to the cost of every step (fused GW).
"""

import time
import numpy as np


def gw_entropic_jax(
    Ca: np.ndarray,
    Cb: np.ndarray,
    epsilon: float,
    init_coupling: np.ndarray = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    sink_max_iter: int = 1000,
    sink_tol: float = 1e-6,
    dtype: str = "float32",
) -> tuple[np.ndarray, dict]:
    """Entropic GW by Frank-Wolfe with a log-domain inner Sinkhorn, on the default device.

    Ca, Cb: (n, n) intra-space distance matrices. init_coupling warm-starts the plan (the
    uniform coupling by default). The outer loop stops when the plan's relative change is
    below tol or after max_iter steps; the inner Sinkhorn stops when the change of log u is
    below sink_tol or after sink_max_iter sweeps. dtype 'float64' needs jax_enable_x64.

    Returns (plan as float32, log with iters_done, converged, final_rel_change, seconds).
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as _lse
    from functools import partial as _partial

    jdt = jnp.float64 if str(dtype) == "float64" else jnp.float32

    @_partial(jax.jit, static_argnums=(4, 5, 6, 7))
    def _core(Ca, Cb, eps, T0, mi, ftol, smi, stol):
        n = Ca.shape[0]
        dt = Ca.dtype
        floor = jnp.finfo(dt).tiny
        log_a = jnp.full((n,), -jnp.log(n), dtype=dt)
        log_b = log_a
        Ca2, Cb2 = Ca ** 2, Cb ** 2
        inf = jnp.asarray(jnp.inf, dt)

        def sinkhorn(M, T_ws):
            log_K = -M / eps
            log_u = jnp.log(jnp.clip(T_ws.sum(1), floor, None)) - log_a
            log_v = jnp.zeros((n,), dt)

            def c(s):
                i, lu, lv, e = s
                return jnp.logical_and(i < smi, e > stol)

            def b(s):
                i, lu, lv, _ = s
                lv2 = log_b - _lse(log_K + lu[:, None], axis=0)
                lu2 = log_a - _lse(log_K + lv2[None, :], axis=1)
                return (i + 1, lu2, lv2, jnp.max(jnp.abs(lu2 - lu)))

            _, lu, lv, _ = jax.lax.while_loop(c, b, (jnp.int32(0), log_u, log_v, inf))
            return jnp.exp(lu[:, None] + log_K + lv[None, :])

        def c(s):
            i, T, e = s
            return jnp.logical_and(i < mi, jnp.logical_or(i < 1, e > ftol))

        def b(s):
            i, T, _ = s
            # linearised GW cost (square loss): Ca^2 a + Cb^2 b - 2 Ca T Cb^T
            M = (Ca2 @ T.sum(1))[:, None] + (Cb2 @ T.sum(0))[None, :] - 2.0 * (Ca @ T @ Cb.T)
            Tn = sinkhorn(M, T)
            e = jnp.linalg.norm(Tn - T) / (jnp.linalg.norm(T) + 1e-12)
            return (i + 1, Tn, e)

        it, T, e = jax.lax.while_loop(c, b, (jnp.int32(0), T0, inf))
        return T, it, e

    n = Ca.shape[0]
    Ca_j = jnp.asarray(Ca, jdt)
    Cb_j = jnp.asarray(Cb, jdt)
    T0 = (jnp.full((n, n), 1.0 / (n * n), jdt) if init_coupling is None
          else jnp.asarray(init_coupling, jdt))
    t0 = time.time()
    T, it, e = _core(Ca_j, Cb_j, jnp.asarray(epsilon, jdt), T0,
                     int(max_iter), float(tol), int(sink_max_iter), float(sink_tol))
    T.block_until_ready()
    final_rel = float(e)
    return np.asarray(T).astype(np.float32), {
        "solver": "jax", "backend": jax.default_backend(),
        "iters_done": int(it), "seconds": float(time.time() - t0),
        "converged": bool(int(it) >= 1 and final_rel <= float(tol)),
        "final_rel_change": final_rel, "tol": float(tol)}


def fgw_entropic_jax(
    Ca: np.ndarray,
    Cb: np.ndarray,
    M_lin: np.ndarray,
    alpha: float,
    epsilon: float,
    init_coupling: np.ndarray = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    sink_max_iter: int = 1000,
    sink_tol: float = 1e-6,
    dtype: str = "float32",
    warm_duals: bool = False,
    init_duals=None,
) -> tuple[np.ndarray, dict]:
    """`gw_entropic_jax` with a linear term: every Frank-Wolfe step solves Sinkhorn on
    (1 - alpha) * (GW's linearised cost) + alpha * M_lin, so it descends
    (1 - alpha) GW(T) + 2 alpha <M_lin, T>. alpha = 0 is `gw_entropic_jax` step for step.

    warm_duals: start the inner Sinkhorn of every step from the dual potentials the previous
    step ended with instead of from the current plan's row sums; init_duals = (log_u, log_v)
    seeds the first step. The fixed point of the inner solve does not depend on its start.
    The log reports the final potentials under "duals", the total inner sweeps under
    "sink_sweeps" and the number of steps whose inner solve hit sink_max_iter under
    "sink_capped_steps". dtype 'float64' switches JAX's x64 mode on.
    """
    import jax
    if str(dtype) == "float64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as _lse
    from functools import partial as _partial

    jdt = jnp.float64 if str(dtype) == "float64" else jnp.float32

    @_partial(jax.jit, static_argnums=(8, 9, 10, 11, 12))
    def _core(Ca, Cb, Ml, al, eps, T0, lu0, lv0, mi, ftol, smi, stol, warm):
        n = Ca.shape[0]
        dt = Ca.dtype
        floor = jnp.finfo(dt).tiny
        log_a = jnp.full((n,), -jnp.log(n), dtype=dt)
        log_b = log_a
        Ca2, Cb2 = Ca ** 2, Cb ** 2
        inf = jnp.asarray(jnp.inf, dt)

        def sinkhorn(M, T_ws, lu_ws, lv_ws):
            log_K = -M / eps
            if warm:
                log_u, log_v = lu_ws, lv_ws
            else:
                log_u = jnp.log(jnp.clip(T_ws.sum(1), floor, None)) - log_a
                log_v = jnp.zeros((n,), dt)

            def c(s):
                i, lu, lv, e = s
                return jnp.logical_and(i < smi, e > stol)

            def b(s):
                i, lu, lv, _ = s
                lv2 = log_b - _lse(log_K + lu[:, None], axis=0)
                lu2 = log_a - _lse(log_K + lv2[None, :], axis=1)
                return (i + 1, lu2, lv2, jnp.max(jnp.abs(lu2 - lu)))

            k, lu, lv, _ = jax.lax.while_loop(c, b, (jnp.int32(0), log_u, log_v, inf))
            return jnp.exp(lu[:, None] + log_K + lv[None, :]), lu, lv, k

        def c(s):
            i, T, e, lu, lv, sw, cap = s
            return jnp.logical_and(i < mi, jnp.logical_or(i < 1, e > ftol))

        def b(s):
            i, T, _, lu, lv, sw, cap = s
            Mg = (Ca2 @ T.sum(1))[:, None] + (Cb2 @ T.sum(0))[None, :] - 2.0 * (Ca @ T @ Cb.T)
            Tn, lu2, lv2, k = sinkhorn((1.0 - al) * Mg + al * Ml, T, lu, lv)
            e = jnp.linalg.norm(Tn - T) / (jnp.linalg.norm(T) + 1e-12)
            return (i + 1, Tn, e, lu2, lv2, sw + k, cap + (k >= smi).astype(jnp.int32))

        it, T, e, lu, lv, sw, cap = jax.lax.while_loop(
            c, b, (jnp.int32(0), T0, inf, lu0, lv0, jnp.int32(0), jnp.int32(0)))
        return T, it, e, lu, lv, sw, cap

    n = Ca.shape[0]
    T0 = (jnp.full((n, n), 1.0 / (n * n), jdt) if init_coupling is None
          else jnp.asarray(init_coupling, jdt))
    if init_duals is None:
        lu0, lv0 = jnp.zeros((n,), jdt), jnp.zeros((n,), jdt)
    else:
        lu0, lv0 = jnp.asarray(init_duals[0], jdt), jnp.asarray(init_duals[1], jdt)
    t0 = time.time()
    T, it, e, lu, lv, sw, cap = _core(jnp.asarray(Ca, jdt), jnp.asarray(Cb, jdt), jnp.asarray(M_lin, jdt),
                                 jnp.asarray(alpha, jdt), jnp.asarray(epsilon, jdt), T0, lu0, lv0,
                                 int(max_iter), float(tol), int(sink_max_iter), float(sink_tol), bool(warm_duals))
    T.block_until_ready()
    final_rel = float(e)
    return np.asarray(T).astype(np.float32), {
        "solver": "jax-fgw", "backend": jax.default_backend(), "alpha": float(alpha),
        "iters_done": int(it), "seconds": float(time.time() - t0),
        "converged": bool(int(it) >= 1 and final_rel <= float(tol)),
        "final_rel_change": final_rel, "tol": float(tol),
        "warm_duals": bool(warm_duals), "sink_sweeps": int(sw), "sink_capped_steps": int(cap),
        "duals": (np.asarray(lu, np.float64), np.asarray(lv, np.float64))}
