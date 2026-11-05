"""The fused coupling: NGMA's annealed entropic GW with Memoli's third lower bound as a linear
term. Every Frank-Wolfe step runs on (1 - alpha) * (GW's linearised cost) + alpha * M, which
descends (1 - alpha) GW + 2 alpha <M, T>. alpha is a value in [0, 1] or one of two rules that
read only the two cost matrices:
  bal  alpha fixed so that the two terms are equal at the uniform coupling;
  ann  alpha annealed with epsilon, linearly from 1 at the first stage to 0 at the last, so
       the returned plan is a stationary point of GW's own objective; ann<a0> starts from
       a0 < 1 instead of 1.
"""
import numpy as np

from ..geometry.ldd import tlb_cost
from .gw import fgw_entropic_jax

FGW_RULES = ("bal", "ann")


def fgw_parse_alpha(token):
    """A fixed alpha as a float, or the rule tokens `bal` / `ann` / `ann<a0>` (a0 in (0, 1))."""
    if isinstance(token, str) and token in FGW_RULES:
        return token
    if isinstance(token, str) and token.startswith("ann"):
        a0 = float(token[3:])
        if not 0.0 < a0 < 1.0:
            raise ValueError(f"ann<a0> needs 0 < a0 < 1: {token}")
        return f"ann{a0:g}"
    x = float(token)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"fgw alpha must be in [0, 1] or one of {FGW_RULES}: {token}")
    return x


FGW_EPS_RULES = ("gw", "mix", "gmix")
FGW_PREFIX = {"gw": "FGW", "mix": "FGWM", "gmix": "FGWG"}
#: The LDD-LP's epsilon relative to the max of its cost; the second input of the mixed
#: epsilon schedules.
FGW_LDD_EPS_REL = 3e-4


def fgw_alpha_tag(alpha, eps_rule: str = "gw") -> str:
    """The coupling directory suffix: FGW_a05 (0.5), FGW_a075, FGW_a1, FGW_abal, FGW_ann on GW's
    epsilon schedule; FGWM_... on the mixed schedule; FGWG_... on the geometric mix;
    ann0.75 -> ..._ann075."""
    prefix = FGW_PREFIX[eps_rule]
    if alpha == "bal":
        return f"{prefix}_abal"
    if alpha == "ann":
        return f"{prefix}_ann"
    if isinstance(alpha, str) and alpha.startswith("ann"):
        return f"{prefix}_ann" + f"{float(alpha[3:]):g}".replace(".", "")
    return f"{prefix}_a" + f"{float(alpha):g}".replace(".", "")


def fgw_alpha_from_tag(tag: str):
    """The inverse of fgw_alpha_tag: FGW_a075 -> (0.75, 'gw'), FGWG_ann -> ('ann', 'gmix')."""
    prefix, _, rest = tag.partition("_")
    rule = {v: k for k, v in FGW_PREFIX.items()}.get(prefix)
    if rule is None or not rest.startswith("a"):
        raise ValueError(f"not a fused-coupling directory: {tag}")
    if rest == "abal":
        return "bal", rule
    if rest == "ann":
        return "ann", rule
    if rest.startswith("ann"):
        d = rest[3:]
        return f"ann{float('0.' + d[1:]):g}", rule
    digits = rest[1:]
    return (float(digits) if digits in ("0", "1") else float("0." + digits[1:])), rule


def fgw_eps_schedule(eps_schedule, alphas, M, ldd_eps_rel: float = FGW_LDD_EPS_REL, eps_rule: str = "gw"):
    """The epsilon of every stage.

    `gw`: GW's schedule as it is. With rho = ldd_eps_rel * max(M) / eps_last^GW (the LDD-LP's
    epsilon in the fused units, relative to GW's last stage), `mix` interpolates the two
    schedules linearly, eps_k = eps_k^GW [(1 - a_k) + a_k rho], and `gmix` geometrically,
    eps_k = eps_k^GW rho^a_k. The two agree at alpha 0 (GW's schedule) and alpha 1 (the
    LDD-LP's epsilon carried through GW's stages).
    """
    eps = [float(e) for e in eps_schedule]
    if eps_rule == "gw":
        return eps, {"eps_rule": "gw"}
    if eps_rule not in ("mix", "gmix"):
        raise ValueError(f"eps_rule must be one of {FGW_EPS_RULES}: {eps_rule}")
    rho = ldd_eps_rel * float(np.asarray(M).max()) / eps[-1]
    if eps_rule == "gmix":
        return [e * rho ** a for e, a in zip(eps, alphas)], {"eps_rule": "gmix", "rho": rho,
                                                           "ldd_eps_rel": ldd_eps_rel}
    return [e * ((1.0 - a) + a * rho) for e, a in zip(eps, alphas)], {"eps_rule": "mix", "rho": rho,
                                                                      "ldd_eps_rel": ldd_eps_rel}


def fgw_alpha_balance(Ca_n, Cb_n, M):
    """alpha at which the two terms of the fused objective are equal at the uniform coupling:
    alpha* = GW(T0) / (GW(T0) + 2 mean(M)), GW(T0) = mean(Ca^2) + mean(Cb^2) - 2 mean(Ca) mean(Cb)."""
    A, B = np.asarray(Ca_n, np.float64), np.asarray(Cb_n, np.float64)
    gw0 = float((A ** 2).mean() + (B ** 2).mean() - 2.0 * A.mean() * B.mean())
    lin0 = 2.0 * float(np.asarray(M).mean())
    return gw0 / (gw0 + lin0), gw0, lin0


def fgw_anneal_alphas(n_stages: int, a0: float = 1.0) -> list:
    """a0 at the first stage down to 0 at the last, linearly: [1, 2/3, 1/3, 0] for four stages at
    a0 = 1."""
    return [a0 * (1.0 - k / (n_stages - 1)) for k in range(n_stages)] if n_stages > 1 else [a0]


def fgw_alphas_per_stage(alpha, Ca_n, Cb_n, M, n_stages: int):
    """The alpha of every anneal stage for a fixed value or a rule, and what the rule computed."""
    if alpha == "bal":
        a, gw0, lin0 = fgw_alpha_balance(Ca_n, Cb_n, M)
        return [a] * n_stages, {"alpha_balance": a, "gw_at_uniform": gw0, "linear_at_uniform": lin0}
    if alpha == "ann":
        return fgw_anneal_alphas(n_stages), {}
    if isinstance(alpha, str) and alpha.startswith("ann"):
        a0 = float(alpha[3:])
        return fgw_anneal_alphas(n_stages, a0), {"alpha_start": a0}
    return [float(alpha)] * n_stages, {}


def fgw_coupling(Ca: np.ndarray, Cb: np.ndarray, alpha, eps_schedule, max_iter: int = 500,
                 tol: float = 1e-9, dtype: str = "float64", normalize: str = "shared",
                 eps_rule: str = "gw", ldd_eps_rel: float = FGW_LDD_EPS_REL,
                 sink_max_iter: int = 100_000, warm_duals: bool = False) -> tuple[np.ndarray, dict]:
    """The fused coupling of two distance matrices.

    The distances are normalised as GW's are (shared: both over 0.5 (mean_a + mean_b);
    perside: each over its mean) and cast to float32; M is `tlb_cost` on the normalised
    distances. The stages of eps_schedule run in turn, each warm-started from the previous
    plan, with at most max_iter Frank-Wolfe steps stopping at tol, the stage weights of
    `fgw_alphas_per_stage` and the epsilons of `fgw_eps_schedule`. warm_duals carries the
    inner Sinkhorn's potentials across Frank-Wolfe steps and, rescaled by eps_prev / eps,
    across stages. Returns (plan in the solver's dtype, record of the solve).
    """
    Ca, Cb = np.asarray(Ca), np.asarray(Cb)
    if normalize == "perside":
        sa, sb = float(Ca.mean()) or 1.0, float(Cb.mean()) or 1.0
        Ca_n, Cb_n, scale = (Ca / sa).astype(np.float32), (Cb / sb).astype(np.float32), [sa, sb]
    else:
        scale = 0.5 * (float(Ca.mean()) + float(Cb.mean()))
        Ca_n, Cb_n = (Ca / scale).astype(np.float32), (Cb / scale).astype(np.float32)
    M = tlb_cost(Ca_n, Cb_n)
    schedule = [float(e) for e in eps_schedule]
    per_stage, rule = fgw_alphas_per_stage(alpha, Ca_n, Cb_n, M, len(schedule))
    eps_used, eps_info = fgw_eps_schedule(schedule, per_stage, M, ldd_eps_rel, eps_rule)
    rule = {**rule, **eps_info, "eps_per_stage": eps_used}
    pi, stages, duals, eps_prev = None, [], None, None
    for eps, al in zip(eps_used, per_stage):
        init_duals = None
        if warm_duals and duals is not None:
            # log_u = f / eps: the potentials f, g carry over, so rescale by the eps ratio
            init_duals = (duals[0] * (eps_prev / eps), duals[1] * (eps_prev / eps))
        pi, log = fgw_entropic_jax(Ca_n, Cb_n, M, al, epsilon=eps, init_coupling=pi, max_iter=max_iter,
                                   tol=tol, sink_max_iter=sink_max_iter, sink_tol=tol, dtype=dtype,
                                   warm_duals=warm_duals, init_duals=init_duals)
        duals, eps_prev = log["duals"], eps
        stages.append({"eps": eps, "alpha": float(al), "iters": log["iters_done"], "converged": log["converged"],
                       "final_rel_change": log["final_rel_change"], "seconds": round(log["seconds"], 1),
                       "sink_sweeps": log["sink_sweeps"], "sink_capped_steps": log["sink_capped_steps"]})
    pi = np.asarray(pi)
    P = pi.astype(np.float64); n_a, n_b = P.shape
    A, B = Ca_n.astype(np.float64), Cb_n.astype(np.float64)
    gw_val = float((A ** 2).mean() + (B ** 2).mean() - 2.0 * np.einsum("ik,ik->", A, (P @ B) @ P.T))
    lin = float((P * M).sum())
    info = {"coupling": "FGW", "alpha": alpha, "alpha_per_stage": per_stage, **rule,
            "linear_term": "Memoli TLB: W_2^2 between distance distributions, on GW's normalised distances",
            "objective": "(1 - alpha) GW + 2 alpha <M, T>, by Frank-Wolfe on (1 - alpha) L(x)T + alpha M",
            "eps_schedule": schedule, "max_iter": max_iter, "tol": tol, "sink_max_iter": sink_max_iter,
            "warm_duals": bool(warm_duals), "dtype": dtype, "normalize": normalize,
            "scale": scale, "stages": stages, "seconds": round(sum(s["seconds"] for s in stages), 1),
            "gw_cost": gw_val, "linear_cost": lin, "fused_cost": (1.0 - per_stage[-1]) * gw_val + 2.0 * per_stage[-1] * lin,
            "tlb_max": float(M.max()), "tlb_median": float(np.median(M)),
            "row_marginal_rel_err": float(np.abs(P.sum(1) * n_a - 1).max()),
            "col_marginal_rel_err": float(np.abs(P.sum(0) * n_b - 1).max()),
            "finite": bool(np.isfinite(P).all())}
    return pi, info
