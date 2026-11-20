"""The epsilon schedule of the annealed entropic GW solvers: a geometric sequence from
eps_start down to eps_final, each stage warm-started from the previous one.

    build_eps_schedule(1.0, 1e-3, n_steps=4)  -> [1, 0.1, 0.01, 0.001]
"""
from __future__ import annotations

import math

import numpy as np


def build_eps_schedule(eps_start: float, eps_final: float,
                       n_steps: int | None = None,
                       decay: float | None = None) -> list:
    """Geometric schedule from eps_start down to eps_final, both included.

    n_steps: number of stages including both endpoints. decay: per-stage division factor
    (> 1), from which the number of stages follows; it takes precedence over n_steps. A
    range with nothing to anneal over, or neither parameter asking for more than one stage,
    gives [eps_final].
    """
    eps_start, eps_final = float(eps_start), float(eps_final)
    if eps_final <= 0:
        raise ValueError(f"eps_final must be > 0, got {eps_final}")
    if eps_start <= eps_final:
        return [eps_final]

    if decay is not None:
        if decay <= 1.0:
            raise ValueError(f"eps_decay must be > 1, got {decay}")
        n = 1 + math.ceil(math.log(eps_start / eps_final) / math.log(decay) - 1e-12)
    elif n_steps is not None and n_steps > 1:
        n = int(n_steps)
    else:
        return [eps_final]

    sched = [float(e) for e in np.geomspace(eps_start, eps_final, n)]
    sched[0], sched[-1] = eps_start, eps_final   # exact endpoints
    return sched


def schedule_from_cfg(gw_cfg) -> list:
    """The schedule of a gw settings block: keys epsilon (final), eps_start, n_anneal_steps,
    eps_decay."""
    eps_final = float(gw_cfg.get("epsilon", 1e-3))
    eps_start = float(gw_cfg.get("eps_start", eps_final))
    decay = gw_cfg.get("eps_decay", None)
    return build_eps_schedule(
        eps_start, eps_final,
        n_steps=int(gw_cfg.get("n_anneal_steps", 1)),
        decay=float(decay) if decay is not None else None,
    )
