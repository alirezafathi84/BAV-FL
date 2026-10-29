"""Budget-aware federated configuration planner (the BAV-FL controller).

For a hard per-site (eps, delta) constraint, the planner chooses the number of rounds T and,
for every hospital k, its Poisson batch size B_k and local steps per round m_k by minimising
the standard convex SGD excess-risk bound with the EXACT accountant-derived noise multiplier:

    Phi_k(B_k, m_k, T) = D^2 / (2 lr S_k)  +  (lr / 2) * ( C^2 + d * sigma_k^2 C^2 / B_k^2 ),
    S_k = m_k * T,  sigma_k = sigma(q_k = B_k/n_k, S_k ; eps, delta)   (RDP accountant)

    min_T  sum_k p_k  min_{B_k, m_k}  Phi_k .

The first term is optimisation bias (falls with steps), the second is gradient variance from
clipping + DP noise (rises with steps because the accountant must raise sigma).  Every input is
public (n_k, d, C, lr, eps, delta, and the prior scale D = sqrt(d) for unit-scale coefficients),
so planning consumes NO privacy budget and needs no tuning runs on private data.
"""
import math
import numpy as np
from .privacy import FixedAllocator

B_GRID = (8, 16, 32, 64, 128)
T_GRID = (10, 15, 20, 30, 40)


def site_bound(n_k, B, m, T, *, eps, delta, d, lr, C, D):
    Be = min(B, n_k)
    q = Be / n_k
    sigma = FixedAllocator(q, m, T, eps, delta).sigma
    S = m * T
    return D ** 2 / (2 * lr * S) + lr / 2 * (C ** 2 + d * (sigma * C / Be) ** 2), sigma


def plan(n, *, eps, delta, d, lr=0.5, C=1.0, D=None, B_grid=B_GRID, T_grid=T_GRID, max_epochs_per_round=2):
    """n: {site: n_k}. Returns dict(T=..., sites={site: dict(B, m, sigma, bound)}, objective)."""
    D = math.sqrt(d) if D is None else D
    N = sum(n.values())
    best = None
    for T in T_grid:
        tot, choice = 0.0, {}
        for s, nk in n.items():
            cands = []
            for B in sorted({min(b, nk) for b in B_grid}):
                for m in range(1, max_epochs_per_round * math.ceil(nk / B) + 1):
                    phi, sig = site_bound(nk, B, m, T, eps=eps, delta=delta, d=d, lr=lr, C=C, D=D)
                    cands.append((phi, B, m, sig))
            phi, B, m, sig = min(cands)
            choice[s] = dict(B=B, m=m, sigma=sig, bound=phi)
            tot += nk / N * phi
        if best is None or tot < best["objective"]:
            best = dict(T=T, sites=choice, objective=tot)
    return best
