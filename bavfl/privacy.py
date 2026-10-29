"""Privacy accounting (RDP, Poisson-subsampled Gaussian) and noise allocators.

Allocators decide the per-round noise multiplier sigma_t for ONE client whose
local DP-SGD uses sampling rate q and `steps` noisy steps per round, under a hard
(eps_budget, delta) constraint over T rounds.

* FixedAllocator      -- standard DP-FedAvg: one sigma for all rounds.
* ExpDecayAllocator   -- sigma_t = sigma_0 * gamma^t (noise-decay baseline).
* BudgetOptimalAllocator (proposed, "BONA") -- receding-horizon solution of
      min_{sigma_1..T} sum_t sigma_t^2 / s_t^2   s.t.  eps_RDP(sigma_1..T) <= eps_budget
  Using RDP_t ~ c * q^2 / sigma_t^2 the KKT conditions give sigma_t ∝ sqrt(s_t),
  where s_t is the (public, post-processed) gradient-signal norm.  Each round the
  remaining budget is re-solved exactly with the accountant (bisection on the
  Lagrange scale), so the constraint holds with equality at T and never overshoots.
"""
import functools
import numpy as np

ORDERS = np.concatenate([np.arange(2, 65), [72, 80, 96, 128, 192, 256]]).astype(float)
SIGMA_MIN, SIGMA_MAX = 0.55, 30.0
_AMAX = int(ORDERS.max())
_I = np.arange(_AMAX + 1)
_VALID = _I[None, :] <= ORDERS[:, None]
from scipy.special import gammaln as _gl
_LOGBINOM = np.where(_VALID, _gl(ORDERS[:, None] + 1) - _gl(_I[None, :] + 1)
                     - _gl(np.maximum(ORDERS[:, None] - _I[None, :], 0) + 1), -np.inf)


def _rdp_poisson_gaussian(q, sigma):
    """Exact RDP of the Poisson-subsampled Gaussian at integer orders (Mironov et al. 2019,
    Sec. 3.3 binomial expansion), vectorised over all orders."""
    if q >= 1.0:
        return ORDERS / (2 * sigma ** 2)
    a = ORDERS[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = _LOGBINOM + (a - _I[None, :]) * np.log1p(-q) + _I[None, :] * np.log(q) \
            + (_I[None, :] ** 2 - _I[None, :]) / (2 * sigma ** 2)
    lt = np.where(_VALID, lt, -np.inf)
    m = lt.max(1)
    loga = m + np.log(np.exp(lt - m[:, None]).sum(1))
    return loga / (ORDERS - 1)


@functools.lru_cache(maxsize=500000)
def _rdp_step_cached(q, sigma_r):
    return _rdp_poisson_gaussian(q, sigma_r)


def rdp_step(q, sigma):
    return _rdp_step_cached(round(float(q), 6), round(float(sigma), 4))


def eps_from_rdp(rdp_vec, delta):
    """RDP -> (eps, delta) conversion of Balle et al. (2020), as used in TF-Privacy/Opacus."""
    a = ORDERS
    eps = rdp_vec - (np.log(delta) + np.log(a)) / (a - 1) + np.log((a - 1) / a)
    return float(np.max([np.min(eps), 0.0]))


def eps_of_schedule(q, steps, sigmas, delta, base_rdp=None):
    tot = np.zeros_like(ORDERS) if base_rdp is None else base_rdp.copy()
    for s in sigmas:
        tot = tot + steps * rdp_step(q, s)
    return eps_from_rdp(tot, delta)


def _bisect(fn_eps, target, lo, hi, iters=40, increasing_in_x=False):
    """find x with fn_eps(x) ~= target (eps decreasing in x unless stated)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        e = fn_eps(mid)
        too_much = e > target
        if increasing_in_x:
            hi, lo = (mid, lo) if too_much else (hi, mid)
        else:
            lo, hi = (mid, hi) if too_much else (lo, mid)
    return hi if not increasing_in_x else lo


class FixedAllocator:
    name = "fixed"

    def __init__(self, q, steps, T, eps_budget, delta):
        self.q, self.steps, self.T, self.delta = q, steps, T, delta
        self.sigma = _bisect(lambda s: eps_of_schedule(q, steps, [s] * T, delta),
                             eps_budget, SIGMA_MIN, SIGMA_MAX * 4)
        self.history = []

    def next_sigma(self, t, signal_hist=None):
        self.history.append(self.sigma)
        return self.sigma

    def spent(self):
        return eps_of_schedule(self.q, self.steps, self.history, self.delta)


class ExpDecayAllocator(FixedAllocator):
    """Noise decay baseline: sigma_T = ratio * sigma_0 (geometric), calibrated."""
    name = "decay"

    def __init__(self, q, steps, T, eps_budget, delta, ratio=0.5):
        self.q, self.steps, self.T, self.delta = q, steps, T, delta
        g = ratio ** (1.0 / max(T - 1, 1))
        prof = np.array([g ** t for t in range(T)])
        k = _bisect(lambda k: eps_of_schedule(q, steps, np.maximum(k * prof, SIGMA_MIN), delta),
                    eps_budget, SIGMA_MIN, SIGMA_MAX * 4)
        self.sched = np.maximum(k * prof, SIGMA_MIN)
        self.history = []

    def next_sigma(self, t, signal_hist=None):
        s = float(self.sched[t]); self.history.append(s); return s


class SignalEstimator:
    """Public estimate of the per-step gradient-signal norm s_t.

    Each round the server observes the aggregate A_t (already DP-protected) and knows the
    injected noise variance V_t (sigmas are public), so z_t = (||A_t||^2 - V_t)/(lr*S)^2 is an
    UNBIASED (possibly negative) estimate of s_t^2.  We fit s_t^2 = s0^2 exp(-2 b t) by weighted
    least squares over all z_t (no flooring bias) -- pure post-processing, zero privacy cost.
    """
    B_GRID = np.linspace(0.0, 0.25, 51)

    def __init__(self, floor=1e-4):
        self.z, self.floor = [], floor

    def update(self, z):
        self.z.append(float(z))

    def _fit(self):
        z = np.asarray(self.z); t = np.arange(len(z)); w = np.linspace(0.5, 1.0, len(z))
        best = None
        for b in self.B_GRID:
            e = np.exp(-2 * b * t)
            s0 = max(np.sum(w * e * z) / np.sum(w * e * e), self.floor)
            r = np.sum(w * (z - s0 * e) ** 2)
            if best is None or r < best[0]:
                best = (r, s0, b)
        return best[1], best[2]

    def ready(self):
        return len(self.z) >= 3

    def current(self):
        if not self.ready():
            return None
        s0, b = self._fit()
        return float(np.sqrt(s0 * np.exp(-2 * b * (len(self.z) - 1))))

    def forecast(self, T, t):
        """relative signal profile for rounds t..T-1 (first entry = 1)."""
        if not self.ready():
            return np.ones(T - t)
        _, b = self._fit()
        return np.exp(-b * np.arange(T - t))


class BudgetOptimalAllocator:
    """Proposed receding-horizon budget-optimal noise allocation (BONA)."""
    name = "bona"

    def __init__(self, q, steps, T, eps_budget, delta, exponent=0.5):
        self.q, self.steps, self.T, self.eps, self.delta = q, steps, T, eps_budget, delta
        self.exponent = exponent               # sigma ∝ s^exponent ; 0.5 is the KKT solution
        self.history = []
        self.base_rdp = np.zeros_like(ORDERS)

    def next_sigma(self, t, signal_hist):
        shape = signal_hist.forecast(self.T, t) ** self.exponent  # relative sigma profile
        q, st, d, base = self.q, self.steps, self.delta, self.base_rdp

        def eps_k(k):
            sig = np.clip(k * shape, SIGMA_MIN, SIGMA_MAX)
            return eps_of_schedule(q, st, sig, d, base_rdp=base)
        k = _bisect(eps_k, self.eps, SIGMA_MIN / shape.max() * 0.5, SIGMA_MAX * 4 / shape.min())
        sigma = float(np.clip(k * shape[0], SIGMA_MIN, SIGMA_MAX))
        self.base_rdp = self.base_rdp + st * rdp_step(q, sigma)
        self.history.append(sigma)
        return sigma

    def spent(self):
        return eps_from_rdp(self.base_rdp, self.delta)


ALLOCATORS = {"fixed": FixedAllocator, "decay": ExpDecayAllocator, "bona": BudgetOptimalAllocator}
