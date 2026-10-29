"""BAV-FL training loop and baselines.

A method is described by a config dict:
  dp        : None | "fixed" | "decay" | "bona"       (noise allocator)
  secagg    : bool                                      (cryptographic pairwise masking)
  verify    : None | "static" | "coupled"               (per-client norm-bound verification)
  robust    : None | "median" | "trimmed"               (plaintext robust aggregation; no SecAgg)
  prox_mu   : float                                     (FedProx proximal term)
"""
import math
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from .privacy import ALLOCATORS, SignalEstimator
from .secagg import SecureAggregator

BETA = 1e-6          # failure probability for the honest-update norm bound


# ----------------------------------------------------------------------------- local training
def local_train(model, w0, X, y, rng, *, lr, B, steps, C=None, sigma=None, prox_mu=0.0, wd=0.0, return_noise=False):
    """Returns the model delta after `steps` local steps.

    DP mode (sigma not None): Poisson sampling with rate B/n, per-sample clipping to C,
    Gaussian noise N(0, sigma^2 C^2) on the clipped sum, normalised by expected batch B.
    """
    n = len(y)
    w = w0.copy()
    q = min(B / n, 1.0)
    perm, pos = rng.permutation(n), 0
    xi = np.zeros(model.dim)                # accumulated noise contribution to the update
    for _ in range(steps):
        if sigma is not None:
            idx = np.flatnonzero(rng.random(n) < q)
            if len(idx):
                G = model.per_sample_grad(w, X[idx], y[idx])
                norms = np.linalg.norm(G, axis=1, keepdims=True)
                G = G / np.maximum(1.0, norms / C)
                g = G.sum(0)
            else:
                g = np.zeros(model.dim)
            z = rng.normal(0, sigma * C, model.dim) / (q * n)
            g = g / (q * n) + z
            xi -= lr * z
        else:
            if pos + B > n:
                perm, pos = rng.permutation(n), 0
            idx = perm[pos:pos + B]; pos += B
            g = model.per_sample_grad(w, X[idx], y[idx]).mean(0)
        if prox_mu:
            g = g + prox_mu * (w - w0)
        if wd:
            g = g + wd * w
        w = w - lr * g
    if return_noise:
        return w - w0, xi
    return w - w0


# ----------------------------------------------------------------------------- attacks
def apply_attack(kind, honest_update, bound, rng, scale=10.0):
    u = honest_update
    nrm = np.linalg.norm(u) + 1e-12
    if kind == "signflip_scaled":           # boosted model-replacement style
        return -scale * u
    if kind == "gaussian":                  # random large-norm garbage
        return rng.normal(0, scale * nrm / np.sqrt(len(u)), len(u))
    if kind == "bounded_signflip":          # adaptive adversary that knows the bound
        return -0.999 * bound * u / nrm
    return u                                 # label-flip is applied to the data instead


# ----------------------------------------------------------------------------- evaluation
def cindex(y, score):
    """Harrell's concordance index; y = [event, time]."""
    E, T = y[:, 0], y[:, 1]
    comp = (T[:, None] < T[None, :]) & (E[:, None] == 1)
    if not comp.any():
        return np.nan
    d = score[:, None] - score[None, :]
    return float(((d > 0) + 0.5 * (d == 0))[comp].mean())


def evaluate(model, w_by_site, test):
    if next(iter(test.values()))[1].ndim == 2:        # survival task -> C-index
        ys, ps, sc = [], [], []
        for s, (X, y) in test.items():
            w = w_by_site[s] if isinstance(w_by_site, dict) else w_by_site
            p = model.predict(w, X); ys.append(y); ps.append(p)
            c = cindex(y, p)
            if not np.isnan(c): sc.append(c)
        return {"auc": cindex(np.vstack(ys), np.concatenate(ps)), "bacc": float("nan"), "site_auc": float(np.mean(sc))}
    ys, ps, site_auc = [], [], []
    for s, (X, y) in test.items():
        w = w_by_site[s] if isinstance(w_by_site, dict) else w_by_site
        p = model.predict(w, X)
        ys.append(y); ps.append(p)
        if 0 < y.sum() < len(y):
            site_auc.append(roc_auc_score(y, p))
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {"auc": roc_auc_score(y, p),
            "bacc": balanced_accuracy_score(y, (p >= 0.5).astype(int)),
            "site_auc": float(np.mean(site_auc))}


# ----------------------------------------------------------------------------- federated run
def run_fl(model, train, test, cfg, seed, *, T=30, E=1, B=16, lr=0.5, C=1.0,
           eps=3.0, delta=1e-4, malicious=(), attack=None, kappa=4.0, return_trace=False, site_cfg=None):
    rng = np.random.default_rng(seed)
    sites = list(train)
    n = {s: len(train[s][1]) for s in sites}
    N = sum(n.values())
    p = {s: n[s] / N for s in sites}                       # public, registered weights
    Bk = {s: min(B, n[s]) for s in sites}
    steps = {s: E * math.ceil(n[s] / B) for s in sites}
    if site_cfg:                                            # planner output: per-site batch & steps
        Bk = {s: min(site_cfg[s]["B"], n[s]) for s in sites}
        steps = {s: site_cfg[s]["m"] for s in sites}
    q = {s: Bk[s] / n[s] for s in sites}
    d = model.dim
    alloc = {s: ALLOCATORS[cfg["dp"]](q[s], steps[s], T, eps, delta) for s in sites} if cfg.get("dp") else None
    sa = SecureAggregator(sites) if cfg.get("secagg") else None
    w = model.init(np.random.default_rng(seed + 7))
    signal_hist, trace = SignalEstimator(), {"sigma": {s: [] for s in sites}, "signal": [], "rejected": [], "auc": []}
    det = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    rms_hist = []                                            # per-step client-level signal^2 estimates

    for t in range(T):
        ups, sig, bounds, verified_q = {}, {}, {}, {}
        for s in sites:
            Xs, ys = train[s]
            if s in malicious and attack == "labelflip":
                ys = 1.0 - ys
            sigma = alloc[s].next_sigma(t, signal_hist) if alloc else None
            sig[s] = sigma
            u = local_train(model, w, Xs, ys, rng, lr=lr, B=Bk[s], steps=steps[s], C=C,
                            sigma=sigma, prox_mu=cfg.get("prox_mu", 0.0),
                            wd=cfg.get("wd", 0.0), return_noise=True)
            u, xi = u if isinstance(u, tuple) else (u, np.zeros(d))
            # honest-update norm bound from PUBLIC quantities only
            noise_part = 0.0 if sigma is None else lr * math.sqrt(steps[s]) * sigma * C / (q[s] * n[s]) \
                * (math.sqrt(d) + math.sqrt(2 * math.log(1 / BETA)))
            sig_static = 1.5 * lr * steps[s] * C
            vn = str(cfg.get("verify", "")).endswith("_vn")      # proof also attests correct noise sampling
            if str(cfg.get("verify", "")).startswith("rms") and len(rms_hist) >= 2:
                r2 = max(np.mean(rms_hist[-3:]), (0.02 * C) ** 2)
                sig_part = min(sig_static, cfg.get("kappa", 2.0) * 1.5 * lr * steps[s] * math.sqrt(r2))
            elif str(cfg.get("verify", "")).startswith("coupled") and signal_hist.ready():
                sig_part = min(sig_static, kappa * 1.5 * lr * steps[s] * signal_hist.current())
            else:
                sig_part = sig_static
            bounds[s] = sig_part if vn else sig_part + noise_part
            if s in malicious and attack not in (None, "labelflip"):
                if vn and attack == "bounded_signflip":   # adaptive adversary: honest noise + max in-bound push
                    u = xi + apply_attack(attack, u - xi, bounds[s], rng)
                else:
                    u = apply_attack(attack, u, bounds[s], rng)
            ups[s] = u
            verified_q[s] = u - xi if vn else u     # quantity whose norm the proof attests
            if alloc:
                trace["sigma"][s].append(sigma)

        # ---- verification (per-client proof of ||u|| <= M; server learns accept/reject only)
        if cfg.get("verify"):
            accepted = [s for s in sites if np.linalg.norm(verified_q[s]) <= bounds[s]]
        else:
            accepted = list(sites)
        for s in sites:
            bad, ok = s in malicious, s in accepted
            det["tp" if bad and not ok else "fn" if bad else "fp" if not ok else "tn"] += 1
        trace["rejected"].append([s for s in sites if s not in accepted])
        if not accepted:
            continue

        # ---- aggregation
        pa = {s: p[s] / sum(p[a] for a in accepted) for s in accepted}
        if cfg.get("robust") == "median":
            agg = np.median(np.stack([ups[s] for s in accepted]), axis=0)
        elif cfg.get("robust") == "trimmed":
            M = np.sort(np.stack([ups[s] for s in accepted]), axis=0)
            k = 1 if len(accepted) >= 4 else 0
            agg = M[k:len(accepted) - k].mean(0)
        elif sa is not None:
            agg, _ = sa.aggregate({s: pa[s] * ups[s] for s in accepted}, rnd=t)
        else:
            agg = sum(pa[s] * ups[s] for s in accepted)
        # ---- client-level signal RMS: sum_k p_k (||u_k||^2 - E||xi_k||^2)/(lr m_k)^2, aggregated securely
        if alloc:
            rms_hist.append(sum(pa[s] * (np.dot(ups[s], ups[s]) - d * (lr * sig[s] * C / Bk[s]) ** 2 * steps[s])
                                / (lr * steps[s]) ** 2 for s in accepted))
        # ---- privacy-aware server step (pure post-processing => no privacy cost)
        if alloc and cfg.get("server") == "stein":
            noise_var_pre = d * sum((pa[s] * lr * sig[s] * C / (q[s] * n[s])) ** 2 * steps[s] for s in accepted)
            shrink = max(0.0, 1.0 - noise_var_pre / max(np.dot(agg, agg), 1e-12))
            trace.setdefault("shrink", []).append(shrink)
            agg_step = shrink * agg
        else:
            agg_step = agg
        w = w + agg_step
        if cfg.get("ema"):
            w_ema = w.copy() if t == 0 else cfg["ema"] * w_ema + (1 - cfg["ema"]) * w
        if cfg.get("tail"):
            t0 = int(T * (1 - cfg["tail"]))
            if t >= t0:
                k_ = t - t0
                w_ema = w.copy() if k_ == 0 else (k_ * w_ema + w) / (k_ + 1)

        # ---- public signal estimate (post-processing of the DP aggregate => free)
        Sbar = sum(pa[s] * steps[s] for s in accepted)
        noise_var = 0.0
        if alloc:
            noise_var = d * sum((pa[s] * lr * sig[s] * C / (q[s] * n[s])) ** 2 * steps[s] for s in accepted)
        s2 = (np.dot(agg, agg) - noise_var) / (lr * Sbar) ** 2
        signal_hist.update(s2)
        trace["signal"].append(signal_hist.current())
        if return_trace:
            trace["auc"].append(evaluate(model, w_ema if (cfg.get("ema") and t>0) else w, test)["site_auc"])

    w_out = w_ema if (cfg.get("ema") or cfg.get("tail")) else w
    res = evaluate(model, w_out, test)
    res["eps_spent"] = max(a.spent() for a in alloc.values()) if alloc else float("inf")
    res["det"] = det
    if sa is not None:
        res["secagg_ms"] = 1000 * float(np.mean(sa.timing))
    if return_trace:
        res["trace"] = trace
    res["w"] = w_out
    return res


# ----------------------------------------------------------------------------- non-federated references
def run_centralized(model, train, test, seed, *, T=30, E=1, B=16, lr=0.5, C=1.0, eps=None, delta=1e-4):
    from .privacy import FixedAllocator
    rng = np.random.default_rng(seed)
    X = np.vstack([v[0] for v in train.values()]); y = np.concatenate([v[1] for v in train.values()])
    steps = T * E * math.ceil(len(y) / B)
    sigma = None
    if eps is not None:
        sigma = FixedAllocator(min(B / len(y), 1.0), steps, 1, eps, delta).sigma
    w = model.init(np.random.default_rng(seed + 7))
    w = w + local_train(model, w, X, y, rng, lr=lr, B=B, steps=steps, C=C, sigma=sigma)
    r = evaluate(model, w, test); r["w"] = w
    return r


def run_local(model, train, test, seed, *, T=30, E=1, B=16, lr=0.5):
    rng = np.random.default_rng(seed)
    ws = {}
    for s, (X, y) in train.items():
        w0 = model.init(np.random.default_rng(seed + 7))
        ws[s] = w0 + local_train(model, w0, X, y, rng, lr=lr, B=B, steps=T * E * math.ceil(len(y) / B))
    return evaluate(model, ws, test)
