"""Empirical privacy validation.
(1) Gradient inversion: for logistic regression a single-record gradient g=(p-y)[x,1] reveals
    x exactly as g_w/g_b (Phong et al. 2017).  We measure re-identification of the record by the
    server under: no protection, SecAgg only, DP only, DP+SecAgg (worst case: one record per step).
(2) Membership inference: loss-threshold attack (Yeom et al. 2018) AUC on final models, vs the
    (eps,delta)-DP upper bound on attack AUC."""
import json, math, numpy as np
from sklearn.metrics import roc_auc_score
from bavfl.data import load_sites, split_sites
from bavfl.models import make_model
from bavfl.fl import run_fl
from .common import SEEDS, DELTA, get_plan

def reid(xhat, target_idx, pool):
    dist = np.linalg.norm(pool - xhat, axis=1)
    return float(dist[target_idx] <= dist.min() + 1e-9)

def inversion(n_trials=400, seed=0):
    rng = np.random.default_rng(seed); sites = load_sites()
    tr, te = split_sites(sites, SEEDS[0]); m = make_model('logreg', 17)
    w = run_fl(m, tr, te, {}, SEEDS[0], T=10)['w']                 # mid-training public model
    names = list(tr); offs = np.cumsum([0] + [len(tr[s][1]) for s in names])
    pool = np.vstack([tr[s][0] for s in names])
    def grad(s, i, C=None, sigma=0.0):
        X, y = tr[s]; g = m.per_sample_grad(w, X[i:i+1], y[i:i+1])[0]
        if C is not None:
            g = g / max(1.0, np.linalg.norm(g) / C) + rng.normal(0, sigma * C, len(g))
        return g
    plan_sig = {e: np.mean([v['sigma'] for v in get_plan(e)['sites'].values()]) for e in (1.0, 2.0, 4.0, 8.0)}
    settings = [("none", None, 0.0, False), ("secagg_only", None, 0.0, True), ("clip_only", 1.0, 0.0, False)]
    for e, sg in plan_sig.items():
        settings += [(f"dp_eps{e:g}", 1.0, sg, False), (f"dp+secagg_eps{e:g}", 1.0, sg, True)]
    out = []
    for name, C, sig, sa in settings:
        hits, errs = [], []
        for _ in range(n_trials):
            k = rng.integers(len(names)); s = names[k]; i = rng.integers(len(tr[s][1]))
            g = grad(s, i, C, sig)
            if sa:   # server only sees the sum over all hospitals (each contributes one record)
                for j, s2 in enumerate(names):
                    if j != k: g = g + grad(s2, rng.integers(len(tr[s2][1])), C, sig)
            xhat = g[:-1] / g[-1] if abs(g[-1]) > 1e-12 else np.zeros(17)
            x = tr[s][0][i]
            hits.append(reid(xhat, offs[k] + i, pool)); errs.append(np.linalg.norm(xhat - x) / (np.linalg.norm(x) + 1e-9))
        out.append(dict(setting=name, sigma=sig, reid_rate=float(np.mean(hits)), median_rel_err=float(np.median(errs)),
                        chance=1 / len(pool)))
        print(out[-1], flush=True)
    return out

def mia_auc(model, w, tr, te):
    Xm = np.vstack([v[0] for v in tr.values()]); ym = np.concatenate([v[1] for v in tr.values()])
    Xn = np.vstack([v[0] for v in te.values()]); yn = np.concatenate([v[1] for v in te.values()])
    sc = np.concatenate([-model.loss(w, Xm, ym), -model.loss(w, Xn, yn)])
    lab = np.concatenate([np.ones(len(ym)), np.zeros(len(yn))])
    return roc_auc_score(lab, sc)

def dp_auc_bound(eps, delta):
    # max AUC of a test whose ROC satisfies TPR <= e^eps FPR + delta and FNR <= e^eps TNR + delta
    f = np.linspace(0, 1, 20001)
    tpr = np.minimum.reduce([np.ones_like(f), math.exp(eps) * f + delta, 1 - math.exp(-eps) * (1 - delta - f)])
    return float(np.trapezoid(np.clip(tpr, 0, 1), f))

def mia(seeds=SEEDS[:10]):
    sites = load_sites(); out = []
    for mname in ('logreg', 'mlp'):
        dim = make_model(mname, 17).dim
        runs = [("fedavg_nonprivate", None), ("fedavg_nonprivate_long", None)] + [(f"bavfl_eps{e:g}", e) for e in (1.0, 2.0, 8.0)]
        for name, eps in runs:
            a, u = [], []
            for sd in seeds:
                tr, te = split_sites(sites, sd); m = make_model(mname, 17)
                if eps is None:
                    r = run_fl(m, tr, te, {}, sd, T=200 if 'long' in name else 30)
                else:
                    P = get_plan(eps, d=dim, tag=mname)
                    r = run_fl(m, tr, te, {"dp": "fixed", "secagg": True, "verify": "coupled"}, sd, eps=eps,
                               delta=DELTA, T=P['T'], site_cfg=P['sites'])
                a.append(mia_auc(m, r['w'], tr, te)); u.append(r['site_auc'])
            row = dict(model=mname, method=name, eps=eps, mia_auc=float(np.mean(a)), mia_sd=float(np.std(a)),
                       site_auc=float(np.mean(u)), dp_bound=None if eps is None else dp_auc_bound(eps, DELTA))
            out.append(row); print(row, flush=True)
    return out

if __name__ == '__main__':
    res = {"inversion": inversion(), "mia": mia()}
    json.dump(res, open('results/privacy_attacks.json', 'w'), indent=1)
