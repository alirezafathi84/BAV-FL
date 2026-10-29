import json, math, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from bavfl.data import load_sites, SITE_NAMES
from .analyze import load, ci

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight"})
COL = {"bavfl": "#1f5fa8", "dp_default": "#9a9a9a", "dp_fedprox": "#c98a2b", "dp_decay": "#6aa36f",
       "dp_tuned": "#b04a5a", "central_dp": "#7a5ca8"}
LAB = {"bavfl": "BAV-FL (ours)", "dp_default": "DP-FedAvg (default)", "dp_fedprox": "DP-FedProx",
       "dp_decay": "DP-FedAvg + noise decay", "dp_tuned": "DP-FedAvg (grid-tuned*)",
       "central_dp": "Centralised DP-SGD"}


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.4, 3.3)); ax.axis("off"); ax.set_xlim(0, 10.4); ax.set_ylim(-0.3, 4.4)
    def box(x, y, w, h, t, c, fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=c, ec="#333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
    def arr(a, b, **k):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=0.9, **k))
    box(0.1, 3.25, 3.3, 1.0, "Budget-aware planner (offline)\n$\\min\\ \\Phi(B_k, m_k, T)$ with exact RDP $\\sigma_k$\nPublic inputs only: $n_k, d, \\varepsilon, \\delta, C$", "#dbe7f5")
    for i, s in enumerate(["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]):
        box(0.1, 2.25 - i * 0.68, 2.6, 0.56, f"{s}\nlocal DP-SGD $(B_k, m_k, \\sigma_k)$", "#eef4e8", 6.6)
        arr((2.75, 2.53 - i * 0.68), (3.2, 1.35))
    box(3.2, 0.55, 2.2, 1.6, "Norm-bound proof\n$\\|u_k\\| \\leq M_k(\\sigma_k, \\hat{s}_t)$\nserver learns\naccept / reject only", "#fbecd5")
    box(5.75, 0.55, 2.1, 1.6, "Secure aggregation\nX25519 + HKDF\n+ AES-CTR masks\nserver sees $\\Sigma$ only", "#f3dde1")
    box(8.2, 0.55, 2.1, 1.6, "Global model\n$w_{t+1} = w_t + \\sum_k p_k u_k$", "#e4e4f2")
    box(5.75, 3.25, 4.55, 1.0, "Public signal estimator $\\hat{s}_t$\n(post-processing of the DP aggregate)\nzero additional privacy cost", "#efefef")
    arr((1.6, 3.25), (1.4, 2.85)); arr((5.4, 1.35), (5.75, 1.35)); arr((7.85, 1.35), (8.2, 1.35))
    arr((9.25, 2.15), (9.25, 3.25)); arr((5.75, 3.5), (4.3, 2.15), ls="--")
    ax.text(8.0, -0.15, "Model broadcast to all hospitals each round", fontsize=6.3, ha="center", style="italic")
    fig.savefig("figures/fig1_architecture.png"); plt.close(fig)


def fig_data():
    sites = load_sites(); names = list(sites)
    n = [len(sites[s][1]) for s in names]; prev = [sites[s][1].mean() for s in names]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    lab = ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
    ax[0].bar(lab, n, color="#1f5fa8"); ax[0].set_ylabel("Patients"); ax[0].set_title("(a) Cohort size")
    ax[1].bar(lab, prev, color="#b04a5a"); ax[1].set_ylabel("Disease prevalence"); ax[1].set_ylim(0, 1)
    ax[1].set_title("(b) Label shift across hospitals")
    for a in ax: a.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout(); fig.savefig("figures/fig2_data.png"); plt.close(fig)


def fig_privacy_utility():
    df = load()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, metric, ttl in [(axs[0], "site_auc", "(a) Within-hospital AUC (primary)"), (axs[1], "auc", "(b) Pooled AUC")]:
        for m in ["dp_default", "dp_fedprox", "dp_decay", "dp_tuned", "central_dp", "bavfl"]:
            g = df[df.method == m]
            if g.empty: continue
            eps = sorted(g.eps.unique()); mu = [g[g.eps == e][metric].mean() for e in eps]
            lo = [ci(g[g.eps == e][metric])[0] for e in eps]; hi = [ci(g[g.eps == e][metric])[1] for e in eps]
            ax.plot(eps, mu, "o-", color=COL[m], label=LAB[m], lw=2.2 if m == "bavfl" else 1.1, ms=3.5)
            if m in ("bavfl", "dp_default"):
                ax.fill_between(eps, lo, hi, color=COL[m], alpha=0.13)
        fa = df[df.method == "fedavg"][metric].mean()
        ax.axhline(fa, color="k", ls="--", lw=0.8, label="FedAvg (non-private)")
        ax.set_xscale("log", base=2); ax.set_xlabel("Privacy budget ε (per hospital, δ=1e-4)"); ax.set_title(ttl)
        ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    axs[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig("figures/fig3_privacy_utility.png"); plt.close(fig)


def fig_planner():
    P = json.load(open("results/plans.json"))
    eps = sorted({float(k.split("|")[1]) for k in P if k.startswith("logreg") and k.endswith("|1.0")})
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for s, c in zip(["cleveland", "hungarian", "switzerland", "va"], ["#1f5fa8", "#c98a2b", "#b04a5a", "#6aa36f"]):
        rows = [P[f"logreg|{e}|1.0"]["sites"][s] for e in eps]
        axs[0].plot(eps, [r["B"] for r in rows], "o-", color=c, label=s.capitalize(), ms=3)
        axs[1].plot(eps, [r["m"] * P[f"logreg|{e}|1.0"]["T"] for r, e in zip(rows, eps)], "o-", color=c, ms=3)
        axs[2].plot(eps, [r["sigma"] for r in rows], "o-", color=c, ms=3)
    for a, t in zip(axs, ["(a) Batch size B_k", "(b) Total local steps S_k", "(c) Noise multiplier σ_k"]):
        a.set_xscale("log", base=2); a.set_xlabel("ε"); a.set_title(t)
        a.set_xticks([0.5, 1, 2, 4, 8]); a.set_xticklabels(["0.5", "1", "2", "4", "8"])
    axs[1].set_yscale("log"); axs[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig("figures/fig4_planner.png"); plt.close(fig)


def fig_attacks():
    A = pd.DataFrame([json.loads(l) for l in open("results/attacks.jsonl")])
    att = ["None", "signflip_scaled", "gaussian", "labelflip", "bounded_signflip"]
    attl = ["No attack", "Scaled sign-flip", "Gaussian", "Label flip", "Bounded sign-flip"]
    dfs = ["no_defense_SA", "static_bound_SA", "bavfl_coupled_SA", "coord_median", "trimmed_mean"]
    dfl = ["FedAvg+SecAgg (no defence)", "Static norm bound+SecAgg", "BAV-FL coupled bound+SecAgg",
           "Coord. median (no SecAgg)", "Trimmed mean (no SecAgg)"]
    cols = ["#9a9a9a", "#c98a2b", "#1f5fa8", "#6aa36f", "#7a5ca8"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); w = 0.16
    for j, (d, l, c) in enumerate(zip(dfs, dfl, cols)):
        mu = [A[(A.attack == a) & (A.defense == d)].site_auc.mean() for a in att]
        sd = [A[(A.attack == a) & (A.defense == d)].site_auc.std() for a in att]
        ax.bar(np.arange(5) + (j - 2) * w, mu, w, yerr=sd, color=c, label=l, error_kw=dict(lw=0.6))
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(attl); ax.set_ylim(0.0, 0.92)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_ylabel("Within-hospital AUC"); ax.legend(fontsize=6.3, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout(); fig.savefig("figures/fig5_attacks.png"); plt.close(fig)


def fig_privacy_attacks():
    R = json.load(open("results/privacy_attacks.json"))
    inv = R["inversion"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
    lab = [r["setting"].replace("_", " ").replace("eps", "ε=") for r in inv]
    v = [r["reid_rate"] for r in inv]
    axs[0].barh(lab, v, color=["#b04a5a" if x > 0.05 else "#1f5fa8" for x in v])
    for i, x in enumerate(v):
        axs[0].text(x + 0.02, i, f"{100*x:.1f}%", va="center", fontsize=6.5)
    axs[0].invert_yaxis(); axs[0].set_xlim(0, 1.18); axs[0].set_xlabel("Record re-identification rate")
    axs[0].set_title("(a) Gradient-inversion attack"); axs[0].tick_params(axis="y", labelsize=6.8)
    M = pd.DataFrame(R["mia"])
    nm = {"fedavg_nonprivate": "non-private", "fedavg_nonprivate_long": "non-priv. long",
          "bavfl_eps1": "ours ε=1", "bavfl_eps2": "ours ε=2", "bavfl_eps8": "ours ε=8"}
    x0 = 0
    for mname, c, ttl in [("logreg", "#1f5fa8", "Logistic regression"), ("mlp", "#b04a5a", "MLP")]:
        g = M[M.model == mname]; xs = np.arange(len(g)) + x0
        axs[1].bar(xs, g.mia_auc, yerr=g.mia_sd, color=c, alpha=0.85, error_kw=dict(lw=0.6), label=ttl)
        axs[1].set_xticks(list(axs[1].get_xticks()) if x0 else [])
        x0 += len(g) + 1
    ticks = list(range(5)) + list(range(6, 11))
    axs[1].set_xticks(ticks); axs[1].set_xticklabels([nm[m] for m in M.method], rotation=55, ha="right", fontsize=6.6)
    axs[1].axhline(0.5, color="k", lw=0.6, ls=":"); axs[1].set_ylim(0.45, 0.62)
    axs[1].set_ylabel("Membership-inference AUC"); axs[1].set_title("(b) Loss-threshold MIA")
    axs[1].legend(fontsize=6.8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig("figures/fig6_privacy_attacks.png"); plt.close(fig)


def fig_bound_validation():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    from bavfl.data import split_sites
    from bavfl.privacy import FixedAllocator
    tr, _ = split_sites(load_sites(), 0); n = {s: len(v[1]) for s, v in tr.items()}; N = sum(n.values())
    for ax, eps in zip(axs, [1.0, 3.0]):
        g = json.load(open(f"results/dev/grid_logreg_{eps}.json")); phi, y = [], []
        for c in g:
            v = 0
            for s in n:
                B = min(c["B"], n[s]); S = c["E"] * math.ceil(n[s] / c["B"]); q = B / n[s]
                sg = FixedAllocator(q, S, c["T"], eps, 1e-4).sigma; St = S * c["T"]
                v += n[s] / N * (18 / (2 * 0.5 * St) + 0.25 * (1 + 18 * (sg / B) ** 2))
            phi.append(v); y.append(c["site_auc"])
        ax.scatter(phi, y, s=14, color="#1f5fa8"); i = int(np.argmin(phi))
        ax.scatter([phi[i]], [y[i]], s=60, facecolor="none", edgecolor="#b04a5a", lw=1.5, label="planner choice")
        ax.set_xscale("log"); ax.set_xlabel("Planner objective Φ (lower = better)"); ax.set_ylabel("Within-hospital AUC")
        ax.set_title(f"ε = {eps:g} (development seeds 0–7)"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig("figures/fig7_bound_validation.png"); plt.close(fig)


def fig_tcga():
    df = load("results/tcga/utility.jsonl")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, metric, ttl in [(axs[0], "site_auc", "(a) Within-region C-index (primary)"), (axs[1], "auc", "(b) Pooled C-index")]:
        for m in ["dp_default", "dp_fedprox", "dp_decay", "dp_tuned", "central_dp", "bavfl"]:
            g = df[df.method == m]; eps = sorted(g.eps.unique())
            mu = [g[g.eps == e][metric].mean() for e in eps]
            ax.plot(eps, mu, "o-", color=COL[m], label=LAB[m], lw=2.2 if m == "bavfl" else 1.1, ms=3.5)
            if m in ("bavfl", "dp_default"):
                ax.fill_between(eps, [ci(g[g.eps == e][metric])[0] for e in eps], [ci(g[g.eps == e][metric])[1] for e in eps], color=COL[m], alpha=0.13)
        ax.axhline(df[df.method == "fedavg"][metric].mean(), color="k", ls="--", lw=0.8, label="FedAvg (non-private)")
        ax.set_xscale("log", base=2); ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
        ax.set_xlabel("Privacy budget ε (per region, δ=1e-4)"); ax.set_title(ttl)
    axs[0].legend(fontsize=6.3, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig("figures/fig8_tcga.png"); plt.close(fig)



if __name__ == "__main__":
    for f in [fig_architecture, fig_data, fig_privacy_utility, fig_planner, fig_attacks, fig_privacy_attacks, fig_bound_validation, fig_tcga]:
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAIL", f.__name__, repr(e))
