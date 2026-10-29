"""Fed-TCGA-BRCA (FLamby): 1,088 breast-cancer patients from six real geographic regions.
Outcome: overall survival (event E, time T in years). Data-independent preprocessing:
age is centred/scaled by fixed constants (58 y, 13 y); all other features are binary."""
import os, numpy as np, pandas as pd
from .data import DATA_DIR
REGIONS = ["Northeast", "South", "West", "Midwest", "Europe", "Canada"]

def load_sites(data_dir=os.path.join(DATA_DIR, "tcga")):
    d = pd.read_csv(os.path.join(data_dir, "brca.csv")); c = pd.read_csv(os.path.join(data_dir, "centers.csv"))
    d["region"] = d.pid.str.split("-").str[1].map(dict(zip(c["TSS Code"], c["Region"])))
    feats = [k for k in d.columns if k not in ("pid", "E", "T", "region")]
    X = d[feats].to_numpy(float); ai = feats.index("age_at_index")
    X[:, ai] = np.clip((X[:, ai] - 58.0) / 13.0, -4, 4)
    Y = np.stack([d.E.to_numpy(float), d["T"].to_numpy(float) / 365.25], 1)
    return {r: (X[(d.region == r).to_numpy()], Y[(d.region == r).to_numpy()]) for r in REGIONS}

def split_sites(sites, seed, test_frac=0.3):
    rng = np.random.default_rng(seed); tr, te = {}, {}
    for s, (X, Y) in sites.items():
        tri, tei = [], []
        for c in (0, 1):
            idx = np.flatnonzero(Y[:, 0] == c); rng.shuffle(idx); k = int(round(test_frac * len(idx)))
            tei += list(idx[:k]); tri += list(idx[k:])
        tri = np.array(tri); rng.shuffle(tri); tei = np.array(tei)
        tr[s] = (X[tri], Y[tri]); te[s] = (X[tei], Y[tei])
    return tr, te
