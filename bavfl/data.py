"""Fed-Heart-Disease: four real institutions from the UCI Heart Disease collection.

Preprocessing uses ONLY fixed, data-independent constants (clinical reference
values), so no pooled statistic ever crosses a hospital boundary.
"""
import os
import numpy as np
import pandas as pd

SITES = ["cleveland", "hungarian", "switzerland", "va"]
SITE_NAMES = {"cleveland": "Cleveland Clinic (USA)", "hungarian": "Hungarian Inst. Cardiology",
              "switzerland": "Univ. Hosp. Zurich/Basel", "va": "VA Long Beach (USA)"}
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
# fixed reference centres / scales (public clinical constants, not estimated from data)
REF = {"age": (54, 9), "trestbps": (130, 18), "chol": (240, 50),
       "thalach": (140, 25), "oldpeak": (1.0, 1.1)}
FEATURES = ["age", "sex", "cp1", "cp2", "cp3", "trestbps", "chol", "fbs", "restecg1",
            "restecg2", "thalach", "exang", "oldpeak", "miss_bp", "miss_chol",
            "miss_fbs", "miss_exercise"]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _featurize(df):
    df = df.copy()
    df.loc[df["chol"] == 0, "chol"] = np.nan          # 0 encodes "not measured"
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    out = pd.DataFrame(index=df.index)
    for c in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        m, s = REF[c]
        out[c] = ((df[c] - m) / s).fillna(0.0).clip(-4, 4)
    out["sex"] = df["sex"].fillna(0.5)
    for k in (1, 2, 3):
        out[f"cp{k}"] = (df["cp"] == k).astype(float)
    out["fbs"] = df["fbs"].fillna(0.0)
    out["restecg1"] = (df["restecg"] == 1).astype(float)
    out["restecg2"] = (df["restecg"] == 2).astype(float)
    out["exang"] = df["exang"].fillna(0.0)
    out["miss_bp"] = df["trestbps"].isna().astype(float)
    out["miss_chol"] = df["chol"].isna().astype(float)
    out["miss_fbs"] = df["fbs"].isna().astype(float)
    out["miss_exercise"] = (df["thalach"].isna() | df["oldpeak"].isna()).astype(float)
    X = out[FEATURES].to_numpy(float)
    y = (df["num"] > 0).astype(float).to_numpy()
    return X, y


def load_sites(data_dir=DATA_DIR):
    sites = {}
    for s in SITES:
        df = pd.read_csv(os.path.join(data_dir, f"processed.{s}.data"), header=None,
                         names=COLS, na_values="?")
        sites[s] = _featurize(df)
    return sites


def split_sites(sites, seed, test_frac=0.3):
    """Stratified per-site train/test split (re-drawn for every seed)."""
    rng = np.random.default_rng(seed)
    train, test = {}, {}
    for s, (X, y) in sites.items():
        tr_idx, te_idx = [], []
        for c in (0, 1):
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            n_te = int(round(test_frac * len(idx)))
            te_idx.extend(idx[:n_te]); tr_idx.extend(idx[n_te:])
        tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
        rng.shuffle(tr_idx)
        train[s] = (X[tr_idx], y[tr_idx])
        test[s] = (X[te_idx], y[te_idx])
    return train, test


if __name__ == "__main__":
    sites = load_sites()
    for s, (X, y) in sites.items():
        print(f"{s:12s} n={len(y):4d} prevalence={y.mean():.3f} d={X.shape[1]}")
