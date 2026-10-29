# BAV-FL — Budget-Aware Verifiable Federated Learning

Code accompanying *"BAV-FL: Budget-Aware Verifiable Federated Learning for
Multi-Institutional Clinical Risk Prediction"*.

Everything runs on a single CPU in ~30 minutes (NumPy only; no GPU needed).

## Setup
```bash
pip install numpy scipy pandas scikit-learn matplotlib cryptography dp-accounting
```
`dp-accounting` is only used by `tests/test_accountant.py` to cross-check our accountant.

## Data
UCI Heart Disease (Janosi, Steinbrunn, Pfisterer, Detrano, 1988; CC BY 4.0) — four real
institutions: Cleveland Clinic, Hungarian Institute of Cardiology, University Hospitals
Zurich/Basel, VA Long Beach. Put `processed.{cleveland,hungarian,switzerland,va}.data` in `data/`
(from https://archive.ics.uci.edu/dataset/45/heart+disease).

Second benchmark: Fed-TCGA-BRCA from FLamby (https://github.com/owkin/FLamby, MIT licence;
data from the TCGA Pan-Cancer Clinical Data Resource) — `brca.csv` and `centers.csv` in `data/tcga/`.

## Package layout
| file | content |
|---|---|
| `bavfl/data.py` | heart-disease sites, data-independent preprocessing, stratified per-site splits |
| `bavfl/data_tcga.py` | TCGA-BRCA regions (survival: event, time) |
| `bavfl/models.py` | logistic regression, MLP and exponential PH survival model with closed-form per-sample gradients |
| `bavfl/privacy.py` | exact vectorised RDP accountant (Poisson-subsampled Gaussian), noise allocators |
| `bavfl/planner.py` | **budget-aware configuration planner** (the BAV-FL controller) |
| `bavfl/secagg.py` | pairwise-mask secure aggregation with X25519 + HKDF-SHA256 + AES-CTR |
| `bavfl/fl.py` | federated loop: DP-SGD, verification gate, SecAgg, robust aggregators, attacks |
| `experiments/` | utility benchmark, poisoning, privacy attacks, baseline tuning, analysis, figures |

## Reproduce everything
```bash
python tests/test_accountant.py          # accountant == Google dp_accounting
sh reproduce.sh                          # all experiments -> results/, figures/
python -m experiments.analyze            # tables + Wilcoxon/Holm statistics
```
Method development used seeds 0–9; **all reported results use fresh seeds 100–119**.

## Threat model (summary)
Honest-but-curious server; up to one malicious hospital (of four). The norm-bound proof is an
idealised functionality (as realised by zero-knowledge range proofs in RoFL/EIFFeL); its
cryptographic cost is not implemented. SecAgg is the semi-honest, no-dropout core of
Bonawitz et al. (2017).
