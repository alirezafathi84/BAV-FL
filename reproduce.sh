#!/bin/sh
set -e
mkdir -p results figures
python -m experiments.utility fedavg,central,local 1
python -m experiments.utility bavfl,dp_default,dp_fedprox,dp_decay,central_dp,bavfl_Dhalf,bavfl_D2 0.5,1,2,4,8
python -m experiments.attacks
python -m experiments.tune_baseline 0.5,1,2,4,8
python -m experiments.utility dp_tuned 0.5,1,2,4,8
python -m experiments.privacy_attacks
python -m experiments.tcga
python -m experiments.summary
python -m experiments.figures
