import json, math, os
from bavfl.data import load_sites, split_sites
from bavfl.planner import plan
SEEDS = list(range(100, 120))           # fresh seeds, never used during method development
DELTA = 1e-4
D_FEAT = 17
PLAN_CACHE = 'results/plans.json'

def train_sizes():
    tr, _ = split_sites(load_sites(), SEEDS[0])
    return {s: len(v[1]) for s, v in tr.items()}

def get_plan(eps, D_mult=1.0, d=D_FEAT + 1, tag='logreg'):
    key = f'{tag}|{eps}|{D_mult}'
    cache = json.load(open(PLAN_CACHE)) if os.path.exists(PLAN_CACHE) else {}
    if key not in cache:
        cache[key] = plan(train_sizes(), eps=eps, delta=DELTA, d=d, D=D_mult * math.sqrt(d))
        json.dump(cache, open(PLAN_CACHE, 'w'), indent=1)
    return cache[key]
