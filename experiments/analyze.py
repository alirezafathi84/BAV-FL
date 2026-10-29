import json, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
def load(path='results/utility.jsonl'):
    return pd.DataFrame([json.loads(l) for l in open(path)])
def ci(x, B=5000, seed=0):
    r=np.random.default_rng(seed); x=np.asarray(x)
    bs=r.choice(x,(B,len(x))).mean(1); return np.percentile(bs,[2.5,97.5])
def holm(p):
    p=np.asarray(p); o=np.argsort(p); adj=np.empty_like(p); m=len(p); run=0
    for i,k in enumerate(o): run=max(run,(m-i)*p[k]); adj[k]=min(run,1)
    return adj
def compare(df, ref='bavfl', metric='site_auc', others=None):
    rows=[]
    for eps in sorted(df.eps.unique()):
        a=df[(df.method==ref)&(df.eps==eps)].set_index('seed')[metric]
        if a.empty: continue
        for m in (others or sorted(df.method.unique())):
            if m==ref: continue
            b=df[(df.method==m)&((df.eps==eps)|(df.method.isin(['fedavg','central','local'])))].drop_duplicates('seed').set_index('seed')[metric]
            if b.empty: continue
            dlt=(a-b.reindex(a.index)).dropna()
            p=wilcoxon(dlt).pvalue if (dlt!=0).any() else 1.0
            rows.append(dict(eps=eps,vs=m,mean_diff=dlt.mean(),ci_lo=ci(dlt)[0],ci_hi=ci(dlt)[1],wins=(dlt>0).mean(),p=p))
    r=pd.DataFrame(rows)
    if len(r): r['p_holm']=np.concatenate([holm(g.p.values) for _,g in r.groupby('eps')])
    return r
if __name__=='__main__':
    df=load(); pd.set_option('display.width',200)
    for metric in ['site_auc','auc','bacc']:
        t=df.groupby(['eps','method'])[metric].agg(['mean','std']).round(4)
        print(metric); print(t.unstack('method')['mean'].round(4).to_string())
    print(compare(df).round(4).to_string())
