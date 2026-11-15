"""Exploratory review of saved outputs; never writes into the frozen run."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MATCHED = ROOT / 'experiments/matched-20261107'
sys.path.insert(0, str(ROOT / 'sd-tokenskip-utility-router'))
from train_selector import load, folds

GAMMAS = (.5, .6, .7, .8, .9, 1.)
MODELS = ('qwen2.5-3b', 'qwen2.5-7b')
SEED = 20261113
WEIGHT = .1


def read(path):
    rows = [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s]
    return sorted(rows, key=lambda r: r['source_row_index'])


def auc(scores, positive):
    positive = np.asarray(positive, dtype=bool)
    n1 = int(positive.sum()); n0 = len(positive) - n1
    if not n1 or not n0:
        return None
    ranks = rankdata(scores, method='average')
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_auc(scores, labels, rng, draws=2000):
    n = len(labels); values = []
    for _ in range(draws):
        idx = rng.integers(n, size=n)
        result = auc(scores[idx], labels[idx])
        if result is not None:
            values.append(result)
    return [float(v) for v in np.quantile(values, [.025, .975])], len(values)


def test_diagnostics(model):
    folder = MATCHED / 'results' / model
    routes = sorted(json.loads((folder/'routing.json').read_text(encoding='utf8')), key=lambda r:r['source_row_index'])
    fixed = [read(folder/f'fixed_{g:.1f}.jsonl') for g in GAMMAS]
    sd = read(folder/'sd_frozen.jsonl')
    expected = list(range(1319))
    assert [r['source_row_index'] for r in routes] == expected
    for rows in fixed + [sd]:
        assert [r['source_row_index'] for r in rows] == expected
        assert all(a['question_sha256']==b['question_sha256'] for a,b in zip(rows,routes))
    y = np.array([[r['accuracy'] for r in rows] for rows in fixed],dtype=bool).T
    score = np.array([r['feature_risk_score'] for r in routes],dtype=float)
    base = y[:,-1]; rng = np.random.default_rng(SEED)
    analysis=[]
    for idx, gamma in enumerate(GAMMAS[:-1]):
        target = ~y[base,idx]
        ci,used=bootstrap_auc(score[base],target,rng)
        analysis.append({'gamma':gamma,'n_baseline_correct':int(base.sum()),'loss_events':int(target.sum()),
                         'no_loss':int((~target).sum()),'auc':auc(score[base],target),
                         'auc_ci95':ci,'bootstrap_draws_valid':used})
    sd_y=np.array([r['accuracy'] for r in sd],dtype=bool)
    discord=sd_y != base
    sample=[]
    for direction,mask in [('lost_vs_gamma1',base & ~sd_y),('recovered_vs_gamma1',~base & sd_y)]:
        indices=np.flatnonzero(mask)
        # Predetermined first three IDs in each direction; not selected for an appealing story.
        for i in indices[:3]:
            sample.append({'source_row_index':int(i),'direction':direction,
                           'sd_prediction':sd[i]['prediction'],'baseline_prediction':fixed[-1][i]['prediction'],
                           'answer':sd[i]['answer'],'sd_finish_reason':sd[i]['finish_reason'],
                           'baseline_finish_reason':fixed[-1][i]['finish_reason'],
                           'sd_separator_present':'\n\nThe final answer is:' in sd[i]['model_output'],
                           'baseline_separator_present':'\n\nThe final answer is:' in fixed[-1][i]['model_output']})
    return {'auc_compression_loss':analysis,'sd_vs_gamma1_disagreements':int(discord.sum()),
            'lost':int((base & ~sd_y).sum()),'recovered':int((~base & sd_y).sum()),
            'length_stops_among_disagreements':{
                'sd':int(sum(sd[i]['finish_reason']=='length' for i in np.flatnonzero(discord))),
                'gamma1':int(sum(fixed[-1][i]['finish_reason']=='length' for i in np.flatnonzero(discord)))},
            'sample_selection':'First three source indices in each disagreement direction; manual content review required',
            'sample':sample}


def baseline_bootstrap(model):
    folder = ROOT/'sd-tokenskip-cleanroom-v3/runs/sd-v2-v3-gamma05-10-20261025'/model/'development/tuning'
    data = [read(folder/f'gamma-{g:.1f}/predictions.jsonl') for g in GAMMAS]
    assert all(len(d)==180 for d in data)
    assert all([r['question_sha256'] for r in d]==[r['question_sha256'] for r in data[0]] for d in data)
    assert all(len({r['source_row_index'] for r in d}) == 180 for d in data)
    assert all(all(float(r['gamma']) == g for r in d) for g, d in zip(GAMMAS, data))
    y=np.array([[r['accuracy'] for r in d] for d in data],float).T
    t=np.array([[r['cot_length'] for r in d] for d in data],float).T
    rng=np.random.default_rng(SEED);counts={f'{g:.1f}':0 for g in GAMMAS}
    for _ in range(10000):
        idx=rng.integers(len(y),size=len(y)); acc=y[idx].mean(axis=0)
        util=100*acc+10*(1-t[idx].mean(axis=0)/t[idx,-1].mean())
        winner=max(range(len(GAMMAS)),key=lambda j:(util[j],acc[j],GAMMAS[j]))
        counts[f'{GAMMAS[winner]:.1f}']+=1
    return {'n_tuning':180,'bootstrap_draws':10000,'winner_counts':counts,
            'scope':'Resample tuning questions; this does not capture retraining or an independent test.'}


def fit_and_route(x,y,t,train,test,cols,use_fallback):
    xtr=x[train][:,cols]; xt=x[test][:,cols]
    mean=xtr.mean(axis=0); std=xtr.std(axis=0); std[std==0]=1
    ztr=((xtr-mean)/std).mean(axis=1); zt=((xt-mean)/std).mean(axis=1)
    thresholds=np.quantile(ztr,[.5,.7]);train_group=np.searchsorted(thresholds,ztr,side='left')
    test_group=np.searchsorted(thresholds,zt,side='left')
    lower=np.quantile(xtr,.01,axis=0);upper=np.quantile(xtr,.99,axis=0)
    eligible=np.all((xtr>=lower)&(xtr<=upper),axis=1) if use_fallback else np.ones(len(train),bool)
    outside=np.any((xt<lower)|(xt>upper),axis=1) if use_fallback else np.zeros(len(test),bool)
    denom=t[train,-1].mean();indices=np.full(len(test),5,dtype=int)
    for group in range(3):
        ids=train[(train_group==group)&eligible]
        if len(ids)<30: continue
        acc=y[ids].mean(axis=0)
        utility=100*acc-10*t[ids].mean(axis=0)/denom
        pick=max(range(6),key=lambda j:(round(float(utility[j]),12),acc[j],GAMMAS[j]))
        indices[(test_group==group)&~outside]=pick
    return indices


def ablation(model):
    x,text,y,t,ids,hashes=load(model)
    specs={'length_only':([0],True),'three_features':([0,1,2],True),
           'six_features':(list(range(6)),True),'six_no_fallback':(list(range(6)),False)}
    assigned={name:np.zeros(len(y),dtype=int) for name in specs}; fixed=np.zeros(len(y),dtype=int)
    for train,test in folds(len(y),seed=SEED):
        acc=y[train].mean(axis=0);cost=t[train].mean(axis=0);denom=cost[-1]
        pick=max(range(6),key=lambda j:(round(float(100*acc[j]-10*cost[j]/denom),12),acc[j],GAMMAS[j]))
        fixed[test]=pick
        for name,(cols,fallback) in specs.items():assigned[name][test]=fit_and_route(x,y,t,train,test,cols,fallback)
    assigned['fold_selected_fixed']=fixed
    ar=np.arange(len(y));base_t=t[:,-1].mean();result={}
    for name,route in assigned.items():
        c=y[ar,route];length=t[ar,route]; utility=100*c-10*length/base_t
        fixed_u=100*y[ar,fixed]-10*t[ar,fixed]/base_t;delta=utility-fixed_u
        rng=np.random.default_rng(SEED); samples=[]
        for _ in range(5000):samples.append(float(delta[rng.integers(len(y),size=len(y))].mean()))
        result[name]={'correct':int(c.sum()),'n':len(c),'accuracy_percent':float(100*c.mean()),
                      'mean_cot_tokens':float(length.mean()),'delta_utility_vs_fixed':float(delta.mean()),
                      'delta_utility_ci95':np.quantile(samples,[.025,.975]).tolist(),
                      'gamma_counts':{f'{g:.1f}':int((route==i).sum()) for i,g in enumerate(GAMMAS)}}
    return {'protocol':'5-fold CV on 900 exposed development questions; fit normalization, quantiles, per-group utility and fixed reference within each training fold; same questions across all gamma outputs; no accuracy gate; no-fallback removes envelope checks but keeps the minimum-group-size default',
            'scope':'Exploratory offline replay; bootstrap conditional on fitted fold policies; not an independent evaluation.',
            'seed':SEED,'input_sha256':hashes,'methods':result}


def main():
    report={'scope':'Exploratory analysis; no GPU or new generation. Test results have been previously inspected.',
            'test':{},'baseline_stability':{},'development_ablation':{}}
    for model in MODELS:
        report['test'][model]=test_diagnostics(model)
        report['baseline_stability'][model]=baseline_bootstrap(model)
        report['development_ablation'][model]=ablation(model)
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({m:{'auc_05':report['test'][m]['auc_compression_loss'][0],
                         'fixed_selection':report['baseline_stability'][m]['winner_counts'],
                         'ablation':report['development_ablation'][m]['methods']} for m in MODELS},indent=2))


if __name__ == '__main__': main()
