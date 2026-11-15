"""Development-only screening; six counterfactuals always share a question fold."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
from selector import FEATURES,GAMMAS,fit,predict,fixed_index

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
SEED=20261107
MODES=("linear","quadratic","lexical")
ALPHAS=(10.,100.,1000.)
KAPPAS=(0.,.5,1.,2.)
LAM=.1

def readrows(path):return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load(model):
    root=REPO/"sd-tokenskip-cleanroom-v3/runs/sd-v2-v3-gamma05-10-20261025"/model/"development"
    features=[];texts=[];ys=[];ts=[];identities=[];hashes={}
    for split in ("fit","tuning","validation"):
        f=root/split/"routing_assignments.jsonl";assign=readrows(f);hashes[f.relative_to(REPO).as_posix()]=sha(f)
        byindex={r["source_row_index"]:r for r in assign};indices=sorted(byindex)
        grids=[]
        for gamma in GAMMAS:
            f=root/split/f"gamma-{gamma:.1f}/predictions.jsonl";rows=readrows(f)
            assert sorted(r["source_row_index"] for r in rows)==indices
            rows=sorted(rows,key=lambda r:r["source_row_index"])
            assert all(r["question_sha256"]==byindex[r["source_row_index"]]["question_sha256"]
                       and float(r["gamma"]) == gamma for r in rows)
            grids.append(rows);hashes[f.relative_to(REPO).as_posix()]=sha(f)
        for j,i in enumerate(indices):
            row=byindex[i];features.append([row["features"][n] for n in FEATURES]);identities.append(row["question_sha256"])
            # The ratio-1 prompt contains the exact question between fixed delimiters.
            prompt=grids[-1][j]["prompt"]
            prefix="<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
            assert prompt.count(prefix)==1
            question=prompt.split(prefix)[1].split("<|im_end|>\n<|im_start|>assistant\n")[0]
            assert hashlib.sha256(question.encode()).hexdigest()==row["question_sha256"]
            texts.append(question)
            ys.append([bool(g[j]["accuracy"]) for g in grids]);ts.append([g[j]["cot_length"] for g in grids])
    assert len(features)==900 and len(set(identities))==900
    return np.array(features),texts,np.array(ys,float),np.array(ts,float),identities,hashes

def folds(n,k=5,seed=SEED):
    parts=np.array_split(np.random.default_rng(seed).permutation(n),k)
    for val in parts:
        mask=np.ones(n,bool);mask[val]=False
        yield np.flatnonzero(mask),val

def scores(y,t,routes):
    idx=np.arange(len(y));chosen_y=y[idx,routes];chosen_t=t[idx,routes]
    utility=100*chosen_y+100*LAM*(1-chosen_t/t[:,-1].mean())
    return utility,{"n":len(y),"correct":int(chosen_y.sum()),"accuracy_percent":100*float(chosen_y.mean()),
                    "mean_cot_tokens":float(chosen_t.mean()),"utility":float(utility.mean()),
                    "gamma_counts":{f"{g:.1f}":int((routes==j).sum()) for j,g in enumerate(GAMMAS)}}

def screen(x,texts,y,t,seed=SEED,embeddings=None):
    modes=MODES+(("semantic",) if embeddings is not None else ())
    configs=[(mode,alpha,kappa) for mode in modes for alpha in ALPHAS for kappa in KAPPAS]
    routes={str(c):np.zeros(len(y),int) for c in configs};baseline=np.zeros(len(y),int)
    for train,val in folds(len(y),seed=seed):
        baseline[val]=fixed_index(y[train],t[train],LAM)
        for mode in modes:
            for alpha in ALPHAS:
                model=fit(x[train],[texts[i] for i in train],y[train],t[train],mode,alpha,LAM,embeddings=None if embeddings is None else embeddings[train])
                for kappa in KAPPAS:
                    r,_=predict(model,x[val],[texts[i] for i in val],kappa,embeddings=None if embeddings is None else embeddings[val])
                    routes[str((mode,alpha,kappa))][val]=r
    base_u,base_metrics=scores(y,t,baseline)
    evaluations=[]
    for mode,alpha,kappa in configs:
        r=routes[str((mode,alpha,kappa))];u,m=scores(y,t,r)
        delta=u-base_u
        evaluations.append({"mode":mode,"alpha":alpha,"kappa":kappa,**m,
                            "delta_utility_vs_cv_fixed":float(delta.mean()),
                            "paired_standard_error":float(delta.std(ddof=1)/math.sqrt(len(delta))),
                            "routed_fraction":float((r!=baseline).mean())})
    best=max(evaluations,key=lambda d:d["delta_utility_vs_cv_fixed"])
    threshold=best["delta_utility_vs_cv_fixed"]-best["paired_standard_error"]
    eligible=[d for d in evaluations if d["delta_utility_vs_cv_fixed"]>=threshold-1e-12]
    # Include the fixed policy in selection: do not force adaptation without evidence.
    disabled=0>=threshold
    chosen=min(eligible,key=lambda d:(d["routed_fraction"],-d["kappa"],-d["alpha"],modes.index(d["mode"])))
    return chosen,disabled,evaluations,base_metrics,routes[str((chosen["mode"],chosen["alpha"],chosen["kappa"]))],baseline

def run(model,use_embeddings=False):
    x,texts,y,t,ids,input_hashes=load(model)
    embeddings=None
    if use_embeddings:
        embedding_path = ROOT / "data/development-embeddings.npz"
        metadata = json.loads(embedding_path.with_suffix(".json").read_text(encoding="utf-8"))
        assert sha(embedding_path) == metadata["embedding_sha256"]
        with np.load(embedding_path, allow_pickle=False) as cached:
            assert cached["question_sha256"].tolist() == ids
            embeddings = cached["embeddings"].astype(float)
        input_hashes[embedding_path.relative_to(REPO).as_posix()] = sha(embedding_path)
    chosen,disabled,evaluations,base_metrics,selected,baseline=screen(x,texts,y,t,embeddings=embeddings)
    # Outer folds estimate the entire hyperparameter/fixed-policy selection process.
    nested=np.zeros(len(y),int);outer_fixed=np.zeros(len(y),int);outer=[]
    for fold,(train,val) in enumerate(folds(len(y),seed=SEED+1)):
        c,off,_,_,_,_=screen(x[train],[texts[i] for i in train],y[train],t[train],seed=SEED+10+fold,embeddings=None if embeddings is None else embeddings[train])
        fitted=fit(x[train],[texts[i] for i in train],y[train],t[train],c["mode"],c["alpha"],LAM,embeddings=None if embeddings is None else embeddings[train])
        fitted.update(kappa=c["kappa"],routing_disabled=off)
        nested[val]=predict(fitted,x[val],[texts[i] for i in val],embeddings=None if embeddings is None else embeddings[val])[0]
        outer_fixed[val]=fitted["baseline_index"]
        outer.append({"fold":fold,"n_train":len(train),"n_evaluation":len(val),"routing_disabled":off,
                      "selected":{k:c[k] for k in ("mode","alpha","kappa")}})
    u,m=scores(y,t,nested);bu,bm=scores(y,t,outer_fixed);delta=u-bu
    ci=np.quantile(np.random.default_rng(SEED).choice(delta,(20000,len(delta))).mean(1),[.025,.975])
    fitted=fit(x,texts,y,t,chosen["mode"],chosen["alpha"],LAM,embeddings=embeddings)
    fitted.update(kappa=chosen["kappa"],routing_disabled=disabled,training_question_sha256=ids,
                  source_sha256=input_hashes,selection_seed=SEED,
                  development_scope="All 900 historical development questions; the old validation split is now development for this extension. GSM8K test outputs are never read.")
    final_routes,detail=predict(fitted,x,texts,embeddings=embeddings)
    result={"model":model,"n_development":900,"selected_hyperparameters":chosen,
            "routing_disabled_by_one_standard_error_rule":disabled,
            "final_fixed_baseline_gamma":fitted["baseline_gamma"],"cv_fixed":base_metrics,
            "nested_cross_validation":m,"nested_fixed":bm,
            "nested_delta_utility":float(delta.mean()),"nested_delta_utility_ci95":ci.tolist(),
            "outer_folds":outer,"all_candidates":evaluations,
            "training_route_counts":{f"{g:.1f}":int((final_routes==j).sum()) for j,g in enumerate(GAMMAS)},
            "interpretation":"Development screening only. Hyperparameters and baseline must be frozen before independent evaluation; no superiority is asserted from these selected development scores."}
    variant = "with_minilm" if use_embeddings else "without_minilm"
    output = ROOT / "analysis" / variant
    output.mkdir(parents=True, exist_ok=True)
    (output / (model + "-candidate.json")).write_text(json.dumps(fitted,indent=2)+"\n",encoding="utf-8",newline="\n")
    (output / (model + "-development.json")).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("all_candidates","outer_folds","selected_hyperparameters")}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",choices=("qwen2.5-3b","qwen2.5-7b"));p.add_argument("--use-embeddings",action="store_true");args=p.parse_args()
    for model in ([args.model] if args.model else ("qwen2.5-3b","qwen2.5-7b")):run(model,args.use_embeddings)
