"""Predict paired utility changes and revert to a development-selected fixed ratio.

No LLM, gold answer, rationale, or generated text is consulted by predict().
The uncertainty gate is a model-based heuristic, not a coverage guarantee.
"""
import hashlib
import re
import numpy as np

FEATURES=("question_token_count","numeric_span_count","operator_count",
          "clause_connector_count","numeric_unit_pair_count","relation_marker_count")
GAMMAS=np.array([.5,.6,.7,.8,.9,1.])
WORD=re.compile(r"[a-z]+")

def lexical(texts,dim=128):
    x=np.zeros((len(texts),dim))
    for i,text in enumerate(texts):
        words=WORD.findall(text.lower())
        terms=words+[a+" "+b for a,b in zip(words,words[1:])]
        for term in terms:
            h=int.from_bytes(hashlib.blake2b(term.encode(),digest_size=8).digest(),"little")
            x[i,h%dim]+=1 if (h>>8)&1 else -1
        norm=np.linalg.norm(x[i])
        if norm:x[i]*=np.sqrt(dim)/norm
    return x

def matrix(raw,texts,mode,center,scale):
    z=(np.log1p(raw)-center)/scale
    z=np.clip(z,-4,4)
    if mode=="quadratic":
        z=np.column_stack([z]+[z[:,i]*z[:,j] for i in range(6) for j in range(i,6)])
    elif mode=="lexical":z=np.column_stack([z,lexical(texts)])
    elif mode!="linear":raise ValueError(mode)
    return z

def project_embeddings(embeddings):
    e=np.asarray(embeddings,dtype=float)
    dim=e.shape[1]
    signs=np.random.default_rng(20261107).integers(0,2,size=(dim,64),dtype=np.int8)*2-1
    e=e/np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
    return e@signs.astype(float)

def fixed_index(y,t,lam=.1):
    score=100*y.mean(0)-100*lam*t.mean(0)/t[:,-1].mean()
    # Higher gamma wins numerical ties.
    return max(range(6),key=lambda j:(round(float(score[j]),10),float(y[:,j].mean()),j))

def fit(raw,texts,y,t,mode="linear",alpha=100.,lam=.1,embeddings=None):
    raw=np.asarray(raw,float);y=np.asarray(y,float);t=np.asarray(t,float)
    center=np.log1p(raw).mean(0);scale=np.log1p(raw).std(0);scale[scale<1e-12]=1
    z=matrix(raw,texts,"linear" if mode=="semantic" else mode,center,scale)
    embedding_stats={}
    if mode=="semantic":
        assert embeddings is not None
        e=project_embeddings(embeddings);em=e.mean(0);es=e.std(0);es[es<1e-12]=1
        z=np.column_stack([z,(e-em)/es])
        embedding_stats={"embedding_dimension":int(np.asarray(embeddings).shape[1]),
                         "embedding_center":em.tolist(),"embedding_scale":es.tolist(),
                         "embedding_projection_seed":20261107,"embedding_projection_width":64}
    zmean=z.mean(0);xc=z-zmean
    baseline=fixed_index(y,t,lam)
    target=100*(y-y[:,baseline,None])-100*lam*(t-t[:,baseline,None])/t[:,-1].mean()
    offset=target.mean(0);yc=target-offset
    gram=xc.T@xc;inverse=np.linalg.inv(gram+alpha*np.eye(gram.shape[0]))
    coef=inverse@xc.T@yc
    fitted=xc@coef+offset
    df=float(np.trace(gram@inverse))+1
    sigma2=((target-fitted)**2).sum(0)/max(len(raw)-df,1)
    covariance=inverse@gram@inverse
    return {"schema":"paired-utility-ridge-v1","mode":mode,"alpha":alpha,"lambda":lam,**embedding_stats,
            "baseline_index":baseline,"baseline_gamma":float(GAMMAS[baseline]),"n_train":len(raw),
            "features":list(FEATURES),"center":center.tolist(),"scale":scale.tolist(),
            "zmean":zmean.tolist(),"coef":coef.tolist(),"offset":offset.tolist(),
            "sigma2":sigma2.tolist(),"covariance":covariance.tolist(),
            "ood_z_threshold":6.0,"feature_clip":4.0,"lexical_dim":128}

def predict(model,raw,texts,kappa=None,embeddings=None):
    raw=np.asarray(raw,float)
    center=np.array(model["center"]);scale=np.array(model["scale"])
    mode=model["mode"]
    z=matrix(raw,texts,"linear" if mode=="semantic" else mode,center,scale)
    if mode=="semantic":
        assert embeddings is not None and np.asarray(embeddings).shape[1]==model["embedding_dimension"]
        e=project_embeddings(embeddings)
        z=np.column_stack([z,(e-np.array(model["embedding_center"]))/np.array(model["embedding_scale"])])
    z=z-np.array(model["zmean"])
    mean=z@np.array(model["coef"])+np.array(model["offset"])
    leverage=1/model["n_train"]+np.einsum("ij,jk,ik->i",z,np.array(model["covariance"]),z)
    se=np.sqrt(np.maximum(leverage[:,None]*np.array(model["sigma2"])[None,:],0))
    kappa=model.get("kappa",0.) if kappa is None else kappa
    score=mean-kappa*se
    baseline=model["baseline_index"]
    score[:,baseline]=0
    ood=(np.abs((np.log1p(raw)-center)/scale)>model["ood_z_threshold"]).any(1)
    score[ood]=-np.inf;score[ood,baseline]=0
    # Reverse order gives the higher gamma on exact ties.
    route=5-np.argmax(score[:,::-1],axis=1)
    route[score[np.arange(len(route)), route] <= 0] = baseline
    if model.get("routing_disabled",False):route[:]=baseline
    return route,{"ood":ood,"predicted_gain":mean[np.arange(len(route)),route],
                  "standard_error":se[np.arange(len(route)),route]}
