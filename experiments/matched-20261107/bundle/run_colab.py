"""Fresh matched controls. Frozen V3; no fitting or selection from test outcomes."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def emit(**value):
    print(json.dumps(value, ensure_ascii=False), flush=True)

def analyze(out, expected):
    import numpy as np
    by_method = {}
    for method_dir in sorted((out / 'batches').glob('*')):
        rows = [r for path in sorted(method_dir.glob('*.json')) for r in read(path)['rows']]
        by_method[method_dir.name] = sorted(rows, key=lambda r:r['source_row_index'])
    summaries = {}
    for name, rows in by_method.items():
        assert len({r['source_row_index'] for r in rows}) == len(rows)
        summaries[name] = {'n':len(rows), 'complete':len(rows)==expected,
            'correct':sum(r['accuracy'] for r in rows),
            'accuracy_percent':100 * sum(r['accuracy'] for r in rows) / len(rows) if rows else None,
            'mean_cot_tokens':sum(r['cot_length'] for r in rows) / len(rows) if rows else None,
            'gamma_counts':dict(Counter(str(r['gamma']) for r in rows))}
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    paired = {}
    v3rows = by_method.get('v3', [])
    if len(v3rows) == expected:
        va = np.array([r['accuracy'] for r in v3rows], dtype=float)
        vt = np.array([r['cot_length'] for r in v3rows], dtype=float)
        for name, rows in by_method.items():
            if name == 'v3' or len(rows) != expected: continue
            assert [r['source_row_index'] for r in rows] == [r['source_row_index'] for r in v3rows]
            assert [r['question_sha256'] for r in rows] == [r['question_sha256'] for r in v3rows]
            ba = np.array([r['accuracy'] for r in rows], dtype=float)
            bt = np.array([r['cot_length'] for r in rows], dtype=float)
            rng = np.random.default_rng(4242)
            da, dt = [], []
            for start in range(0, 20000, 200):
                ix = rng.integers(0, expected, size=(min(200,20000-start),expected))
                da.extend(100*(va[ix]-ba[ix]).mean(axis=1))
                dt.extend(100*(1-vt[ix].mean(axis=1)/bt[ix].mean(axis=1)))
            paired[name] = {'accuracy_delta_pp':float(100*(va-ba).mean()),
                'accuracy_delta_95ci':np.percentile(da,[2.5,97.5]).tolist(),
                'cot_token_reduction_percent':float(100*(1-vt.mean()/bt.mean())),
                'token_reduction_95ci':np.percentile(dt,[2.5,97.5]).tolist(),
                'bootstrap_resamples':20000,'bootstrap_seed':4242}
    result = {'expected_per_method':expected,'methods':summaries,'v3_vs_control_paired':paired,
              'interpretation':'Descriptive paired comparisons on a reused benchmark; pointwise CIs are not adjusted for multiple comparisons. Random control uses one frozen permutation, not a multi-seed estimate. Pilot is not a quality result. Timing is batched wall time, not single-request latency.'}
    atomic(out/'analysis.json', result)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['qwen2.5-3b','qwen2.5-7b'], default='qwen2.5-3b')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--pilot-rows',type=int,default=8)
    p.add_argument('--all-fixed',action='store_true')
    p.add_argument('--analyze-only',action='store_true')
    args=p.parse_args()
    assert args.batch_size > 0 and 0 < args.pilot_rows <= 1319
    if args.analyze_only:
        emit(analysis=analyze(args.output,1319)); return
    import torch
    import common
    import v3
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    for name, expected_hash in read(ROOT/'bundle_manifest.json').items():
        assert digest(ROOT/name)==expected_hash, f'Bundle integrity mismatch: {name}'
    assert torch.cuda.is_available(), 'CUDA GPU required; do not run inference on CPU'
    common.set_seed(42)
    config=read(ROOT/'source_protocol.json')['models'][args.model]
    policy=read(ROOT/f'policies/{args.model}.json')
    selection=read(ROOT/'fixed_selection.json')[args.model]
    raw=common.read_jsonl(ROOT/'test.jsonl')
    assert len(raw)==1319 and digest(ROOT/'test.jsonl')==v3.TEST_SHA256
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    environment={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                 'gpu':torch.cuda.get_device_name(0),
                 'packages':{k:importlib.metadata.version(k) for k in
                             ['transformers','peft','accelerate','numpy','sympy','regex','tokenizers','huggingface-hub']}}
    contract={'model':config,'model_name':args.model,'dtype':'float16','seed':42,
        'max_new_tokens':512,'decoding':'greedy','adapter_on_all_methods':True,
        'batch_size':args.batch_size,'fixed_selection':selection,
        'v3_policy_sha256':digest(ROOT/f'policies/{args.model}.json'),
        'bundle_manifest_sha256':digest(ROOT/'bundle_manifest.json'),
        'all_fixed':args.all_fixed,'random_router':'permutation of question-only V3 gamma assignments, seed 4242',
        'test_usage':'GSM8K reused benchmark; no policy change from test results',
        'generation_engine':'unchanged original common.generate_group and upstream eval.utils',
        'accuracy_gate':None,'latency_claim':False,'environment':environment}
    contract_hash=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        assert read(manifest_path)['contract_sha256']==contract_hash, 'Cannot resume with a changed contract'
    manifest={'contract':contract,'contract_sha256':contract_hash,'status':'loading_model',
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'python':platform.python_version(),'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'gpu_vram_bytes':torch.cuda.get_device_properties(0).total_memory,
        'packages':{k:importlib.metadata.version(k) for k in ['transformers','peft','accelerate','numpy']}}
    atomic(manifest_path,manifest)
    atomic(out/'attempts'/f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}.json',manifest)
    emit(event='loading_model',model=config['model_id'],gpu=manifest['gpu'])
    tokenizer=AutoTokenizer.from_pretrained(config['model_id'],revision=config['model_revision'])
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(config['model_id'],revision=config['model_revision'],
        torch_dtype=torch.float16,device_map={'':'cuda:0'},low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,config['adapter_id'],revision=config['adapter_revision'])
    model=model.merge_and_unload(); model.eval()
    assert all(x.device.type=='cuda' for x in model.parameters()), 'CPU offload is outside this protocol'
    assert all(x.dtype==torch.float16 for x in model.parameters() if x.is_floating_point())
    rows=common.prepare_gsm8k_rows(raw,'test')
    started=time.perf_counter()
    assignments=v3.build_v3_assignments(rows,tokenizer,common,policy['difficulty_spec'])
    routing_seconds=time.perf_counter()-started
    gammas=[v3.v3_gamma_for_assignment(assignments[i],policy) for i in range(1319)]
    random_gammas=gammas.copy(); random.Random(4242).shuffle(random_gammas)
    methods={'fixed_1.0':[1.0]*1319,'fixed_best':[selection['selected_gamma']]*1319,
             'random_4242':random_gammas,'v3':gammas}
    if args.all_fixed:
        for k in range(5,10):
            if k/10 != selection['selected_gamma']:
                methods[f'fixed_{k/10:.1f}']=[k/10]*1319
    frozen=[{'source_row_index':i,'question_sha256':assignments[i]['question_sha256'],
             'v3_features':assignments[i], 'gammas':{m:g[i] for m,g in methods.items()}} for i in range(1319)]
    atomic(out/'frozen_assignments.json',frozen)
    manifest.update(status='pilot',routing_seconds=routing_seconds,methods=list(methods),
                    allocated_model_vram_bytes=torch.cuda.memory_allocated())
    atomic(manifest_path,manifest)
    torch.cuda.synchronize()
    common.generate_group(model,tokenizer,[rows[0]],1.0,1,8)
    emit(event='warmup_complete',allocated_gb=torch.cuda.memory_allocated()/1e9)
    for stage, count in [('pilot',args.pilot_rows),('full',1319)]:
        stage_dir=out/stage
        manifest['status']=stage; atomic(manifest_path,manifest)
        started_stage=time.perf_counter()
        for offset in range(0,count,args.batch_size):
            indices=list(range(offset,min(offset+args.batch_size,count)))
            # Rotate condition order to reduce systematic time/order bias.
            names=list(methods); rotate=(offset//args.batch_size)%len(names)
            for name in names[rotate:]+names[:rotate]:
                target=stage_dir/'batches'/name/f'{offset:05d}.json'
                if target.exists():
                    saved=read(target)
                    assert saved['contract_sha256']==contract_hash
                    assert [r['source_row_index'] for r in saved['rows']]==indices
                    for record in saved['rows']:
                        i=record['source_row_index']
                        assert record['question_sha256']==assignments[i]['question_sha256']
                        assert record['gamma']==methods[name][i] and record['model_output'].strip()
                    continue
                selected=[]; torch.cuda.synchronize(); start=time.perf_counter()
                for gamma in sorted({methods[name][i] for i in indices}):
                    batch=[rows[i] for i in indices if methods[name][i]==gamma]
                    result,_=common.generate_group(model,tokenizer,batch,gamma,args.batch_size,512)
                    for record in result:
                        i=record['source_row_index']; record['question_sha256']=assignments[i]['question_sha256']
                        record['method']=name; record['stage']=stage
                    selected.extend(result)
                torch.cuda.synchronize(); elapsed=time.perf_counter()-start
                selected.sort(key=lambda r:r['source_row_index'])
                assert [r['source_row_index'] for r in selected]==indices
                atomic(target,{'contract_sha256':contract_hash,'batch_wall_seconds':elapsed,
                    'peak_vram_bytes':torch.cuda.max_memory_allocated(),'rows':selected})
                emit(event='batch_complete',stage=stage,method=name,completed_through=indices[-1]+1,
                     total_per_method=count,batch_seconds=round(elapsed,2))
            atomic(out/'progress.json',{'stage':stage,'completed_per_method':indices[-1]+1,
                'expected_per_method':count,'updated_utc':datetime.now(timezone.utc).isoformat()})
        result=analyze(stage_dir,count)
        emit(event=f'{stage}_complete',elapsed_seconds=round(time.perf_counter()-started_stage,2),
             results=result)
        if stage=='pilot': emit(event='starting_full_test',note='Policy and protocol unchanged after pilot; pilot outcomes do not select methods')
    manifest['status']='complete'; manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
    atomic(manifest_path,manifest)
    emit(event='experiment_complete',output=str(out))

if __name__=='__main__':
    main()
