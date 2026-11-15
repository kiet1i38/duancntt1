"""Frozen-policy paired cap and external-benchmark follow-up; new outputs only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
MATCHED = HERE.parent / 'matched-20261107'
BUNDLE = MATCHED / 'bundle'
sys.path.insert(0, str(BUNDLE))
spec = importlib.util.spec_from_file_location('matched_runner', MATCHED / 'runner.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
SVAMP_HASH = '5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e'
CONDITIONS = ('fixed_1.0', 'fixed_0.5', 'sd_frozen')
STOP = False


def halt(*_):
    global STOP
    STOP = True


def prepare_svamp(path):
    assert previous.digest(path) == SVAMP_HASH, 'Unexpected SVAMP data hash'
    raw = json.loads(path.read_text(encoding='utf8'))
    assert len(raw) == 1000 and len({r['ID'] for r in raw}) == 1000
    result = []
    for index, item in enumerate(raw):
        question = item['Body'].strip() + ' ' + item['Question'].strip()
        assert question.strip() and isinstance(item['Answer'], (int, float))
        result.append({'id': item['ID'], 'source_row_index': index,
                       'messages': [{'role': 'user', 'content': question}],
                       'answer': str(item['Answer'])})
    return result


def preflight():
    previous.bundle_check()
    rows = prepare_svamp(HERE / 'svamp-source.json')
    assert len(rows) == 1000
    lock = previous.read(HERE / 'gpu-protocol-lock.json')
    for name, path in {'gpu_runner.py': HERE / 'gpu_runner.py',
                       'analyze_gpu.py': HERE / 'analyze_gpu.py',
                       'setup_gpu.sh': HERE / 'setup_gpu.sh',
                       'run_gpu.sh': HERE / 'run_gpu.sh',
                       'svamp-source.json': HERE / 'svamp-source.json',
                       'bundle_manifest.json': BUNDLE / 'bundle_manifest.json',
                       'matched_runner.py': MATCHED / 'runner.py',
                       'archived_routes_3b.jsonl': MATCHED / 'archived_routes/qwen2.5-3b.jsonl',
                       'archived_routes_7b.jsonl': MATCHED / 'archived_routes/qwen2.5-7b.jsonl'}.items():
        assert previous.digest(path) == lock['sha256'][name], 'Protocol changed: ' + name
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('qwen2.5-3b', 'qwen2.5-7b'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-utc', type=float, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    svamp = preflight()
    assert args.batch_size == 8, 'The locked design uses batch size 8'
    if args.preflight_only:
        print(json.dumps({'event': 'preflight', 'model': args.model, 'svamp_rows': len(svamp),
                          'svamp_sha256': SVAMP_HASH, 'bundle_verified': True}))
        return
    if time.time() >= args.deadline_utc:
        raise SystemExit('Deadline has already passed')
    import torch
    import common
    import v3
    from data_processing.answer_extraction import extract_last_single_answer
    from eval.eval_script import eval_last_single_answer
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available()
    assert torch.cuda.get_device_properties(0).total_memory >= 23 * 1024 ** 3
    signal.signal(signal.SIGINT, halt)
    signal.signal(signal.SIGTERM, halt)
    common.set_seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    config = previous.read(BUNDLE / 'source_protocol.json')['models'][args.model]
    policy_path = BUNDLE / 'policies' / (args.model + '.json')
    policy = previous.read(policy_path)
    contract = {'schema': 1, 'model': args.model, 'model_revision': config,
                'runner_sha256': previous.digest(Path(__file__)),
                'bundle_manifest_sha256': previous.digest(BUNDLE / 'bundle_manifest.json'),
                'policy_sha256': previous.digest(policy_path), 'svamp_sha256': SVAMP_HASH,
                'caps': {'gsm8k': [512, 1024], 'svamp': [512]},
                'conditions': CONDITIONS, 'batch_size': args.batch_size,
                'seed': 42, 'dtype': 'float16', 'greedy': True, 'adapter_enabled': True,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0),
                'environment': {k: previous.importlib.metadata.version(k) for k in
                                ('transformers', 'peft', 'accelerate', 'numpy', 'sympy', 'regex', 'tokenizers', 'huggingface-hub')}}
    contract_hash = previous.stable(contract)
    out = args.output.resolve()
    assert not out.is_relative_to(MATCHED.resolve()), 'Cannot write into frozen matched experiment'
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        assert previous.read(manifest_path)['contract_sha256'] == contract_hash, 'Changed execution contract'
    previous.atomic(manifest_path, {'contract': contract, 'contract_sha256': contract_hash, 'status': 'loading'})
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'])
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['model_revision'],
        torch_dtype=torch.float16, device_map={'': 'cuda:0'}, attn_implementation='sdpa', low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, config['adapter_id'], revision=config['adapter_revision']).merge_and_unload()
    model.eval()
    assert all(p.device.type == 'cuda' and (not p.is_floating_point() or p.dtype == torch.float16) for p in model.parameters())
    gsm = common.prepare_gsm8k_rows(common.read_jsonl(BUNDLE / 'test.jsonl'), 'test')
    assert len(gsm) == 1319

    def route(rows, dataset):
        assignments = v3.build_v3_assignments(rows, tokenizer, common, policy['difficulty_spec'])
        gammas = {i: v3.v3_gamma_for_assignment(a, policy) for i, a in assignments.items()}
        if dataset == 'gsm8k':
            archive = {r['source_row_index']: r for r in common.read_jsonl(MATCHED / 'archived_routes' / (args.model + '.jsonl'))}
            assert set(archive) == set(assignments) == set(range(len(rows)))
            for i in archive:
                assert assignments[i]['question_sha256'] == archive[i]['question_sha256']
                assert gammas[i] == archive[i]['gamma']
        return assignments, gammas

    def generate(rows, gamma, cap, assignments):
        prompts = [common.build_prompt(r, tokenizer, gamma) for r in rows]
        encoded = tokenizer(prompts, padding='longest', return_tensors='pt', add_special_tokens=False)
        input_ids = encoded.input_ids.to('cuda')
        mask = encoded.attention_mask.to('cuda')
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            tokens = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=cap,
                do_sample=False, temperature=0.0, top_p=1.0, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        full = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        decoded_prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        eos = model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        output = []
        for j, (row, prompt, a, b) in enumerate(zip(rows, prompts, full, decoded_prompts)):
            text = a[len(b):]
            assert text.strip(), 'Empty generation'
            token_ids = tokens[j, input_ids.shape[1]:].tolist()
            stop = next((k for k, t in enumerate(token_ids) if t in eos), None)
            cot = text.split('\n\nThe final answer is:')[0]
            result = {'id': row['id'], 'source_row_index': row['source_row_index'],
                      'question_sha256': assignments[row['source_row_index']]['question_sha256'],
                      'prompt': prompt, 'gamma': gamma, 'model_output': text,
                      'prediction': extract_last_single_answer(common.question_text(row), text, task='cot'),
                      'answer': row['answer'], 'cot_length': int(tokenizer(cot, return_tensors='pt')['input_ids'].shape[1]),
                      'generated_token_count_excluding_eos': stop if stop is not None else len(token_ids),
                      'finish_reason': 'eos' if stop is not None else 'length'}
            result['accuracy'] = bool(result['prediction']) and bool(eval_last_single_answer(result))
            output.append(result)
        return output, elapsed

    gsm_assignments, gsm_gammas = route(gsm, 'gsm8k')
    svamp_assignments, svamp_gammas = route(svamp, 'svamp')
    # A three-question GSM pilot verifies the generation engine without using external outcomes.
    pilot_path = out / 'pilot.json'
    if not pilot_path.exists():
        old, _ = common.generate_group(model, tokenizer, gsm[:3], 0.8, 3, 512)
        fresh, seconds = generate(gsm[:3], 0.8, 512, gsm_assignments)
        parity = all(a['model_output'] == b['model_output'] and a['cot_length'] == b['cot_length']
                     and a['accuracy'] == b['accuracy'] for a, b in zip(old, fresh))
        assert parity, 'Engine parity pilot failed; no results were collected'
        previous.atomic(pilot_path, {'contract_sha256': contract_hash, 'parity': parity, 'n': 3, 'seconds': seconds})
    else:
        pilot = previous.read(pilot_path)
        assert pilot['contract_sha256'] == contract_hash and pilot['parity'] is True

    for dataset, rows, assignments, gammas, caps in (
        ('gsm8k', gsm, gsm_assignments, gsm_gammas, (512, 1024)),
        ('svamp', svamp, svamp_assignments, svamp_gammas, (512,))):
        for cap in caps:
            for condition in CONDITIONS:
                groups = ({g: [r for r in rows if gammas[r['source_row_index']] == g] for g in sorted(set(gammas.values()))}
                          if condition == 'sd_frozen' else {float(condition.removeprefix('fixed_')): rows})
                count = 0
                for gamma, group in groups.items():
                    for start in range(0, len(group), args.batch_size):
                        chosen = group[start:start + args.batch_size]
                        target = out / dataset / str(cap) / condition / f'gamma-{gamma:.1f}-{start:05d}.json'
                        if target.exists():
                            saved = previous.read(target)
                            assert saved['contract_sha256'] == contract_hash
                            assert [r['source_row_index'] for r in saved['rows']] == [r['source_row_index'] for r in chosen]
                            assert all(r['gamma'] == gamma and r['question_sha256'] == assignments[r['source_row_index']]['question_sha256'] for r in saved['rows'])
                        else:
                            if STOP or time.time() >= args.deadline_utc:
                                previous.atomic(manifest_path, {'contract': contract, 'contract_sha256': contract_hash, 'status': 'checkpointed'})
                                print(json.dumps({'event': 'checkpoint_stop', 'dataset': dataset, 'cap': cap, 'condition': condition}), flush=True)
                                return
                            torch.cuda.reset_peak_memory_stats()
                            generated, seconds = generate(chosen, gamma, cap, assignments)
                            previous.atomic(target, {'contract_sha256': contract_hash, 'rows': generated,
                                'generation_seconds': seconds, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
                        count += len(chosen)
                        print(json.dumps({'event': 'batch', 'dataset': dataset, 'cap': cap, 'condition': condition,
                                          'completed': count, 'total': len(rows)}), flush=True)
                assert count == len(rows)
    previous.atomic(manifest_path, {'contract': contract, 'contract_sha256': contract_hash, 'status': 'complete'})
    print(json.dumps({'event': 'complete', 'model': args.model}), flush=True)


if __name__ == '__main__':
    main()
