import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, REPORTS
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from scripts import score_caption_mismatches as mismatch_scores
(RES, MODELS) = (mismatch_scores.RES, mismatch_scores.MODELS)
OUT = RES / 'runs/combined_actions'
REPORT = ROOT / str(REPORTS / 'combined_actions')
REFERENCE = RES / 'runs/caption_actions/actions'
(load, dump, rows, stamp) = (mismatch_scores.load, mismatch_scores.dump, mismatch_scores.rows, mismatch_scores.stamp)
yes_probability = mismatch_scores.yes_probability
TF = 0.05960534898891063

def cfg():
    return load(REPORT / 'settings.json')

def writel(path, values):
    with Path(path).open('x') as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

def key(row):
    return tuple(sorted((row['record_id_a'], row['record_id_b'])))

def combined_scores(clip, probabilities, validation_ids):
    ids = sorted(probabilities)
    cp = mismatch_scores.percentile([clip[r] for r in validation_ids], [clip[r] for r in ids])
    valid_reference = [probabilities[r]['probability'] for r in validation_ids if probabilities[r]['valid']]
    pp = mismatch_scores.percentile(valid_reference, [mismatch_scores.effective_probability(probabilities[r]['probability']) for r in ids])
    return {r: float((a + (b if probabilities[r]['valid'] else 1.0)) / 2) for (r, a, b) in zip(ids, cp, pp, strict=True)}

@lru_cache(maxsize=1)
def retained_set_module():
    spec = importlib.util.spec_from_file_location('retained_set_helpers', REPO_ROOT / 'scripts/retained_set_helpers.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def classify_pair(pair, scores, decision, threshold=TF):
    if min((scores[r] for r in pair)) < threshold:
        return 'score_mismatch'
    if decision == 'mismatch':
        return 'vlm_mismatch_review'
    return decision

def pair_action(pair, scores, decision, threshold=TF):
    action = classify_pair(pair, scores, decision, threshold)
    return 'mismatch' if action in ('score_mismatch', 'vlm_mismatch_review') else action

def export(ids, clip, combined, decisions):
    representative = min(ids, key=lambda r: (-clip[r], r))
    actions = {p: pair_action(p, combined, d) for (p, d) in decisions.items()}
    return (representative, retained_set_module().export_cluster(ids, representative, actions))

def inputs():
    (np, pq, _) = mismatch_scores.cpu_modules()
    c = cfg()['inputs']
    manifest = pq.read_table(c['component_record_order'].split(':')[0], columns=['record_id']).column(0).to_pylist()
    assert manifest == sorted(manifest)
    components = np.load(c['components'])
    parts = {r['record_id']: r['partition'] for r in rows(Path(c['dataset']) / 'private/lineage.jsonl')}
    groups = defaultdict(list)
    for (rid, component) in zip(manifest, components, strict=True):
        groups[int(component)].append(rid)
    assert len(manifest) == len(parts) == 50000
    assert all((len({parts[r] for r in g}) == 1 for g in groups.values()))
    meta = pq.read_table(Path(c['clip']) / 'metadata.parquet', columns=['record_id', 'row_index']).to_pydict()
    values = np.load(Path(c['clip']) / 'clipscore.npy')
    clip = {r: float(values[i]) for (r, i) in zip(meta['record_id'], meta['row_index'], strict=True)}
    return (parts, groups, clip)

def action_table(part, combined, clip):
    (np, _, action_checks) = mismatch_scores.cpu_modules()
    population = list(rows(REFERENCE / f'actions_{part}.jsonl'))
    (selected, y, excluded) = mismatch_scores.labeled(part, population)
    assert not excluded
    vlm = [r['parsed']['decision'] == 'mismatch' for r in selected]
    recall = sum((a and b for (a, b) in zip(vlm, y))) / sum(y)
    f = mismatch_scores.pair_scores(selected, combined)
    cp = mismatch_scores.pair_scores(selected, clip)
    tclip = load(ROOT / str(REPORTS / 'mismatch_scores/thresholds.json'))['thresholds']['clipscore']['threshold']
    scores = dict(combined=(f, TF), vlm=([0.0 if v else 1.0 for v in vlm], 0.5), combined_or_vlm=([0.0 if s < TF or v else 1.0 for (s, v) in zip(f, vlm)], 0.5), clipscore=(cp, tclip))
    (result, curves) = ({}, {})
    for (name, (s, threshold)) in scores.items():
        m = mismatch_scores.confusion(s, y, threshold)
        curve = action_checks.pr_curve(s, y)
        m.update(AUPRC=sum(((b['recall'] - a['recall']) * b['precision'] for (a, b) in zip(curve, curve[1:]))), precision_at_vlm_recall=max((p['precision'] for p in curve if p['recall'] >= recall)), vlm_recall_target=recall, threshold=threshold, below_t_f=sum((v < TF for v in mismatch_scores.pair_scores(population, combined))), below_own_threshold=sum((v < tclip for v in mismatch_scores.pair_scores(population, clip))) if name == 'clipscore' else None, score_type='binary' if name in ('vlm', 'combined_or_vlm') else 'continuous')
        (result[name], curves[name]) = (m, curve)
    dump(OUT / f'{part}_PR_curves.json', curves)
    return result

def gpu_worker():
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['TRITON_CACHE_DIR'] = str(OUT / 'cache/triton')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(OUT / 'cache/torchinductor')
    from importlib.metadata import version
    start = time.monotonic()
    target = OUT / 'qwen25_vllm.jsonl'
    receipt_path = OUT / 'qwen25_vllm_session.json'
    records = list(rows(OUT / 'records.jsonl'))
    c = cfg()['inference']
    model_path = str(RES / 'models' / MODELS['qwen25'])
    receipt = dict(started_utc=stamp(), backend='vllm', model=model_path, total_records=len(records), completed=0, invalid=0, runtime_failed=0, inference_seconds=0.0, versions={k: version(k) for k in ('torch', 'transformers')})
    dump(receipt_path, receipt)
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor
        torch.set_num_threads(4)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, min_pixels=c['min_pixels'], max_pixels=c['max_pixels'])
        from vllm import LLM, SamplingParams
        receipt['versions']['vllm'] = version('vllm')
        assert version('vllm') == '0.11.0'
        model = LLM(model=model_path, dtype=c['dtype'], max_model_len=c['max_model_len'], max_num_seqs=c['max_num_seqs'], max_num_batched_tokens=c['max_num_batched_tokens'], enforce_eager=c['enforce_eager'], enable_chunked_prefill=c['enable_chunked_prefill'], limit_mm_per_prompt=c['limit_mm_per_prompt'], mm_processor_kwargs={'min_pixels': c['min_pixels'], 'max_pixels': c['max_pixels']}, gpu_memory_utilization=c['gpu_memory_utilization'], disable_log_stats=True)
        settings = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
        receipt['engine_load_seconds'] = time.monotonic() - start
        batch_size = 16
        with target.open('x') as stream:
            for offset in range(0, len(records), batch_size):
                block = records[offset:offset + batch_size]
                (pictures, inputs, prompts) = ([], [], [])
                try:
                    for record in block:
                        picture = Image.open(record['image_path']).convert('RGB')
                        pictures.append(picture)
                        text = cfg()['prompt'].format(caption=record['caption'])
                        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': record['image_path']}, {'type': 'text', 'text': text}]}]
                        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        prompts.append(prompt)
                        inputs.append(dict(prompt=prompt, multi_modal_data={'image': [picture]}))
                    begin = time.monotonic()
                    outputs = model.generate(inputs, settings, use_tqdm=False)
                    token_lists = []
                    for output in outputs:
                        first = output.outputs[0].logprobs[0]
                        token_lists.append([dict(token_id=int(tid), text=processor.tokenizer.decode([tid]), logprob=float(lp.logprob), rank=int(lp.rank)) for (tid, lp) in first.items() if lp.rank <= 20])
                    receipt['inference_seconds'] += time.monotonic() - begin
                    assert len(token_lists) == len(block)
                    for (record, tokens) in zip(block, token_lists, strict=True):
                        score = yes_probability(tokens)
                        stream.write(json.dumps(dict(record_id=record['record_id'], partition=record['partition'], tokens=tokens, **score)) + '\n')
                        receipt['completed'] += 1
                        receipt['invalid'] += not score['valid']
                    stream.flush()
                    os.fsync(stream.fileno())
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    for record in block:
                        stream.write(json.dumps(dict(record_id=record['record_id'], partition=record['partition'], tokens=[], probability=None, valid=False, yes_mass=0.0, no_mass=0.0, runtime_error=error)) + '\n')
                        receipt['completed'] += 1
                        receipt['invalid'] += 1
                        receipt['runtime_failed'] += 1
                    stream.flush()
                    os.fsync(stream.fileno())
                finally:
                    for picture in pictures:
                        picture.close()
                elapsed = time.monotonic() - start
                receipt['session_seconds'] = elapsed
                dump(receipt_path, receipt)
    except BaseException as exc:
        receipt.update(error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt.update(finished_utc=stamp(), session_seconds=time.monotonic() - start)
        receipt['seconds_per_completed_record'] = receipt['session_seconds'] / receipt['completed'] if receipt['completed'] else None
        dump(receipt_path, receipt)

def export_partition(part, groups, parts, clip, combined, save=False):
    actions = list(rows(REFERENCE / f'actions_{part}.jsonl'))
    decisions = defaultdict(dict)
    for r in actions:
        decisions[r['component']][key(r)] = r['parsed']['decision']
    (totals, exports) = (Counter(), [])
    for (component, ids) in groups.items():
        if parts[ids[0]] != part:
            continue
        (representative, result) = export(ids, clip, combined, decisions[component])
        result = result['b']
        for (state, records) in result.items():
            totals[state] += len(records)
        exports.append(dict(component=component, partition=part, record_ids=ids, representative_id=representative, **result))
    assert sum(totals.values()) == sum((p == part for p in parts.values()))
    if save:
        writel(OUT / f'exports_{part}.jsonl', exports)
    return dict(record_counts=dict(totals), U=len(actions), below_t_f=sum((s < TF for s in mismatch_scores.pair_scores(actions, combined))))

def prepare():
    (parts, groups, clip) = inputs()
    records = {r['record_id']: r for r in rows(Path(cfg()['inputs']['dataset']) / 'public_records.jsonl')}
    selected = sorted((r for members in groups.values() if len(members) >= 2 for r in members if parts[r] == 'train'))
    OUT.mkdir(parents=True, exist_ok=True)
    writel(OUT / 'records.jsonl', [dict(record_id=r, partition=parts[r], caption=records[r]['text'], image_path=records[r]['images'][0]['image_path']) for r in selected])

def finish():
    (parts, groups, clip) = inputs()
    previous = list(rows(RES / 'runs/mismatch_scores/qwen25_vllm.jsonl'))
    training = list(rows(OUT / 'qwen25_vllm.jsonl'))
    probabilities = {r['record_id']: r for r in previous + training}
    assert len(probabilities) == len(previous) + len(training)
    validation = sorted((r['record_id'] for r in previous if r['partition'] == 'validation'))
    combined = combined_scores(clip, probabilities, validation)
    dump(OUT / 'record_scores.json', {r: dict(partition=parts[r], clipscore=clip[r], probability=probabilities[r]['probability'], valid=probabilities[r]['valid'], combined=combined[r]) for r in combined})
    exports = {part: export_partition(part, groups, parts, clip, combined, save=True) for part in ('train', 'validation', 'test')}
    total = {k: sum((e['record_counts'][k] for e in exports.values())) for k in ('retained', 'held', 'deleted')}
    assert sum(total.values()) == len(parts)
    for part in ('train', 'validation', 'test'):
        effective = []
        for row in rows(REFERENCE / f'actions_{part}.jsonl'):
            score = min((combined[r] for r in key(row)))
            decision = row['parsed']['decision']
            effective.append(dict(pair_id=row['pair_id'], record_id_a=row['record_id_a'], record_id_b=row['record_id_b'], partition=part, component=row['component'], fusion_score=score, vlm_decision=decision, action=classify_pair(key(row), combined, decision), effective_action=pair_action(key(row), combined, decision)))
        writel(OUT / f'actions_{part}.jsonl', effective)
    tables = {part: action_table(part, combined, clip) for part in ('validation', 'test')}
    result = dict(threshold=TF, tables=tables, exports=exports, total_exports=total)
    dump(REPORT / 'results.json', result)
    return result
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'infer', 'finish'])
    a = p.parse_args()
    if a.action == 'infer':
        gpu_worker()
    else:
        globals()[a.action]()
