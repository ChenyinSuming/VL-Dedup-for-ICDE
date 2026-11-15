import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
RES = RESOURCES
OUT = RES / 'runs/mismatch_scores'
REPORT = ROOT / str(REPORTS / 'mismatch_scores')
MODELS = {'qwen25': 'Qwen2.5-VL-7B-Instruct'}

def stamp():
    return datetime.now(timezone.utc).isoformat()

def load(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)

def cfg():
    settings = load(REPORT / 'settings.json')
    settings.setdefault('baseline_record_threshold', 0.14425230026245117)
    return settings

def cpu_modules():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    from scripts import run_action_checks as action_checks
    return (np, pq, action_checks)

def yes_probability(tokens):
    masses = {'yes': 0.0, 'no': 0.0}
    for token in tokens:
        if token['rank'] > 20:
            continue
        label = token['text'].strip().casefold()
        if label in masses:
            masses[label] += math.exp(token['logprob'])
    denominator = sum(masses.values())
    return dict(probability=masses['yes'] / denominator if denominator else None, valid=bool(denominator), yes_mass=masses['yes'], no_mass=masses['no'])

def effective_probability(value):
    return 1.0 if value is None else value

def pair_scores(population, scores):
    return [min(scores[row['record_id_a']], scores[row['record_id_b']]) for row in population]

def component_scores(clip, images, texts, simcse):
    import numpy as np
    clip = np.asarray(clip, dtype=np.float64)
    (images, texts, simcse) = [np.asarray(a, dtype=np.float64) for a in (images, texts, simcse)]
    n = len(clip)
    assert n >= 2
    relative = clip - np.median(clip)
    cross = texts @ images.mean(axis=0)
    text = (simcse @ simcse.sum(axis=0) - np.sum(simcse * simcse, axis=1)) / (n - 1)
    return (relative, cross, text)

def percentile(reference, values):
    import numpy as np
    reference = np.sort(np.asarray(reference, dtype=float))
    assert len(reference)
    return (np.searchsorted(reference, values, side='left') + np.searchsorted(reference, values, side='right')) / (2 * len(reference))

def labeled(partition, population):
    (_, _, action_checks) = cpu_modules()
    refs = {tuple(sorted((r['record_id_a'], r['record_id_b']))): r['grade'] for r in action_checks.reference_rows(cfg()['inputs']['dataset'], partition)}
    (selected, positive, excluded) = ([], [], Counter())
    for row in population:
        grade = refs.get(tuple(sorted((row['record_id_a'], row['record_id_b']))))
        if grade in ('L0', 'L2', 'L3', 'M'):
            selected.append(row)
            positive.append(grade == 'M')
        elif grade is not None:
            excluded[grade] += 1
    return (selected, positive, dict(excluded))

def confusion(scores, positive, threshold):
    (np, _, action_checks) = cpu_modules()
    (pred, y) = (np.asarray(scores) < threshold, np.asarray(positive, dtype=bool))
    return action_checks.metrics((pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum(), (~pred & ~y).sum())

def check_counts(observed, expected):
    assert [observed[k] for k in ('TP', 'FP', 'FN', 'TN')] == expected, (observed, expected)

def prepare():
    (np, pq, action_checks) = cpu_modules()
    c = cfg()['inputs']
    data = Path(c['dataset'])
    parts = {r['record_id']: r['partition'] for r in rows(data / 'private/lineage.jsonl')}
    manifest = pq.read_table(c['component_record_order'].split(':')[0]).to_pandas().sort_values('record_id')
    ids = manifest.record_id.tolist()
    components = np.load(c['components'])
    assert len(ids) == len(set(ids)) == len(components) == len(parts) == 50000
    groups = defaultdict(list)
    for (rid, comp) in zip(ids, components, strict=True):
        groups[int(comp)].append(rid)
    assert all((len({parts[r] for r in group}) == 1 for group in groups.values()))
    groups = {k: v for (k, v) in groups.items() if len(v) >= 2 and parts[v[0]] in ('validation', 'test')}
    scoring_ids = sorted((rid for group in groups.values() for rid in group))
    records = {r['record_id']: r for r in rows(data / 'public_records.jsonl')}
    for row in manifest.itertuples():
        assert records[row.record_id]['text'] == row.text
    clipdir = Path(c['clip'])
    meta = pq.read_table(clipdir / 'metadata.parquet').to_pydict()
    index = dict(zip(meta['record_id'], meta['row_index'], strict=True))
    clip = np.load(clipdir / 'clipscore.npy', mmap_mode='r')
    image = np.load(clipdir / 'image_emb.npy', mmap_mode='r')
    text = np.load(clipdir / 'text_emb.npy', mmap_mode='r')
    assert set(index) == set(ids) and sorted(index.values()) == list(range(50000))
    assert np.isfinite(clip).all()
    simpath = Path(c['simcse'])
    if simpath.exists() and Path(c['simcse_mapping']).exists():
        simcse = np.load(simpath, mmap_mode='r')
        simids = pq.read_table(c['simcse_mapping'], columns=['record_id']).column(0).to_pylist()
        simindex = {r: i for (i, r) in enumerate(simids)}
        assert len(simids) == len(set(simids)) == len(simcse) == 50000 and set(simids) == set(ids)
    else:
        (simcse, simindex) = (text, index)
    result = {r: dict(clipscore=float(clip[index[r]])) for r in scoring_ids}
    for group in groups.values():
        (ix, si) = ([index[r] for r in group], [simindex[r] for r in group])
        arrays = component_scores(clip[ix], image[ix], text[ix], simcse[si])
        for (name, values) in zip(('relative', 'cross_image_mean', 'text_only'), arrays, strict=True):
            for (rid, value) in zip(group, values, strict=True):
                assert np.isfinite(value)
                result[rid][name] = float(value)
    population = [r for r in rows(c['pairs']) if r['partition'] in ('validation', 'test')]
    assert len({r['pair_id'] for r in population}) == len(population)
    compmap = dict(zip(ids, map(int, components), strict=True))
    expected = set()
    for (comp, group) in groups.items():
        for (i, a) in enumerate(group):
            for b in group[i + 1:]:
                expected.add(tuple(sorted((a, b))))
    assert expected == {tuple(sorted((r['record_id_a'], r['record_id_b']))) for r in population}
    for r in population:
        (a, b) = (r['record_id_a'], r['record_id_b'])
        assert parts[a] == parts[b] == r['partition'] and compmap[a] == compmap[b] == r['component']
    for part in ('validation', 'test'):
        u = [r for r in population if r['partition'] == part]
        assert len(u) == cfg()['population'][part]['U_new']
    with (OUT / 'records.jsonl').open('x') as stream:
        for rid in scoring_ids:
            record = records[rid]
            stream.write(json.dumps(dict(record_id=rid, partition=parts[rid], component=compmap[rid], caption=record['text'], image_path=record['images'][0]['image_path']), ensure_ascii=False) + '\n')
    dump(OUT / 'cpu_scores.json', result)

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

def all_scores():
    result = load(OUT / 'cpu_scores.json')
    records = list(rows(OUT / 'records.jsonl'))
    ids = [r['record_id'] for r in records]
    (available, unavailable) = ([], {})
    for model_key in MODELS:
        path = OUT / f'{model_key}_vllm.jsonl'
        if not path.exists():
            unavailable[model_key] = None
            continue
        probabilities = list(rows(path))
        assert [r['record_id'] for r in probabilities] == ids
        for row in probabilities:
            result[row['record_id']][model_key] = effective_probability(row['probability'])
        available.append(model_key)
    if 'qwen25' in available:
        valid = {r['record_id']: r['valid'] for r in rows(OUT / 'qwen25_vllm.jsonl')}
        validation = [r['record_id'] for r in records if r['partition'] == 'validation']
        clip = percentile([result[r]['clipscore'] for r in validation], [result[r]['clipscore'] for r in ids])
        prob = percentile([result[r]['qwen25'] for r in validation if valid[r]], [result[r]['qwen25'] for r in ids])
        for (rid, a, b) in zip(ids, clip, prob, strict=True):
            result[rid]['fusion'] = float((a + (b if valid[rid] else 1.0)) / 2)
    return (result, unavailable)

def freeze():
    (_, _, action_checks) = cpu_modules()
    (scores, unavailable) = all_scores()
    population = [r for r in rows(cfg()['inputs']['pairs']) if r['partition'] == 'validation']
    (selected, y, excluded) = labeled('validation', population)
    assert sum(y) == 389 and len(y) - sum(y) == 8950 and (not excluded)
    thresholds = {}
    for name in next(iter(scores.values())):
        values = {rid: score[name] for (rid, score) in scores.items()}
        thresholds[name] = action_checks.select_threshold(pair_scores(selected, values), y)
        thresholds[name]['U_validation_below'] = sum((s < thresholds[name]['threshold'] for s in pair_scores(population, values)))
    dump(OUT / 'all_scores.json', scores)
    receipt = dict(partition='validation', thresholds=thresholds, unavailable=unavailable)
    dump(REPORT / 'thresholds.json', receipt)

def select_cpu():
    (_, _, action_checks) = cpu_modules()
    scores = load(OUT / 'cpu_scores.json')
    population = [r for r in rows(cfg()['inputs']['pairs']) if r['partition'] == 'validation']
    (selected, y, excluded) = labeled('validation', population)
    assert not excluded
    result = {name: action_checks.select_threshold(pair_scores(selected, {rid: s[name] for (rid, s) in scores.items()}), y) for name in next(iter(scores.values()))}
    dump(OUT / 'cpu_validation_thresholds.json', result)

def evaluate():
    (np, _, action_checks) = cpu_modules()
    scores = load(OUT / 'all_scores.json')
    thresholds = load(REPORT / 'thresholds.json')['thresholds']
    population = [r for r in rows(cfg()['inputs']['pairs']) if r['partition'] == 'test']
    (selected, y, excluded) = labeled('test', population)
    assert sum(y) == 372 and len(y) - sum(y) == 8696 and (not excluded)
    actions = {r['pair_id']: r for r in rows(Path(cfg()['inputs']['caption_actions']) / 'actions_test.jsonl')}
    assert set(actions) == {r['pair_id'] for r in population}
    vlm = {pid: row['parsed']['decision'] == 'mismatch' for (pid, row) in actions.items()}
    result = {}
    curves = {}
    for (name, selection) in thresholds.items():
        record_scores = {rid: score[name] for (rid, score) in scores.items()}
        s = pair_scores(selected, record_scores)
        m = confusion(s, y, selection['threshold'])
        curve = action_checks.pr_curve(s, y)
        m.update(threshold=selection['threshold'], AUPRC=sum(((b['recall'] - a['recall']) * b['precision'] for (a, b) in zip(curve, curve[1:]))), precision_at_vlm_recall=max((p['precision'] for p in curve if p['recall'] >= 334 / 372)), precision_at_rounded_vlm_recall=max((p['precision'] for p in curve if p['recall'] >= 0.897849)), U_test_below=sum((v < selection['threshold'] for v in pair_scores(population, record_scores))), U_validation_below=selection['U_validation_below'])
        (result[name], curves[name]) = (m, curve)
    for name in ('vlm_judgment', 'clip_or_vlm'):
        predictions = {r['pair_id']: vlm[r['pair_id']] or (name == 'clip_or_vlm' and min(scores[r['record_id_a']]['clipscore'], scores[r['record_id_b']]['clipscore']) < cfg()['baseline_record_threshold']) for r in population}
        s = [0.0 if predictions[r['pair_id']] else 1.0 for r in selected]
        m = confusion(s, y, 0.5)
        curve = action_checks.pr_curve(s, y)
        valactions = list(rows(Path(cfg()['inputs']['caption_actions']) / 'actions_validation.jsonl'))
        valpred = [r['parsed']['decision'] == 'mismatch' or (name == 'clip_or_vlm' and min(scores[r['record_id_a']]['clipscore'], scores[r['record_id_b']]['clipscore']) < cfg()['baseline_record_threshold']) for r in valactions]
        m.update(threshold=0.5, AUPRC=sum(((b['recall'] - a['recall']) * b['precision'] for (a, b) in zip(curve, curve[1:]))), precision_at_vlm_recall=max((p['precision'] for p in curve if p['recall'] >= 334 / 372)), precision_at_rounded_vlm_recall=max((p['precision'] for p in curve if p['recall'] >= 0.897849)), U_test_below=sum(predictions.values()), U_validation_below=sum(valpred), score_type='binary final action')
        (result[name], curves[name]) = (m, curve)
    dump(OUT / 'test_PR_curves.json', curves)
    output = dict(labeled_test=dict(positive=sum(y), negative=len(y) - sum(y)), rows=result, unavailable=load(REPORT / 'thresholds.json')['unavailable'])
    dump(REPORT / 'results.json', output)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'infer', 'freeze', 'select_cpu', 'evaluate'])
    a = p.parse_args()
    if a.action == 'infer':
        gpu_worker()
    else:
        globals()[a.action]()
