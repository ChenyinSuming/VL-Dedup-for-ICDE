import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import sys
import time
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_caption_action_eval import read_rows, write_json, write_rows
from scripts.caption_action import reference as ref
RES = RESOURCES
OUT = RES / 'runs/stage_e_clusters'
REPORT = ROOT / str(REPORTS / 'stage_e_clusters')
DECISIONS = ('duplicate', 'non_duplicate', 'keep_augment', 'mismatch', 'uncertain')

def pair_id(a, b):
    return 'stage_e_clusters__' + '__'.join(sorted((a, b)))

def enumerate_pairs(ids, components, partitions):
    groups = defaultdict(list)
    for (rid, component) in zip(ids, components, strict=True):
        groups[int(component)].append(rid)
    pairs = []
    for (component, members) in groups.items():
        for (a, b) in itertools.combinations(sorted(members), 2):
            pairs.append(dict(pair_id=pair_id(a, b), record_id_a=a, record_id_b=b, component=component, in_test=partitions[a] == partitions[b] == 'test'))
    return (sorted(pairs, key=lambda r: (r['record_id_a'], r['record_id_b'])), Counter(map(len, groups.values())))

def public_records():
    cfg = json.loads((REPORT / 'settings.json').read_text())
    dataset = Path(cfg['inputs']['dataset_root'])
    return {r['record_id']: r for r in read_rows(dataset / 'public_records.jsonl')}

def decision_counts(rows, predictions, references):
    distribution = {g: dict.fromkeys(DECISIONS, 0) for g in ['L0', 'L2', 'L3', 'M', 'unlabeled']}
    confusion = dict(TP=0, FP=0, FN=0, TN=0)
    fp_l2_l3 = 0
    for row in rows:
        pid = row['pair_id']
        decision = predictions[pid]['parsed']['decision']
        grade = references.get(pid)
        group = grade if grade in ('L0', 'L2', 'L3', 'M') else 'unlabeled'
        distribution[group][decision] += 1
        if group == 'M':
            confusion['TP' if decision == 'mismatch' else 'FN'] += 1
        elif group in ('L0', 'L2', 'L3'):
            confusion['FP' if decision == 'mismatch' else 'TN'] += 1
            fp_l2_l3 += int(group in ('L2', 'L3') and decision == 'mismatch')
    assert sum((sum(c.values()) for c in distribution.values())) == len(rows)
    (tp, fp, fn) = (confusion[k] for k in ('TP', 'FP', 'FN'))
    ratio = lambda a, b: a / b if b else None
    return dict(distribution=distribution, confusion=confusion, FP_L2_L3=fp_l2_l3, precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn), F1=ratio(2 * tp, 2 * tp + fp + fn), keep_augment={g: distribution[g]['keep_augment'] for g in ('L2', 'L3', 'L0')})

def construct():
    import numpy as np
    import pyarrow.parquet as pq
    cfg = json.loads((REPORT / 'settings.json').read_text())
    graph = Path(cfg['inputs']['graph_root'])
    OUT.mkdir(parents=True, exist_ok=True)
    partitions = {r['record_id']: r['partition'] for r in read_rows(Path(cfg['inputs']['dataset_root']) / 'private/lineage.jsonl')}
    records = public_records()
    ids = pq.read_table(graph / 'component_row_ids.parquet').column('record_id').to_pylist()
    components = np.load(graph / 'HGB_K8/components.npy')
    assert len(set(ids)) == len(ids) == len(records) == len(partitions)
    assert set(ids) == set(records) == set(partitions)
    (pairs, sizes) = enumerate_pairs(ids, components, partitions)
    for (name, selected) in [('U', pairs), ('U_test', [r for r in pairs if r['in_test']]), ('U_rest', [r for r in pairs if not r['in_test']])]:
        write_rows(OUT / (name + '.jsonl'), selected)
    result = dict(records=len(ids), components=sum(sizes.values()), size_counts=dict(sizes), U=len(pairs), U_test=sum((r['in_test'] for r in pairs)))
    write_json(OUT / 'construction.json', result)
    return result

def job_for(row, records):
    import pandas as pd
    from scripts import stage_e
    (a, b) = (records[row['record_id_a']], records[row['record_id_b']])
    (image_a, image_b) = (a['images'][0]['image_path'], b['images'][0]['image_path'])
    data = dict(edge_id=row['pair_id'], record_id_a=a['record_id'], record_id_b=b['record_id'], image_path_a_context=image_a, image_path_b_context=image_b, text_a_context=a['text'], text_b_context=b['text'], identical_pixels=stage_e.identical_pixels(image_a, image_b))
    prompt = stage_e.build_prompt_record(pd.Series(data), RES / 'datasets/benchmark', ROOT)
    return dict(pair_id=row['pair_id'], prompt_record=prompt, messages=stage_e.build_qwen_messages(prompt))

def evaluate():
    pairs = read_rows(OUT / 'U_test.jsonl')
    outputs = read_rows(OUT / 'phase1b_verdicts.jsonl')
    predictions = {r['pair_id']: r for r in outputs}
    assert len(predictions) == len(outputs)
    assert {r['pair_id'] for r in pairs} == set(predictions)
    cfg = json.loads((REPORT / 'settings.json').read_text())
    test_ids = set(predictions)
    references = {}
    for label in read_rows(Path(cfg['inputs']['dataset_root']) / 'sealed/relation_applicability.jsonl'):
        if label['partition'] == 'test':
            pid = pair_id(label['record_id_a'], label['record_id_b'])
            if pid in test_ids:
                assert pid not in references
                references[pid] = label['grade']
    result = decision_counts(pairs, predictions, references)
    result.update(U_test=len(pairs), reference_counts=dict(Counter(references.values())))
    write_json(OUT / 'phase1_results.json', result)
    write_json(REPORT / 'phase1_results.json', result)
    return result

def infer_session(pairs, records, cfg, pcfg, phase, target):
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    c = cfg['inference']
    processor = AutoProcessor.from_pretrained(c['model'], local_files_only=True, min_pixels=c['min_pixels'], max_pixels=c['max_pixels'])
    llm = LLM(model=c['model'], dtype=c['dtype'], max_model_len=c['max_model_len'], max_num_seqs=pcfg['max_num_seqs'], max_num_batched_tokens=c['max_num_batched_tokens'], enforce_eager=c['enforce_eager'], enable_chunked_prefill=c['enable_chunked_prefill'], limit_mm_per_prompt=c['limit_mm_per_prompt'], mm_processor_kwargs={'min_pixels': c['min_pixels'], 'max_pixels': c['max_pixels']}, gpu_memory_utilization=c['gpu_memory_utilization'], disable_log_stats=True)
    settings = SamplingParams(temperature=c['temperature'], max_tokens=c['max_tokens'])
    with Path(target).open('w') as stream:
        for offset in range(0, len(pairs), pcfg['submission_batch_size']):
            block = pairs[offset:offset + pcfg['submission_batch_size']]
            jobs = [job_for(row, records) for row in block]
            (images, prompts, inputs) = ([], [], [])
            try:
                for job in jobs:
                    pictures = [Image.open(job['prompt_record'][key]).convert('RGB') for key in ['image_path_a', 'image_path_b']]
                    images.extend(pictures)
                    prompt = processor.apply_chat_template(job['messages'], tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)
                    inputs.append(dict(prompt=prompt, multi_modal_data={'image': pictures}))
                begin = time.monotonic()
                outputs = llm.generate(inputs, settings, use_tqdm=False)
                elapsed = time.monotonic() - begin
                assert len(outputs) == len(block)
                for (row, output, prompt) in zip(block, outputs, prompts, strict=True):
                    raw = output.outputs[0]
                    verdict = dict(pair_id=row['pair_id'], status='completed', raw_return=raw.text, finish_reason=raw.finish_reason, prompt_tokens=len(output.prompt_token_ids), output_tokens=len(raw.token_ids), rendered_prompt=prompt, parsed=ref.strict_parse(raw.text), batch_elapsed_seconds=elapsed, batch_size=len(block))
                    stream.write(json.dumps(verdict, ensure_ascii=False) + '\n')
            finally:
                for picture in images:
                    picture.close()

def infer(phase='test', batch_index=0):
    cfg = json.loads((REPORT / 'settings.json').read_text())
    names = {'1b': 'phase1b', 'test': 'U_test', 'rest': 'U_rest', 'all': 'U'}
    source = OUT / (names[phase] + '.jsonl')
    if phase == '1b' and (not source.exists()):
        source = OUT / 'U_test.jsonl'
    pcfg = cfg.get('phases', {}).get('1b', {'max_num_seqs': 4, 'submission_batch_size': 4})
    target = OUT / ('phase1b_verdicts.jsonl' if phase in ('1b', 'test') else names[phase] + '_verdicts.jsonl')
    infer_session(read_rows(source), public_records(), cfg, pcfg, phase, target)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['construct', 'infer', 'evaluate'])
    p.add_argument('--phase', choices=['1b', 'test', 'rest', 'all'], default='test')
    a = p.parse_args()
    if a.action == 'infer':
        infer(a.phase)
    else:
        globals()[a.action]()
