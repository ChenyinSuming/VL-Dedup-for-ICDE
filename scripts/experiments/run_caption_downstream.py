"""CAPTION_DOWNSTREAM_RUN frozen-input preparation and the 1,000-pair Stage E timing gate."""
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
import argparse
from collections import Counter
from fractions import Fraction
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
RES = RESOURCES
SOURCE = RES / 'runs/caption_pilot'
RUN = RES / 'runs/caption_downstream'
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_caption_pilot import dump, load, rows, writel


def component_metrics(frame, labels):
    sizes = Counter(map(int, labels))
    groups = Counter(map(str, frame.visual_group_id))
    joint = Counter(zip(map(int, labels), map(str, frame.visual_group_id)))
    choose2 = lambda counts: sum(n * (n - 1) // 2 for n in counts)
    tp = choose2(joint.values())
    predicted = choose2(sizes.values())
    actual = choose2(groups.values())
    return dict(TP=tp, FP=predicted-tp, FN=actual-tp,
                exact_F1=str(Fraction(2*tp, predicted+actual)), pairs=predicted)


def prepare():
    import numpy as np
    import pandas as pd
    start = time.monotonic()
    records, pairs, summary = [], [], {}
    gates = {name: load(SOURCE / f'b_{name}.json')
             for name in ['K_frozen', 'graph_gate', 'leakage_gate']}
    assert gates['K_frozen']['K'] == 'inf'
    assert gates['K_frozen']['validation_graph']['exact_f1'] == '27712/28103'
    for part in ['train', 'validation']:
        folder = SOURCE / 'b' / part
        frame = pd.read_parquet(folder / 'corpus.parquet')
        clip = np.load(folder / 'CLIPScore.npy')
        assert frame.record_id.is_unique and len(clip) == len(frame)
        ordered = frame.sort_values('record_id').copy()
        labels = np.load(folder / 'vl_dedup_lr_Kinf_components.npy')
        assert len(labels) == len(ordered)
        ordered['component'] = labels
        metrics = component_metrics(ordered, labels)
        summary[part] = dict(records=len(frame), original_images=int(frame.original_image_id.nunique()),
                             copy_records=int(frame.is_copy.sum()),
                             swapped_records=int(frame.caption_swapped.sum()), graph=metrics)
        clips = dict(zip(frame.record_id, map(float, clip), strict=True))
        for row in ordered.to_dict('records'):
            records.append(dict(record_id=row['record_id'], image_path=row['image_path'],
                                caption=row['text'], clipscore=clips[row['record_id']],
                                partition=part, component=int(row['component'])))
        if part == 'train':
            for component, group in ordered.groupby('component'):
                pairs.extend(dict(pair_id=f'caption_downstream:train:{a}:{b}', record_id_a=a, record_id_b=b)
                             for a, b in itertools.combinations(sorted(group.record_id), 2))
    clean = pd.read_parquet(SOURCE / 'b/clean.parquet')
    summary['clean'] = dict(records=len(clean), original_images=int(clean.image_id.nunique()))
    observed = (summary['train']['records'], summary['train']['copy_records'],
                summary['train']['swapped_records'], summary['train']['original_images'],
                summary['validation']['records'], summary['validation']['original_images'],
                summary['clean']['records'], summary['clean']['original_images'])
    assert observed == (35208, 10199, 6989, 5000, 6852, 1000, 25009, 5000), observed
    assert summary['validation']['graph']['exact_F1'] == '27712/28103'
    assert Fraction(summary['train']['graph']['exact_F1']) >= Fraction(9, 10)
    assert len(pairs) == summary['train']['graph']['pairs']
    assert gates['graph_gate']['gate3'] == 'passed'
    assert gates['leakage_gate']['gate2'] == 'passed'
    pairs.sort(key=lambda r: (r['record_id_a'], r['record_id_b']))
    records.sort(key=lambda r: r['record_id'])
    pilot_pairs = random.Random(20261104).sample(pairs, 1000)
    pilot_records = random.Random(20261105).sample(records, 256)
    writel(RUN / 'records.jsonl', records)
    writel(RUN / 'train_pairs.jsonl', pairs)
    writel(RUN / 'pilot_pairs.jsonl', pilot_pairs)
    writel(RUN / 'pilot_records.jsonl', pilot_records)
    dump(RUN / 'workload.json', dict(records=len(records), pairs=len(pairs),
         pilot_records=len(pilot_records), pilot_pairs=len(pilot_pairs),
         record_seed=20261105, pair_seed=20261104))
    dump(RUN / 'gate1.json', dict(status='passed', observed=summary,
         source=str(SOURCE), leakage=gates['leakage_gate'],
         preparation_seconds=time.monotonic()-start))
    print(json.dumps(dict(status='prepared', **load(RUN / 'workload.json'))), flush=True)


def infer(full=False):
    assert os.environ.get('VL_DEDUP_GPU_LEASE') == '1', 'controller must hold shared flock'
    assert load(RUN / 'gate1.json')['status'] == 'passed'
    assert load(RUN / 'gate2.json')['status'] == 'passed'
    if (RUN / 'active_batch.json').exists():
        raise RuntimeError('Unresolved admitted batch; reconcile before resuming, no automatic retry')
    if full:
        assert load(RUN / 'stage_e_timing_gate.json')['status'] == 'passed'
        if (RUN / 'stage_e_complete.json').exists():
            print(json.dumps(load(RUN / 'stage_e_complete.json')), flush=True)
            return
    elif (RUN / 'stage_e_timing_gate.json').exists():
        print(json.dumps(load(RUN / 'stage_e_timing_gate.json')), flush=True)
        return
    site = str(RES / 'envs/vllm-n3/lib/python3.10/site-packages')
    sys.path.insert(0, site)
    os.environ['PYTHONPATH'] = site
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['TRITON_CACHE_DIR'] = str(RUN / 'runtime_cache/triton')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(RUN / 'runtime_cache/torchinductor')
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from importlib.metadata import version
    from scripts import run_stage_e_clusters as stage_e_clusters
    from scripts import score_caption_mismatches as mismatch_scores
    assert version('vllm') == '0.11.0'
    config = load(ROOT / 'outputs/reports/stage_e_clusters/protocol.json')['inference']
    records = rows(RUN / 'records.jsonl')
    public = {r['record_id']: dict(record_id=r['record_id'], text=r['caption'],
                                 images=[{'image_path': r['image_path']}]) for r in records}
    workload = load(RUN / 'workload.json')
    progress_path = RUN / 'progress.json'
    previous = load(progress_path) if progress_path.exists() else {}
    receipt = dict(status='loading', gpu=subprocess.check_output(['nvidia-smi'], text=True),
                   versions={k: version(k) for k in ['vllm', 'torch', 'transformers']},
                   record_terminal=previous.get('record_terminal', 0),
                   pair_terminal=previous.get('pair_terminal', 0),
                   record_seconds=previous.get('record_seconds', 0.),
                   pair_seconds=previous.get('pair_seconds', 0.),
                   load_seconds=previous.get('load_seconds', 0.),
                   session_start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    dump(progress_path, receipt)
    start = time.monotonic()
    try:
        processor = AutoProcessor.from_pretrained(config['model'], local_files_only=True,
                         min_pixels=config['min_pixels'], max_pixels=config['max_pixels'])
        model = LLM(model=config['model'], dtype=config['dtype'], max_model_len=8192,
                    max_num_seqs=16, max_num_batched_tokens=4096, enforce_eager=True,
                    enable_chunked_prefill=True, limit_mm_per_prompt={'image': 2},
                    mm_processor_kwargs={'min_pixels': config['min_pixels'], 'max_pixels': config['max_pixels']},
                    gpu_memory_utilization=.92, disable_log_stats=True, seed=0)
        receipt['load_seconds'] += time.monotonic()-start

        def batch(kind, block, stream):
            begin = time.monotonic()
            pictures, inputs = [], []
            try:
                for row in block:
                    if kind == 'record':
                        paths = [row['image_path']]
                        prompt_text = 'Caption: ' + row['caption'] + '\nDoes this caption correctly describe the image? Answer with yes or no.'
                        messages = [{'role': 'user', 'content': [
                            {'type': 'image', 'image': paths[0]}, {'type': 'text', 'text': prompt_text}]}]
                    else:
                        job = stage_e_clusters.job_for(row, public)
                        paths = [job['prompt_record'][k] for k in ['image_path_a', 'image_path_b']]
                        messages = job['messages']
                    images = [Image.open(path).convert('RGB') for path in paths]
                    pictures.extend(images)
                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs.append(dict(prompt=prompt, multi_modal_data={'image': images}))
                settings = SamplingParams(temperature=0, max_tokens=1, logprobs=20) if kind == 'record' else SamplingParams(temperature=0, max_tokens=384)
                dump(RUN / 'active_batch.json', dict(kind=kind, ids=[r.get('pair_id', r.get('record_id')) for r in block]))
                outputs = model.generate(inputs, settings, use_tqdm=False)
                assert len(outputs) == len(block)
                results = []
                for row, output in zip(block, outputs, strict=True):
                    raw = output.outputs[0]
                    result = dict(raw_return=raw.text, finish_reason=raw.finish_reason,
                                  prompt_tokens=len(output.prompt_token_ids), output_tokens=len(raw.token_ids))
                    if kind == 'record':
                        tokens = [dict(token_id=int(tid), text=processor.tokenizer.decode([tid]),
                                       logprob=float(lp.logprob), rank=int(lp.rank))
                                  for tid, lp in raw.logprobs[0].items() if lp.rank <= 20]
                        result.update(record_id=row['record_id'], tokens=tokens, **mismatch_scores.yes_probability(tokens))
                    else:
                        result.update(pair_id=row['pair_id'], parsed=stage_e_clusters.ref.strict_parse(raw.text))
                    results.append(result)
                for result in results:
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                receipt[kind + '_terminal'] += len(results)
                receipt[kind + '_seconds'] += time.monotonic()-begin
                receipt.update(status='running', session_wall_seconds=time.monotonic()-start)
                dump(progress_path, receipt)
                (RUN / 'active_batch.json').unlink()
            finally:
                for picture in pictures:
                    picture.close()

        for kind in ['record', 'pair']:
            values = (records if kind == 'record' else rows(RUN / 'train_pairs.jsonl')) if full else rows(RUN / f'pilot_{kind}s.jsonl')
            path = RUN / ('record_scores.jsonl' if kind == 'record' else 'pair_verdicts.jsonl')
            existing = rows(path) if path.exists() else []
            id_key = kind + '_id'
            done = {r[id_key] for r in existing}
            assert len(done) == len(existing)
            assert done <= {r[id_key] for r in values}
            assert len(done) == receipt[kind + '_terminal']
            remaining = [r for r in values if r[id_key] not in done]
            with path.open('a') as stream:
                for offset in range(0, len(remaining), 16):
                    batch(kind, remaining[offset:offset+16], stream)
                    if full and sum(receipt[k] for k in ['load_seconds', 'record_seconds', 'pair_seconds']) > 24*3600:
                        receipt.update(status='actual_stage_e_time_limit_stopped')
                        dump(progress_path, receipt)
                        return
        if full:
            assert receipt['record_terminal'] == workload['records']
            assert receipt['pair_terminal'] == workload['pairs']
            receipt.update(status='complete', actual_stage_e_seconds=sum(
                receipt[k] for k in ['load_seconds', 'record_seconds', 'pair_seconds']))
            dump(progress_path, receipt)
            dump(RUN / 'stage_e_complete.json', receipt)
            print(json.dumps(receipt), flush=True)
            return
        assert receipt['record_terminal'] == 256 and receipt['pair_terminal'] == 1000
        projected = receipt['load_seconds'] + sum(receipt[k+'_seconds']/receipt[k+'_terminal']*workload[k+'s']
                                                    for k in ['record', 'pair'])
        gate = dict(status='passed' if projected <= 24*3600 else 'failed',
                    limit_seconds=24*3600, projected_stage_e_seconds=projected,
                    projected_stage_e_hours=projected/3600,
                    workload=workload, measurement=receipt,
                    full_stage_e_executed=False)
        receipt.update(status='pilot_complete' if gate['status'] == 'passed' else 'timing_gate_failed',
                       session_wall_seconds=time.monotonic()-start)
        dump(progress_path, receipt)
        dump(RUN / 'stage_e_timing_gate.json', gate)
        print(json.dumps(gate), flush=True)
    except BaseException as exc:
        receipt.update(status='failed', error=type(exc).__name__, detail=str(exc),
                       session_wall_seconds=time.monotonic()-start)
        dump(progress_path, receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'pilot', 'stage-e'])
    args = parser.parse_args()
    {'prepare': prepare, 'pilot': lambda: infer(False), 'stage-e': lambda: infer(True)}[args.stage]()
