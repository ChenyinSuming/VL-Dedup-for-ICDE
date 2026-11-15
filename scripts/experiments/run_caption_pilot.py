"""NOISY_POOL_RUN adapters; historical inputs remain read only."""
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
import argparse
from collections import Counter, defaultdict
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
RUN = RES / 'runs/caption_pilot'
REPORT = ROOT / 'outputs/reports/caption_pilot'
SOURCE = RES / 'runs/coco_scale'


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def writel(path, values):
    with Path(path).open('x') as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def prepare_a():
    import numpy as np
    import pandas as pd
    begin = time.monotonic()
    frame = pd.read_parquet(SOURCE / 'corpus.parquet')
    assert len(frame) == 580767 and frame.image_id.nunique() == 116091
    assert frame.record_id.is_unique
    sampled = sorted(random.Random(20261102).sample(sorted(map(int, frame.image_id.unique())), 500))
    ordered = frame.sort_values('record_id').copy()
    labels = np.load(SOURCE / 'vl_dedup_lr_K5_components.npy')
    assert len(labels) == len(ordered)
    ordered['component'] = labels
    selected = set(ordered.loc[ordered.image_id.isin(sampled), 'component'])
    expanded = ordered.loc[ordered.component.isin(selected)].copy()
    clip = dict(zip(frame.record_id, map(float, np.load(SOURCE / 'CLIPScore.npy')), strict=True))
    records = []
    for row in expanded.to_dict('records'):
        records.append(dict(record_id=row['record_id'], image_id=int(row['image_id']),
                            caption=row['text'], image_path=row['image_path'],
                            component=int(row['component']), clipscore=clip[row['record_id']],
                            sampled_image=int(row['image_id']) in sampled))
    groups = expanded.groupby('component').record_id.apply(list).to_dict()
    pairs = [dict(pair_id='r56a:' + str(c) + ':' + str(i), record_id_a=a, record_id_b=b)
             for c, ids in sorted(groups.items())
             for i, (a, b) in enumerate(itertools.combinations(sorted(ids), 2))]
    assert all(len(ids) <= 5 for ids in groups.values())
    assert len(pairs) == sum(len(ids)*(len(ids)-1)//2 for ids in groups.values())
    writel(RUN / 'a_records.jsonl', records)
    writel(RUN / 'a_pairs.jsonl', pairs)
    receipt = dict(sampled_image_ids=sampled, sampled_images=500,
                   sampled_records=sum(r['sampled_image'] for r in records),
                   expanded_records=len(records), expanded_images=len({r['image_id'] for r in records}),
                   components=len(groups), component_sizes=dict(sorted(Counter(map(len, groups.values())).items())),
                   pair_calls=len(pairs), record_calls=len(records),
                   maximum_audit_requests=1000, preparation_wall_seconds=time.monotonic()-begin)
    dump(RUN / 'a_preparation.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'sampled_image_ids'}), flush=True)


def infer_a(prefix='a'):
    assert os.environ.get('VL_DEDUP_GPU_LEASE') == '1', 'GPU worker requires controller flock'
    # Existing vLLM environment is read only; no dependency installation here.
    site = str(RES / 'envs/vllm-n3/lib/python3.10/site-packages')
    sys.path.insert(0, site)
    os.environ['PYTHONPATH'] = site
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['TRITON_CACHE_DIR'] = str(RUN / 'runtime_cache/triton')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(RUN / 'runtime_cache/torchinductor')
    sys.path.insert(0, str(ROOT))
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from importlib.metadata import version
    from scripts import run_stage_e_clusters as stage_e_clusters
    from scripts import score_caption_mismatches as mismatch_scores
    assert version('vllm') == '0.11.0'
    records = rows(RUN / f'{prefix}_records.jsonl')
    pairs = rows(RUN / f'{prefix}_pairs.jsonl')
    workload = load(RUN / 'workload.json') if (RUN / 'workload.json').exists() else dict(records=len(records), pairs=len(pairs))
    session_path = RUN / f'{prefix}_gpu_session.json'
    public = {r['record_id']: dict(record_id=r['record_id'], text=r['caption'],
                                images=[{'image_path': r['image_path']}]) for r in records}
    c = load(ROOT / 'outputs/reports/stage_e_clusters/protocol.json')['inference']
    start = time.monotonic()
    receipt = dict(status='loading', gpu=subprocess.check_output(['nvidia-smi'], text=True),
                   versions={k: version(k) for k in ['vllm', 'torch', 'transformers']},
                   record_terminal=0, pair_terminal=0, record_seconds=0., pair_seconds=0.)
    dump(session_path, receipt)
    try:
        processor = AutoProcessor.from_pretrained(c['model'], local_files_only=True,
                                                 min_pixels=c['min_pixels'], max_pixels=c['max_pixels'])
        model = LLM(model=c['model'], dtype=c['dtype'], max_model_len=c['max_model_len'],
                    max_num_seqs=16, max_num_batched_tokens=4096, enforce_eager=True,
                    enable_chunked_prefill=True, limit_mm_per_prompt={'image': 2},
                    mm_processor_kwargs={'min_pixels': c['min_pixels'], 'max_pixels': c['max_pixels']},
                    gpu_memory_utilization=.92, disable_log_stats=True, seed=0)
        receipt['load_seconds'] = time.monotonic()-start

        def batch(kind, block, stream):
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
                dump(RUN / f'{prefix}_active_batch.json', dict(kind=kind, ids=[r.get('pair_id', r.get('record_id')) for r in block]))
                begin = time.monotonic()
                outputs = model.generate(inputs, settings, use_tqdm=False)
                receipt[kind + '_seconds'] += time.monotonic()-begin
                assert len(outputs) == len(block)
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
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    receipt[kind + '_terminal'] += 1
                stream.flush()
                (RUN / f'{prefix}_active_batch.json').unlink()
                receipt.update(status='running', wall_seconds=time.monotonic()-start)
                dump(session_path, receipt)
            finally:
                for picture in pictures:
                    picture.close()

        with (RUN / f'{prefix}_record_scores.jsonl').open('x') as scores, (RUN / f'{prefix}_pair_verdicts.jsonl').open('x') as verdicts:
            for kind, values, stream in [('record', records[:32], scores), ('pair', pairs[:32], verdicts)]:
                for offset in range(0, len(values), 16):
                    batch(kind, values[offset:offset+16], stream)
            projected = receipt['load_seconds'] + sum(
                receipt[k + '_seconds']/receipt[k + '_terminal']*n
                for k, n in [('record', workload['records']), ('pair', workload['pairs'])])
            receipt.update(projected_stage_E_seconds=projected, timing_gate='passed' if projected <= 21600 else 'failed')
            receipt['workload'] = workload
            dump(session_path, receipt)
            print(json.dumps(receipt), flush=True)
            if projected > 21600:
                receipt['status'] = 'timing_gate_failed'
                dump(session_path, receipt)
                return
            if workload.get('pilot_only'):
                receipt['status'] = 'pilot_complete'
                dump(session_path, receipt)
                return
            for kind, values, stream in [('record', records, scores), ('pair', pairs, verdicts)]:
                for offset in range(32, len(values), 16):
                    batch(kind, values[offset:offset+16], stream)
                    if time.monotonic()-start > 21600:
                        receipt['status'] = 'time_limit_stopped'
                        dump(session_path, receipt)
                        return
            receipt.update(status='complete', wall_seconds=time.monotonic()-start)
            dump(session_path, receipt)
    except BaseException as exc:
        receipt.update(status='failed', error=type(exc).__name__, detail=str(exc), wall_seconds=time.monotonic()-start)
        dump(session_path, receipt)
        raise


def split_b():
    import io
    import zipfile
    import numpy as np
    import pandas as pd
    import imagehash
    from PIL import Image
    begin = time.monotonic()
    source = RUN / 'caption_datasets_official_complete.zip'
    if not source.exists():
        source = RUN / 'caption_datasets.zip'
    with zipfile.ZipFile(source) as archive:
        original = archive.read('dataset_coco.json')
    (RUN / 'dataset_coco.json').write_bytes(original)
    data = json.loads(original)['images']
    counts = Counter(r['split'] for r in data)
    observed = dict(train_union_restval=counts['train']+counts['restval'], val=counts['val'], test=counts['test'])
    expected = dict(train_union_restval=113287, val=5000, test=5000)
    receipt = dict(gate1='passed' if observed == expected else 'failed', raw_split_counts=dict(counts),
                   expected=expected, observed=observed, source_url=load(CONFIG / 'caption_pilot.json')['B']['split_source'])
    dump(RUN / 'b_split_gate.json', receipt)
    print(json.dumps(receipt), flush=True)
    if observed != expected:
        return
    sys.path.insert(0, str(ROOT))
    from scripts.run_coco_scale import load_exclusions
    excluded = {image_id for _, image_id in load_exclusions(RES / 'datasets/VL-Dedup-sources/source_inventory.csv')}
    pools = {name: sorted(int(r['cocoid']) for r in data if r['split'] in splits and int(r['cocoid']) not in excluded)
             for name, splits in [('train', ['train', 'restval']), ('validation', ['val'])]}
    chosen = {name: sorted(random.Random(20261102).sample(pools[name], n))
              for name, n in [('train', 5000), ('validation', 1000)]}
    chosen['test'] = sorted(int(r['cocoid']) for r in data if r['split'] == 'test')
    dump(RUN / 'b_membership.json', dict(image_ids=chosen, exclusion_count=len(excluded),
         eligible_pool_sizes={name: len(values) for name, values in pools.items()},
         test_intersection_with_benchmark=len(set(chosen['test']) & excluded)))
    frame = pd.read_parquet(SOURCE / 'corpus.parquet', columns=['image_id'])
    values = np.load(SOURCE / 'pHash.npy', mmap_mode='r')
    first = ~frame.image_id.duplicated()
    hashes = {int(i): int(v) for i, v in zip(frame.image_id[first], values[first], strict=True)}
    missing = sorted(set.union(*(set(v) for v in chosen.values())) - hashes.keys())
    archives = [zipfile.ZipFile(str(COCO_ARCHIVES / (split + '.zip'))) for split in ['train2017', 'val2017']]
    members = {int(Path(name).stem): (archive, name) for archive in archives for name in archive.namelist() if name.endswith('.jpg')}
    for image_id in missing:
        archive, name = members[image_id]
        with Image.open(io.BytesIO(archive.read(name))) as picture:
            hashes[image_id] = int(str(imagehash.phash(picture.convert('RGB'))), 16)
    for archive in archives:
        archive.close()
    dump(RUN / 'b_image_phash.json', {str(i): hashes[i] for i in sorted(set.union(*(set(v) for v in chosen.values())))})
    # <=2 changed bits leaves one of three disjoint bit chunks identical.
    chunks = [(0, (1 << 22)-1), (22, (1 << 21)-1), (43, (1 << 21)-1)]
    overlaps = {}
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        buckets = defaultdict(set)
        for image_id in chosen[right]:
            for part, (offset, mask) in enumerate(chunks):
                buckets[(part, (hashes[image_id] >> offset) & mask)].add(image_id)
        matches = []
        for image_id in chosen[left]:
            candidates = set.union(*(buckets[(part, (hashes[image_id] >> offset) & mask)]
                                     for part, (offset, mask) in enumerate(chunks)))
            for other in sorted(candidates):
                distance = (hashes[image_id] ^ hashes[other]).bit_count()
                if distance <= 2:
                    matches.append(dict(left_id=image_id, right_id=other, hamming=distance))
        overlaps[left + '_vs_' + right] = dict(image_id_overlap=sorted(set(chosen[left]) & set(chosen[right])), phash_matches=matches)
    passed = all(not v['image_id_overlap'] and not v['phash_matches'] for v in overlaps.values())
    receipt = dict(gate2='passed' if passed else 'failed', overlaps=overlaps,
                   cached_hash_images=sum(len(v) for v in chosen.values())-len(missing),
                   newly_computed_hash_images=len(missing), wall_seconds=time.monotonic()-begin)
    dump(RUN / 'b_leakage_gate.json', receipt)
    print(json.dumps({'gate2': receipt['gate2'], 'overlaps': {k: dict(image_id=len(v['image_id_overlap']), phash=len(v['phash_matches'])) for k, v in overlaps.items()}, 'wall_seconds': receipt['wall_seconds']}), flush=True)


def export_a():
    sys.path.insert(0, str(ROOT))
    from scripts import run_combined_actions as combined_actions
    assert load(RUN / 'a_gpu_session.json')['status'] == 'complete'
    records = rows(RUN / 'a_records.jsonl')
    pairs = rows(RUN / 'a_pairs.jsonl')
    verdicts = rows(RUN / 'a_pair_verdicts.jsonl')
    probabilities = {r['record_id']: r for r in rows(RUN / 'a_record_scores.jsonl')}
    assert len(probabilities) == len(records) and len(verdicts) == len(pairs)
    assert [r['pair_id'] for r in verdicts] == [r['pair_id'] for r in pairs]
    reference = RES / 'runs/mismatch_scores'
    validation = [r['record_id'] for r in rows(reference / 'records.jsonl') if r['partition'] == 'validation']
    assert len(validation) == 6132
    validation_set = set(validation)
    cpu = load(reference / 'cpu_scores.json')
    clip = {r: cpu[r]['clipscore'] for r in validation}
    clip.update({r['record_id']: r['clipscore'] for r in records})
    probabilities.update({r['record_id']: r for r in rows(reference / 'qwen25_vllm.jsonl') if r['record_id'] in validation_set})
    fusion = combined_actions.combined_scores(clip, probabilities, validation)
    decisions = {tuple(sorted((pair['record_id_a'], pair['record_id_b']))): verdict['parsed']['decision']
                 for pair, verdict in zip(pairs, verdicts, strict=True)}
    groups = defaultdict(list)
    for record in records:
        groups[record['component']].append(record['record_id'])
    export = []
    states = {}
    for component, ids in sorted(groups.items()):
        selected = {p: decisions[p] for p in itertools.combinations(sorted(ids), 2)}
        representative, arms = combined_actions.export(ids, clip, fusion, selected)
        export.append(dict(component=component, representative=representative, **arms['b']))
        for state, members in arms['b'].items():
            for record_id in members:
                states[record_id] = state
    writel(RUN / 'a_exports.jsonl', export)
    dump(RUN / 'a_fusion.json', {r['record_id']: fusion[r['record_id']] for r in records})
    by_image = defaultdict(Counter)
    for record in records:
        if record['sampled_image']:
            by_image[record['image_id']][states[record['record_id']]] += 1
    distribution = Counter(v['retained'] for v in by_image.values())
    totals = sum(by_image.values(), Counter())
    result = dict(images=len(by_image), records=sum(totals.values()), retained_count_distribution=dict(sorted(distribution.items())),
                  record_states=dict(totals), invalid_record_scores=sum(not probabilities[r['record_id']]['valid'] for r in records),
                  invalid_pair_verdicts=sum(not r['parsed']['contract_valid'] for r in verdicts),
                  coverage=dict(strict=None, loose=None, deleted_denominator=totals['deleted'], status='pending_audit'))
    result['expanded_context_records'] = len(records) - result['records']
    assert result['images'] == 500 and result['records'] == sum(r['sampled_image'] for r in records)
    dump(RUN / 'a_results_before_audit.json', result)
    print(json.dumps(result), flush=True)


def prepare_b_pilot():
    import numpy as np
    import pandas as pd
    assert load(RUN / 'b_graph_gate.json')['gate3'] == 'passed'
    cap = load(RUN / 'b_K_frozen.json')['K']
    all_records, all_pairs, sizes = {}, [], {}
    for part in ['train', 'validation']:
        folder = RUN / 'b' / part
        frame = pd.read_parquet(folder / 'corpus.parquet')
        clip = dict(zip(frame.record_id, map(float, np.load(folder / 'CLIPScore.npy')), strict=True))
        ordered = frame.sort_values('record_id').copy()
        ordered['component'] = np.load(folder / f'vl_dedup_lr_K{cap}_components.npy')
        for row in ordered.to_dict('records'):
            all_records[row['record_id']] = dict(record_id=row['record_id'], image_id=row['image_id'],
                image_path=row['image_path'], caption=row['text'], clipscore=clip[row['record_id']], partition=part)
        before = len(all_pairs)
        for component, group in ordered.groupby('component'):
            all_pairs.extend(dict(pair_id=f'r56b:{part}:{component}:{i}', record_id_a=a, record_id_b=b)
                for i, (a, b) in enumerate(itertools.combinations(sorted(group.record_id), 2)))
        sizes[part] = dict(records=len(frame), pairs=len(all_pairs)-before)
    rng = random.Random(20261102)
    pair_sample = rng.sample(all_pairs, 32)
    record_ids = rng.sample(sorted(all_records), 32)
    requested_records = record_ids + sorted({r[k] for r in pair_sample for k in ['record_id_a', 'record_id_b']} - set(record_ids))
    folder = RUN / 'b_E_pilot'
    folder.mkdir(exist_ok=True)
    writel(folder / 'b_records.jsonl', [all_records[r] for r in requested_records])
    writel(folder / 'b_pairs.jsonl', pair_sample)
    dump(folder / 'workload.json', dict(records=len(all_records), pairs=len(all_pairs), partitions=sizes,
         K=cap, pilot_only=True, record_scoring_sample=record_ids,
         selection='Random(20261102), 32 uniform eligible pairs and 32 uniform records; extra endpoints are pair context only'))
    print(json.dumps(dict(K=cap, partitions=sizes, records=len(all_records), pairs=len(all_pairs))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare-a', 'infer-a', 'export-a', 'split-b', 'prepare-b-pilot', 'infer-b-pilot'])
    args = parser.parse_args()
    if args.stage == 'infer-b-pilot':
        RUN = RUN / 'b_E_pilot'
        infer_a('b')
    else:
        {'prepare-a': prepare_a, 'infer-a': infer_a, 'export-a': export_a,
         'split-b': split_b, 'prepare-b-pilot': prepare_b_pilot}[args.stage]()
