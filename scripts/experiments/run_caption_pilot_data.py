"""NOISY_POOL_RUN noisy COCO construction and frozen CLASSIFIER_RUN visual graphs."""
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
from collections import Counter
from contextlib import contextmanager
from fractions import Fraction
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_caption_pilot import RUN, REPORT, RES, SOURCE, load, dump


def construct():
    import pandas as pd
    from PIL import Image
    begin = time.monotonic()
    assert load(RUN / 'b_split_gate.json')['gate1'] == 'passed'
    assert load(RUN / 'b_leakage_gate.json')['gate2'] == 'passed'
    membership = load(RUN / 'b_membership.json')['image_ids']
    originals = {int(r['cocoid']): r for r in load(RUN / 'dataset_coco.json')['images']}
    paths = pd.read_parquet(SOURCE / 'corpus.parquet', columns=['image_id', 'image_path']).drop_duplicates('image_id').set_index('image_id').image_path.to_dict()
    source = ROOT / 'scripts/build_edit_stress.py'
    spec = importlib.util.spec_from_file_location('noisy_pool_edits', source)
    edits = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edits)
    receipt = {}
    for part, seed in [('train', 20261103), ('validation', 20261104)]:
        rng = random.Random(seed)
        ids = membership[part]
        copy_sources = sorted(rng.sample(ids, len(ids)//5))
        counts = {i: rng.randint(1, 3) for i in copy_sources}
        slots = [(i, j) for i in copy_sources for j in range(1, counts[i]+1)]
        levels = ['S2']*(len(slots)//2) + ['S4']*(len(slots)//2)
        if len(slots) % 2:
            levels.append(rng.choice(['S2', 'S4']))
        rng.shuffle(levels)
        swapped = rng.sample(ids, len(ids)//5)
        donors = {i: swapped[(j+1) % len(swapped)] for j, i in enumerate(swapped)}
        assert all(i != donor for i, donor in donors.items())
        assets = [(i, 0, int(i), paths[i], None) for i in ids]
        folder = RUN / 'b' / part
        (folder / 'images').mkdir(parents=True, exist_ok=True)
        for n, ((image_id, copy_index), level) in enumerate(zip(slots, levels, strict=True)):
            params = edits.parameters(level, rng.randrange(1 << 32), flipped=False)
            output = folder / 'images' / f'{image_id:012}_{copy_index}.jpg'
            assert not output.exists(), 'never overwrite an existing edited asset'
            with Image.open(paths[image_id]) as image:
                transformed, crop_box = edits.transform(image, params)
                transformed.save(output, format='JPEG', quality=params['quality'])
                transformed.close()
            params.update(crop_box=list(crop_box), level=level)
            assets.append((image_id, copy_index, 1000000+n, str(output), params))
        records = []
        for image_id, copy_index, asset_id, image_path, params in assets:
            donor = donors.get(image_id, image_id)
            for caption in originals[donor]['sentences']:
                records.append(dict(record_id=f'noisy_pool:{part}:{image_id:012}:{copy_index}:{caption["sentid"]}',
                    image_id=asset_id, original_image_id=image_id, copy_index=copy_index,
                    is_copy=copy_index != 0, caption_source_image_id=donor, caption_swapped=donor != image_id,
                    text=caption['raw'], image_path=image_path, split=part, partition=part,
                    visual_group_id=str(image_id), edit_parameters=json.dumps(params, sort_keys=True)))
        frame = pd.DataFrame(records)
        frame.to_parquet(folder / 'corpus.parquet', index=False)
        assert frame.record_id.is_unique and len(ids) == frame.original_image_id.nunique()
        receipt[part] = dict(original_images=len(ids), selected_copy_sources=len(copy_sources),
                            added_image_assets=len(slots), mild_copies=levels.count('S2'), strong_copies=levels.count('S4'),
                            selected_swapped_originals=len(swapped), records=len(frame),
                            copy_records=int(frame.is_copy.sum()), swapped_caption_records=int(frame.caption_swapped.sum()),
                            copy_and_swapped_records=int((frame.is_copy & frame.caption_swapped).sum()))
        dump(folder / 'noise_receipt.json', receipt[part])
        if part == 'train':
            clean = []
            for image_id in ids:
                for caption in originals[image_id]['sentences']:
                    clean.append(dict(image_id=image_id, image_path=paths[image_id], text=caption['raw'], caption_id=caption['sentid']))
            pd.DataFrame(clean).to_parquet(RUN / 'b/clean.parquet', index=False)
            receipt['clean'] = dict(images=len(ids), records=len(clean))
        print(json.dumps({part: receipt[part]}), flush=True)
    receipt['wall_seconds'] = time.monotonic()-begin
    dump(RUN / 'b_noise_receipt.json', receipt)


@contextmanager
def own_stage(name, gpu=False):
    # Called by COCO_SCALE_RUN with OUT redirected. Only its permitted GPU lease is shared.
    from scripts import run_coco_scale as coco
    receipt_path = coco.OUT / (name + '_stage.json')
    assert not receipt_path.exists()
    lock_path = RES / 'locks/gpu.lock' if gpu else RUN / 'cpu.lock'
    waiting = time.monotonic()
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        begin = time.monotonic()
        receipt = dict(stage=name, status='running', lock_wait_seconds=begin-waiting)
        dump(receipt_path, receipt)
        try:
            yield receipt
            receipt['status'] = 'complete'
        except BaseException as exc:
            receipt.update(status='failed', error=type(exc).__name__, detail=str(exc))
            raise
        finally:
            receipt['wall_seconds'] = time.monotonic()-begin
            dump(receipt_path, receipt)


def features(part):
    from scripts import run_coco_scale as coco
    coco.OUT = RUN / 'b' / part
    coco.REPORT = REPORT / 'b' / part
    coco.REPORT.mkdir(parents=True, exist_ok=True)
    coco.stage = own_stage
    coco.stage_a()


def graph(part, selected_k=None):
    import joblib
    import numpy as np
    import pandas as pd
    from scripts import run_coco_scale as coco
    from scripts import build_candidates as candidate
    from scripts.cluster_with_size_cap import build_graph
    begin = time.monotonic()
    folder = RUN / 'b' / part
    frame = pd.read_parquet(folder / 'corpus.parquet')
    names = ['CLIP_image', 'CLIP_text', 'pHash', 'SSCD']
    image, text, hashes, sscd = [np.load(folder / (n+'.npy'), mmap_mode='r') for n in names]
    candidate.faiss.omp_set_num_threads(8)
    candidate.tqdm = lambda it, *args, **kwargs: it
    feature_dir = folder / 'features/phash'
    feature_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(record_id=frame.record_id, phash_int=hashes)).to_parquet(feature_dir / 'phash.parquet', index=False)
    edges = {}
    channel_counts = dict(phash=candidate.build_phash_edges(frame, folder / 'features', 2, edges))
    for array, channel, k in [(image, 'clip_image', 30), (text, 'clip_text', 10)]:
        array = np.array(array, dtype=np.float32, order='C', copy=True)
        candidate.faiss.normalize_L2(array)
        channel_counts[channel] = candidate.build_embedding_edges(array, channel, channel+'_score', k, None, edges, frame.partition.to_numpy())
    channel_counts['text_exact'] = candidate.build_text_exact_edges(frame, 20, edges)
    candidates = candidate.edges_to_frame(edges, frame)
    candidates.to_parquet(folder / 'candidates.parquet', index=False)
    bundle = joblib.load((RESOURCES / 'runs/scorer_noasset') / 'vl_dedup_lr.joblib')
    raw, calibrated = coco.predict(frame, candidates, image, text, hashes, sscd, bundle)
    ids, a, b, selected = coco.positive_edges(frame, candidates, raw, calibrated, 2/3)
    groups = frame.set_index('record_id').loc[ids].visual_group_id.to_numpy()
    caps = [selected_k] if selected_k is not None else load(CONFIG / 'caption_pilot.json')['B']['dedup']['K_grid']
    results = []
    for cap in caps:
        labels, _, _, counts = build_graph(len(frame), a, b, cap)
        stats = coco.graph_metrics(labels, groups)
        np.save(folder / f'vl_dedup_lr_K{cap}_components.npy', labels)
        stats.update(K=cap, graph_counts=counts, pairs_for_stage_E=sum(int(n)*(int(n)-1)//2 for n in Counter(labels).values()))
        results.append(stats)
    def exact_f1(row):
        return Fraction(2*row['tp'], 2*row['tp']+row['fp']+row['fn'])
    if selected_k is None:
        best = max(results, key=lambda r: (exact_f1(r), -(float('inf') if r['K'] == 'inf' else r['K'])))
        dump(RUN / 'b_K_frozen.json', dict(K=best['K'], validation_graph=best, selection='validation exact Fraction F1, smaller K tie'))
    receipt = dict(partition=part, records=len(frame), candidates=len(candidates),
                   candidate_channels=channel_counts, graph_results=results,
                   wall_seconds=time.monotonic()-begin)
    dump(folder / 'graph_receipt.json', receipt)
    if part == 'train':
        best = results[0]
        dump(RUN / 'b_graph_gate.json', dict(gate3='passed' if exact_f1(best) >= Fraction(9,10) else 'failed',
             K=selected_k, metrics=best, deleted_swapped_captions=None,
             deletion_status='not evaluated before Stage E-F; graph gate precedes expensive inference'))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['construct', 'features', 'graph'])
    parser.add_argument('--partition', choices=['train', 'validation'])
    args = parser.parse_args()
    if args.stage == 'construct':
        construct()
    elif args.stage == 'features':
        features(args.partition)
    else:
        k = load(RUN / 'b_K_frozen.json')['K'] if args.partition == 'train' else None
        graph(args.partition, k)
