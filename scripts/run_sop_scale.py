from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, REPORTS
import argparse
from collections import Counter
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('_sop_scale_coco_scale', ROOT / 'scripts/run_coco_scale.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
OUT = shared.RES / 'runs/sop_scale'
REPORT = ROOT / str(REPORTS / 'sop_scale')
A2 = OUT / 'stage_a_amendment2'
(shared.OUT, shared.REPORT) = (OUT, REPORT)

def benchmark_sample():
    import pandas as pd
    lineage = pd.read_json(shared.DATA / 'private/lineage.jsonl', lines=True)
    ids = sorted(lineage[lineage.source_dataset.eq('stanford_online_products')].record_id)
    assert len(ids) == 14000
    positions = [i * (len(ids) - 1) // 99 for i in range(100)]
    sample_ids = [ids[p] for p in positions]
    frame = pd.read_parquet(shared.SOURCE / 'public_manifest.parquet').set_index('record_id').loc[sample_ids].reset_index()
    public = {r['record_id']: r['text'] for r in map(json.loads, (shared.DATA / 'public_records.jsonl').open())}
    assert all((r.text == public[r.record_id] for r in frame.itertuples()))
    return (frame, positions)

def caption_requests(frame, path):
    from scripts.extract_features import resolve_image_path
    prompt = shared.read(shared.DATA / 'contract.json')['generation_prompt']
    with Path(path).open('w') as stream:
        for r in frame.itertuples():
            request = dict(id=r.record_id, messages=[dict(role='user', content=[dict(type='image', url=str(resolve_image_path(shared.DATA, r.image_path))), dict(type='text', text=prompt)])])
            stream.write(json.dumps(request) + '\n')

def generate_captions(frame, name):
    from types import SimpleNamespace
    import torch
    from scripts.run_image_caption_batch import run
    request_path = OUT / (name + '_requests.jsonl')
    caption_requests(frame, request_path)
    torch.set_num_threads(4)
    torch.manual_seed(23)
    args = SimpleNamespace(requests=request_path, model=shared.RES / 'models/InternVL3-2B-hf', output=OUT / name, family='internvl', max_new_tokens=80, batch_size=48, max_patches=1)
    run(args)
    responses = [json.loads(line) for line in (args.output / 'responses.jsonl').open()]
    return {r['id']: r['response'] for r in responses}

def captions():
    import pandas as pd
    with shared.stage('captions', gpu=True) as receipt:
        receipt['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'], text=True)
        print('GPU before loading:', receipt['gpu'], flush=True)
        receipt['threads'] = 4
        frame = pd.read_parquet(OUT / 'corpus.parquet')
        begin = time.perf_counter()
        generated = generate_captions(frame.iloc[:2000], 'captions_pilot2000')
        pilot_seconds = time.perf_counter() - begin
        generated.update(generate_captions(frame.iloc[2000:], 'captions_remaining'))
        assert len(generated) == len(frame)
        frame['text'] = frame.record_id.map(generated)
        assert frame.text.notna().all() and frame.text.str.strip().ne('').all()
        frame.to_parquet(OUT / 'corpus.parquet', index=False)
        receipt.update(records=len(frame), generation_wall_seconds=time.perf_counter() - begin, records_per_second=len(frame) / (time.perf_counter() - begin))

def assemble_final_captions(batch_1, batch_2):
    first = {r['id']: r for r in batch_1}
    second = {r['id']: r for r in batch_2}
    assert len(first) == len(batch_1) and len(second) == len(batch_2)
    assert second.keys() <= first.keys()
    rows = []
    for record_id in sorted(first):
        response = second.get(record_id, first[record_id])
        text = response['response'].strip()
        assert text, record_id
        rows.append(dict(record_id=record_id, text=text, caption_batch=2 if record_id in second else 1, caption_status=response['status']))
    return rows

def finalize_captions():
    import pandas as pd
    with shared.stage('caption_assembly') as receipt, shared.configure_threads(8):
        batch_1 = []
        for directory in ['captions_pilot2000', 'captions_remaining']:
            batch_1.extend((json.loads(line) for line in (OUT / directory / 'responses.jsonl').open()))
        batch_2 = [json.loads(line) for line in (OUT / 'captions_remaining/responses.jsonl').open()]
        final = pd.DataFrame(assemble_final_captions(batch_1, batch_2))
        counts = dict(records=len(final), batch_1=int(final.caption_batch.eq(1).sum()), batch_2=int(final.caption_batch.eq(2).sum()), generated=int(final.caption_status.eq('generated').sum()), token_limit=int(final.caption_status.eq('token_limit').sum()), empty_after_strip=int(final.text.eq('').sum()))
        frame = pd.read_parquet(OUT / 'corpus.parquet')
        assert set(frame.record_id) == set(final.record_id) and frame.record_id.is_unique
        final.to_parquet(OUT / 'final_captions.parquet', index=False)
        final_index = final.set_index('record_id')
        for column in ['text', 'caption_batch', 'caption_status']:
            frame[column] = frame.record_id.map(final_index[column])
        frame.to_parquet(OUT / 'corpus.parquet', index=False)
        shared.dump(OUT / 'caption_counts.json', counts)
        receipt.update(threads=8, counts=counts)

def stage_a():
    import shutil
    A2.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT / 'corpus.parquet', A2 / 'corpus.parquet')
    shared.OUT = A2
    try:
        shared.stage_a()
    finally:
        shared.OUT = OUT

def stage_b():
    shared.OUT = A2
    try:
        shared.stage_b(threads=8)
    finally:
        shared.OUT = OUT

def stage_cd():
    import joblib
    import numpy as np
    import pandas as pd
    from scripts.baseline_matrix_text import normalize_baseline_text
    from scripts.cluster_with_size_cap import build_graph
    protocol = shared.read(REPORT / 'settings.json')
    with shared.stage('CD') as receipt, shared.configure_threads(8):
        receipt['threads'] = 8
        frame = pd.read_parquet(OUT / 'corpus.parquet')
        candidates = pd.read_parquet(A2 / 'candidates.parquet')
        (image, text, sscd, hashes, simcse) = [np.load(A2 / (name + '.npy'), mmap_mode='r') for name in ['CLIP_image', 'CLIP_text', 'SSCD', 'pHash', 'SimCSE']]
        index = pd.Index(frame.record_id)
        (ca, cb) = (index.get_indexer(candidates.record_id_a), index.get_indexer(candidates.record_id_b))
        assert (ca >= 0).all() and (cb >= 0).all()
        assert not candidates.duplicated(['record_id_a', 'record_id_b']).any()
        same_product = frame.class_id.to_numpy()[ca] == frame.class_id.to_numpy()[cb]
        recalls = {}
        for partition in ['all', 'validation', 'test']:
            mask = np.ones(len(frame), bool) if partition == 'all' else frame.evaluation_partition.eq(partition).to_numpy()
            numerator = int((same_product & mask[ca] & mask[cb]).sum())
            sizes = frame[mask].groupby('class_id').size()
            denominator = sum((int(s) * (int(s) - 1) // 2 for s in sizes))
            recalls[partition] = dict(candidate_positive_pairs=numerator, positive_pairs=denominator, recall=numerator / denominator)
        shared.dump(OUT / 'candidate_recall.json', recalls)
        (costs, validation, selections, graph_info) = ({}, {}, {}, {})
        for method in ['classifier_LR', 'classifier_HGB', 'B2', 'B6']:
            (begin, cpu) = (time.perf_counter(), time.process_time())
            if method in {'classifier_LR', 'classifier_HGB'}:
                bundle = joblib.load(shared.visual_classifier / (method + '.joblib'))
                threshold = 2 / 3 if method == 'classifier_LR' else 1.0
                assert bundle['threshold'] == threshold
                (raw, calibrated) = shared.predict(frame, candidates, image, text, hashes, sscd, bundle)
            else:
                threshold = protocol['baselines'][method]['threshold']
                calibrated = np.empty(len(candidates), np.float64)
                model = joblib.load(shared.baseline_surfaces / 'B6.joblib') if method == 'B6' else None
                normalized = frame.text.map(normalize_baseline_text).to_numpy() if method == 'B6' else None
                for offset in range(0, len(candidates), 8192):
                    (aa, bb) = (ca[offset:offset + 8192], cb[offset:offset + 8192])
                    visual = np.sum(sscd[aa] * sscd[bb], axis=1, dtype=np.float32)
                    if method == 'B6':
                        language = np.sum(simcse[aa] * simcse[bb], axis=1, dtype=np.float32)
                        x = np.column_stack([visual, language, (normalized[aa] == normalized[bb]).astype(np.float32)])
                        calibrated[offset:offset + len(aa)] = model.predict_proba(x)[:, 1]
                    else:
                        calibrated[offset:offset + len(aa)] = visual
                raw = calibrated
            costs[method] = dict(C_wall_seconds=time.perf_counter() - begin, C_cpu_seconds=time.process_time() - cpu, threads=8)
            np.save(OUT / (method + '_raw.npy'), raw)
            np.save(OUT / (method + '_calibrated.npy'), calibrated)
            begin = time.perf_counter()
            (ids, a, b, selected) = shared.positive_edges(frame, candidates, raw, calibrated, threshold)
            costs[method]['edge_selection_sort_wall_seconds'] = time.perf_counter() - begin
            ordered = frame.set_index('record_id').loc[ids]
            groups = ordered.visual_group_id.to_numpy()
            val_mask = ordered.evaluation_partition.eq('validation').to_numpy()
            validation[method] = []
            caps = protocol['stage_D']['K'] if method in {'classifier_LR', 'classifier_HGB'} else ['inf']
            graph_info[method] = {}
            for cap in caps:
                begin = time.perf_counter()
                (labels, _, _, graph_counts) = build_graph(len(frame), a, b, cap)
                seconds = time.perf_counter() - begin
                np.save(OUT / f'{method}_K{cap}_components.npy', labels)
                val = shared.graph_metrics(labels[val_mask], groups[val_mask])
                validation[method].append(dict(K=cap, **val))
                graph_info[method][str(cap)] = dict(D_union_find_wall_seconds=seconds, graph_counts=graph_counts)
            if method in {'classifier_LR', 'classifier_HGB'}:
                selections[method] = select_k(validation[method])
            shared.dump(OUT / 'validation_grid.json', validation)
            print('validation complete', method, selections.get(method), flush=True)
        shared.dump(OUT / 'selection_frozen.json', dict(selection_partition='validation', choices=selections, rule='exact F1 maximum; smaller K on ties'))
        rows = []
        for method in ['classifier_LR', 'classifier_HGB', 'B2', 'B6']:
            raw = np.load(OUT / (method + '_raw.npy'), mmap_mode='r')
            calibrated = np.load(OUT / (method + '_calibrated.npy'), mmap_mode='r')
            threshold = (2 / 3 if method == 'classifier_LR' else 1.0) if method in {'classifier_LR', 'classifier_HGB'} else protocol['baselines'][method]['threshold']
            (ids, a, b, selected) = shared.positive_edges(frame, candidates, raw, calibrated, threshold)
            ordered = frame.set_index('record_id').loc[ids]
            groups = ordered.visual_group_id.to_numpy()
            for (cap_string, info) in graph_info[method].items():
                cap = 'inf' if cap_string == 'inf' else int(cap_string)
                labels = np.load(OUT / f'{method}_K{cap}_components.npy')
                projections = {}
                for partition in ['all', 'validation', 'test']:
                    mask = np.ones(len(frame), bool) if partition == 'all' else ordered.evaluation_partition.eq(partition).to_numpy()
                    projections[partition] = dict(S=int((mask[a] & mask[b]).sum()), **shared.graph_metrics(labels[mask], groups[mask]))
                if cap == 5:
                    stats = projections['all']
                    assert stats['max_cluster'] <= 5 and stats['U'] <= 2 * len(frame)
                row = dict(method=method, K=cap, selected=selections.get(method) == cap, projections=projections, **info)
                if method in {'classifier_LR', 'classifier_HGB'}:
                    stats = projections['all']
                    row.update(pair_vlm_gpu_hours=stats['U'] * 0.437 / 3600, record_vlm_gpu_hours=stats['records_in_clusters_size_ge2'] * 0.0661 / 3600)
                rows.append(row)
        retained_ids = sorted(frame[frame.caption_status.eq('token_limit')].record_id)
        cluster_sizes = {record_id: {} for record_id in retained_ids}
        for cap in dict.fromkeys([5, selections['classifier_LR'], 'inf']):
            labels = np.load(OUT / f'classifier_LR_K{cap}_components.npy')
            sizes = Counter(map(int, labels))
            for record_id in retained_ids:
                position = ids.get_loc(record_id)
                cluster_sizes[record_id][str(cap)] = sizes[int(labels[position])]
        token_limit_sizes = dict(method='classifier_LR', selected_K=selections['classifier_LR'], scope='Full global component size, including all partitions', records=cluster_sizes)
        shared.dump(OUT / 'token_limit_cluster_sizes.json', token_limit_sizes)
        shared.dump(OUT / 'graph_results.json', dict(rows=rows, costs=costs, selections=selections, candidate_recall=recalls, token_limit_cluster_sizes=token_limit_sizes))
        receipt['rows'] = len(rows)

def evaluation_partition(superclass):
    return 'validation' if int(superclass) in (1, 10) else 'test' if int(superclass) == 7 else 'train'

def retained_path(path, benchmark, multiview):
    return path not in benchmark and path not in multiview

def select_k(validation_rows):
    return max(validation_rows, key=lambda r: (Fraction(2 * r['tp'], 2 * r['tp'] + r['fp'] + r['fn']) if 2 * r['tp'] + r['fp'] + r['fn'] else Fraction(0), -float(r['K'])))['K']

def corpus():
    import pandas as pd
    from zipfile import ZipFile
    protocol = shared.read(REPORT / 'settings.json')
    sources = protocol['sources']
    with shared.stage('corpus') as stage:
        archive_path = Path(sources['archive'])
        inventory = pd.read_csv(sources['exclusions'])
        benchmark_rows = inventory[inventory.source.eq('stanford_online_products')]
        benchmark = set(benchmark_rows.image_path.str.removeprefix('raw/stanford_online_products/'))
        mv_rows = pd.read_csv(sources['MV_exclusions'])
        multiview = set(mv_rows.sop_path)
        assert len(benchmark_rows) == len(benchmark) == 14000
        assert len(mv_rows) == len(multiview) == 8550
        assert not benchmark & multiview
        authority = pd.read_parquet(shared.RES / 'datasets/VL-DedupBench-50K_release/data/manifests/vl_dedupbench_50k_manifest.parquet')
        mapping = authority[authority.group_id.str.startswith('sop_super_')][['group_id', 'partition']].drop_duplicates()
        assert len(mapping) == 12
        assert all((evaluation_partition(int(r.group_id.rsplit('_', 1)[1])) == r.partition for r in mapping.itertuples()))
        with ZipFile(archive_path) as archive:
            names = [n for n in archive.namelist() if n.endswith('/Ebay_info.txt') or n == 'Ebay_info.txt']
            assert len(names) == 1, names
            metadata = archive.read(names[0]).decode('utf-8')
            (OUT / 'Ebay_info.txt').write_text(metadata)
            (rows, seen) = ([], set())
            for line in metadata.splitlines()[1:]:
                if not line.strip():
                    continue
                (image_id, class_id, super_id, path) = line.split()
                assert path not in seen
                seen.add(path)
                if retained_path(path, benchmark, multiview):
                    rows.append(dict(record_id=f'sop_full:{int(image_id):06d}', image_id=int(image_id), class_id=int(class_id), super_class_id=int(super_id), sop_path=path, visual_group_id=f'sop_product:{class_id}', split='sop_global', partition='sop_global', evaluation_partition=evaluation_partition(super_id), text='', image_path=str(OUT / 'images' / path)))
            rows.sort(key=lambda r: r['image_id'])
            frame = pd.DataFrame(rows)
            sizes = frame.groupby('class_id').size()

            def count(local):
                groups = local.groupby('class_id').size()
                return dict(records=len(local), products=len(groups), positive_pairs=sum((int(s) * (int(s) - 1) // 2 for s in groups)))
            counts = dict(**count(frame), singleton_products=int((sizes == 1).sum()), max_product_size=int(sizes.max()), all_record_pairs=len(frame) * (len(frame) - 1) // 2, partitions={p: count(frame[frame.evaluation_partition.eq(p)]) for p in ('validation', 'test', 'train')}, group_size_counts={str(k): int(v) for (k, v) in sorted(Counter(sizes).items())}, K5_split_groups_lower_bound=int((sizes > 5).sum()), K5_FN_lower_bound=sum((int(s) * (int(s) - 1) // 2 - int(s) // 5 * 10 - int(s) % 5 * (int(s) % 5 - 1) // 2 for s in sizes)))
            unmatched = sorted((benchmark | multiview) - seen)
            retained_overlap = set(frame.sop_path) & (benchmark | multiview)
            frame.to_parquet(OUT / 'corpus.parquet', index=False)
            prefix = names[0][:-len('Ebay_info.txt')]
            for r in frame.itertuples():
                dest = Path(r.image_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(prefix + r.sop_path) as source, dest.open('wb') as output:
                    import shutil
                    shutil.copyfileobj(source, output)
            stage.update(records=len(frame), extracted_images=len(frame))
        shared.dump(OUT / 'corpus_counts.json', counts)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['corpus', 'captions', 'finalize_captions', 'A', 'B', 'CD'])
    args = parser.parse_args()
    {'corpus': corpus, 'captions': captions, 'finalize_captions': finalize_captions, 'A': stage_a, 'B': stage_b, 'CD': stage_cd}[args.phase]()
if __name__ == '__main__':
    main()
