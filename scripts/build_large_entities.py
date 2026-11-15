import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
os.environ['GRAPH_CPU_THREADS'] = '4'
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from scripts import build_candidates as candidate
from scripts.pair_features import FEATURES, ObservableFeatures
from scripts.cluster_with_size_cap import GRID, build_graph, edge_order
RES = RESOURCES
DATA = RES / 'datasets/benchmark'
SOURCE = RES / 'runs/pair_features'
OUT = RES / 'runs/large_entities'
REPORT = ROOT / str(REPORTS / 'large_entities')
MODEL_PATHS = {'HGB': SOURCE / 'hist_gradient_boosting.joblib', 'LR10': RES / 'runs/ten_feature_classifier/logistic.joblib'}
THRESHOLDS = {'HGB': 1.0, 'LR10': 0.25}
SEED = 20_261_029

def now():
    return datetime.now(timezone.utc).isoformat()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def read(path):
    return json.loads(Path(path).read_text())

def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]

def write_rows(path, rows):
    with Path(path).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

@contextmanager
def stage(name, gpu=False):
    (started, cpu) = (time.perf_counter(), time.process_time())
    item = {}
    try:
        yield item
    finally:
        item.update(wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu)
        dump(OUT / (name + '_stage.json'), item)

def sample_groups(lineage):
    groups = defaultdict(list)
    for row in lineage:
        groups[row['visual_group_id']].append(row)
    rng = np.random.Generator(np.random.PCG64(SEED))
    (result, counts) = ([], {})
    for (pi, partition) in enumerate(['validation', 'test']):
        eligible = sorted((g for (g, rows) in groups.items() if len(rows) == 5 and all((r['source_dataset'] == 'coco2017' and r['original_level'] in {'L0', 'L2'} for r in rows)) and ({r['partition'] for r in rows} == {partition})))
        counts[partition] = len(eligible)
        if len(eligible) < 300:
            raise ValueError(f'{partition}: only {len(eligible)} eligible groups')
        for (rank, gid) in enumerate(rng.choice(eligible, 300, replace=False).tolist()):
            result.append(dict(visual_group_id=gid, partition=partition, partition_index=pi, draw_index=rank, target_size=[10, 20, 40][rank // 100], members=sorted(groups[gid], key=lambda r: r['record_id'])))
    return (result, counts)

def caption_sources(members, public):
    (seen, sources) = (set(), [])
    for row in sorted(members, key=lambda r: r['record_id']):
        text = public[row['record_id']]['text']
        if text not in seen:
            seen.add(text)
            sources.append(row['record_id'])
    return sources

def render_copy(source, seed, seen):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    (w, h) = source.size
    (cw, ch) = (max(1, round(0.88 * w)), max(1, round(0.88 * h)))
    for attempt in range(1, 101):
        (dx, dy) = [int(rng.integers(-int(0.03 * n), int(0.03 * n) + 1)) for n in [w, h]]
        (x, y) = (np.clip((w - cw) // 2 + dx, 0, w - cw), np.clip((h - ch) // 2 + dy, 0, h - ch))
        box = [int(x), int(y), int(x + cw), int(y + ch)]
        brightness = float(rng.uniform(0.92, 1.08))
        image = ImageEnhance.Brightness(source.crop(box)).enhance(brightness)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=70)
        payload = buffer.getvalue()
        if payload not in seen:
            seen.add(payload)
            return (payload, dict(seed=seed, crop_box=box, brightness=brightness, jpeg_quality=70, attempts=attempt))
    raise ValueError('Could not construct distinct JPEG bytes in 100 deterministic draws')

def construct():
    import imagehash
    protocol = read(REPORT / 'settings.json')
    with stage('construct') as receipt:
        lineage = read_rows(DATA / 'private/lineage.jsonl')
        public = {r['record_id']: r for r in read_rows(DATA / 'public_records.jsonl')}
        frame = pd.read_parquet(SOURCE / 'public_manifest.parquet').sort_values('record_id').reset_index(drop=True)
        assert len(frame) == 50000 and frame.record_id.is_unique
        for r in frame.itertuples():
            assert public[r.record_id]['text'] == r.text
            assert public[r.record_id]['images'][0]['image_path'] == r.image_path
        (groups, counts) = sample_groups(lineage)
        assert counts == protocol['sampling']['eligible_counts_independently_verified']
        dump(OUT / 'settings.json', protocol)
        dump(OUT / 'selected_groups.json', groups)
        (new, composition, distinct) = ([], defaultdict(Counter), defaultdict(Counter))
        image_dir = OUT / 'images'
        image_dir.mkdir(exist_ok=False)
        for group in groups:
            members = group['members']
            levels = {r['original_level'] for r in members}
            key = 'all_L2' if levels == {'L2'} else 'contains_L0'
            composition[group['partition']][key] += 1
            sources = caption_sources(members, public)
            assert len(sources) in {4, 5}
            distinct[group['partition']][str(len(sources))] += 1
            paths = {public[r['record_id']]['images'][0]['image_path'] for r in members}
            assert len(paths) == 1
            source_path = next(iter(paths))
            with Image.open(source_path) as opened:
                source = ImageOps.exif_transpose(opened).convert('RGB')
            seen = set()
            for j in range(group['target_size'] - 5):
                rid = f"large_entities_{group['partition']}_{group['draw_index']:03d}_{j:02d}"
                destination = image_dir / f'{rid}.jpg'
                seed = [SEED, group['partition_index'], group['draw_index'], j]
                (payload, params) = render_copy(source, seed, seen)
                destination.write_bytes(payload)
                caption_id = sources[j % len(sources)]
                with Image.open(io.BytesIO(payload)) as rendered:
                    phash = int(str(imagehash.phash(rendered)), 16)
                new.append(dict(record_id=rid, partition=group['partition'], visual_group_id=group['visual_group_id'], source_dataset='coco2017', text=public[caption_id]['text'], image_path=str(destination), caption_source_record_id=caption_id, source_image_path=source_path, target_size=group['target_size'], draw_index=group['draw_index'], copy_j=j, phash_int=phash, **params))
            source.close()
        assert len(new) == 11000 and Counter((r['partition'] for r in new)) == {'validation': 5500, 'test': 5500}
        assert dict(composition) == protocol['sampling']['selected_composition']
        assert dict(distinct) == protocol['sampling']['selected_distinct_caption_counts']
        write_rows(OUT / 'new_records.jsonl', new)
        pd.DataFrame(new)[['record_id', 'partition', 'visual_group_id', 'target_size', 'caption_source_record_id', 'image_path']].to_csv(OUT / 'new_records.csv', index=False)
        pd.concat([frame, pd.DataFrame(new)[frame.columns]], ignore_index=True).sort_values('record_id').reset_index(drop=True).to_parquet(OUT / 'public_manifest.parquet', index=False)
        meta = [{k: r[k] for k in ['record_id', 'partition', 'visual_group_id']} | {'is_new': False} for r in lineage]
        meta += [{k: r[k] for k in ['record_id', 'partition', 'visual_group_id']} | {'is_new': True} for r in new]
        pd.DataFrame(meta).sort_values('record_id').to_parquet(OUT / 'lineage.parquet', index=False)
        receipt.update(eligible_counts=counts, selected_composition=dict(composition), distinct_caption_counts=dict(distinct), new_records=len(new), total_records=61000, duplicate_render_redraws=sum((r['attempts'] - 1 for r in new)), label_access=False)

def encode():
    with stage('encode', gpu=True) as receipt:
        receipt['nvidia_smi_before_load'] = subprocess.check_output(['nvidia-smi']).decode()
        dump(OUT / 'encode_receipt.json', receipt)
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from scripts.extract_features import clip_feature_tensor
        torch.set_num_threads(4)
        checkpoint = Path(read(REPORT / 'settings.json')['source']['clip_local'])
        model = CLIPModel.from_pretrained(str(checkpoint), local_files_only=True).float().to('cuda').eval()
        processor = CLIPProcessor.from_pretrained(str(checkpoint), local_files_only=True)
        records = read_rows(OUT / 'new_records.jsonl')
        batches = []
        with torch.inference_mode():
            for start in range(0, len(records), 32):
                images = []
                for row in records[start:start + 32]:
                    with Image.open(row['image_path']) as opened:
                        images.append(opened.convert('RGB'))
                inputs = processor(images=images, return_tensors='pt')
                features = clip_feature_tensor(model.get_image_features(**{k: v.to('cuda') for (k, v) in inputs.items()})).float()
                batches.append(torch.nn.functional.normalize(features, dim=1).cpu().numpy())
                for im in images:
                    im.close()
                if start % 1024 == 0:
                    print(f'CLIP {start}/{len(records)}', flush=True)
        embeddings = np.concatenate(batches)
        assert embeddings.shape == (11000, 768) and np.isfinite(embeddings).all()
        np.save(OUT / 'new_image_embeddings.npy', embeddings)
        receipt.update(rows=11000, model=str(checkpoint), dtype='float32', peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), label_access=False)

def features():
    with stage('features') as receipt:
        original = pd.read_parquet(SOURCE / 'public_manifest.parquet')
        frame = pd.read_parquet(OUT / 'public_manifest.parquet')
        new = read_rows(OUT / 'new_records.jsonl')
        indices = pd.Index(original.record_id)
        initial_to_new = pd.Index(frame.record_id).get_indexer(original.record_id)
        new_to_all = pd.Index(frame.record_id).get_indexer([r['record_id'] for r in new])
        feature_root = OUT / 'features'
        (feature_root / 'clip').mkdir(parents=True)
        (feature_root / 'phash').mkdir()
        for name in ['image', 'text']:
            reference = np.load(SOURCE / f'features/clip/{name}_emb.npy')
            assert reference.shape == (50000, 768)
            full = np.empty((61000, 768), dtype=np.float32)
            full[initial_to_new] = reference
            full[new_to_all] = np.load(OUT / 'new_image_embeddings.npy') if name == 'image' else reference[indices.get_indexer([r['caption_source_record_id'] for r in new])]
            assert np.isfinite(full).all()
            np.save(feature_root / f'clip/{name}_emb.npy', full)
        initial_phash = pd.read_parquet(SOURCE / 'features/phash/phash.parquet').set_index('record_id')
        phashes = np.empty(61000, dtype=np.uint64)
        phashes[initial_to_new] = initial_phash.loc[original.record_id, 'phash_int'].to_numpy(dtype=np.uint64)
        phashes[new_to_all] = np.array([r['phash_int'] for r in new], dtype=np.uint64)
        pd.DataFrame({'record_id': frame.record_id, 'phash_int': phashes}).to_parquet(feature_root / 'phash/phash.parquet', index=False)
        receipt.update(records=61000, new_images_encoded=11000, text_embeddings_reused=61000, label_access=False)

def retrieve():
    with stage('retrieve') as receipt:
        frame = pd.read_parquet(OUT / 'public_manifest.parquet')
        frame['partition'] = 'global_large_entities'
        (edges, counts) = ({}, {})
        counts['phash'] = candidate.build_phash_edges(frame, OUT / 'features', 2, edges)
        for (name, k) in [('image', 30), ('text', 10)]:
            vectors = candidate.load_embeddings(OUT / f'features/clip/{name}_emb.npy', len(frame))
            channel = 'clip_' + name
            counts[channel] = candidate.build_embedding_edges(vectors, channel, channel + '_score', k, None, edges, frame.partition.to_numpy())
            print(f'{channel}: union {len(edges)}', flush=True)
        counts['text_exact'] = candidate.build_text_exact_edges(frame, 20, edges)
        pairs = candidate.edges_to_frame(edges, frame).reset_index(drop=True)
        pairs.to_parquet(OUT / 'candidates.parquet', index=False)
        receipt.update(records=61000, candidates=len(pairs), channel_new_edges=counts, partition_filter=False, label_access=False)

def score_graphs():
    import joblib
    with stage('graphs') as receipt:
        frame = pd.read_parquet(OUT / 'public_manifest.parquet')
        pairs = pd.read_parquet(OUT / 'candidates.parquet', columns=['record_id_a', 'record_id_b'])
        ids = pd.Index(frame.record_id)
        (a, b) = [ids.get_indexer(pairs[c]) for c in pairs.columns]
        assert min(a.min(), b.min()) >= 0
        featurizer = ObservableFeatures(frame, np.load(OUT / 'features/clip/image_emb.npy'), np.load(OUT / 'features/clip/text_emb.npy'), pd.read_parquet(OUT / 'features/phash/phash.parquet').phash_int.to_numpy(dtype=np.uint64))
        bundles = {m: joblib.load(p) for (m, p) in MODEL_PATHS.items()}
        raw = {m: np.empty(len(pairs), dtype=np.float64) for m in bundles}
        scores = {m: np.empty(len(pairs), dtype=np.float64) for m in bundles}
        calls = Counter()
        for start in range(0, len(pairs), 32768):
            stop = min(start + 32768, len(pairs))
            x = featurizer.transform(pairs.iloc[start:stop])
            for (model, bundle) in bundles.items():
                width = 11 if model == 'HGB' else 10
                assert bundle['features'] == FEATURES[:width]
                cols = [list(bundle['model'].classes_).index(c) for c in ['duplicate', 'keep_augment_candidate']]
                raw[model][start:stop] = bundle['model'].predict_proba(x[:, :width])[:, cols].sum(axis=1)
                scores[model][start:stop] = bundle['calibrator'].predict(raw[model][start:stop])
                calls[model] += 1
            if start % (32768 * 10) == 0:
                print(f'scores {start}/{len(pairs)}', flush=True)
        cells = []
        for model in bundles:
            assert np.isfinite(raw[model]).all() and np.isfinite(scores[model]).all()
            np.save(OUT / f'{model}_raw.npy', raw[model])
            np.save(OUT / f'{model}_calibrated.npy', scores[model])
            positive = np.flatnonzero(scores[model] >= THRESHOLDS[model])
            order = edge_order(positive, scores[model], a, b, raw[model][positive])
            np.save(OUT / f'{model}_ordered_positive_rows.npy', order)
            for cap in GRID:
                (labels, kept, rejected, counts) = build_graph(len(frame), a[order], b[order], cap)
                folder = OUT / model / f'K_{cap}'
                folder.mkdir(parents=True)
                np.save(folder / 'components.npy', labels)
                np.save(folder / 'retained_mask.npy', kept)
                np.save(folder / 'rejected_sizes.npy', rejected)
                sizes = np.unique(labels, return_counts=True)[1]
                if cap == 'inf':
                    assert kept.all() and counts['cap_rejected_edges'] == 0
                cell = dict(model=model, k=cap, positive_edges=len(order), components=len(sizes), singletons=int((sizes == 1).sum()), max_component=int(sizes.max()), artifacts=str(folder), **counts)
                cells.append(cell)
                print(f'{model} K={cap}: max={sizes.max()}', flush=True)
        dump(OUT / 'graphs_frozen.json', dict(frozen_at_utc=now(), cells=cells, test_labels_opened=False, models={m: str(p) for (m, p) in MODEL_PATHS.items()}))
        receipt.update(candidates=len(pairs), graphs=len(cells), predict_calls=dict(calls), model_rows={m: len(pairs) for m in bundles}, fit_calls=0, label_access=False)

def load_reference(partition):
    if partition == 'test':
        for model in MODEL_PATHS:
            assert (OUT / f'selection_{model}.json').exists()
    pattern = re.compile('"partition"\\s*:\\s*"' + partition + '"')
    rows = []
    with (DATA / 'sealed/relation_applicability.jsonl').open() as stream:
        for line in stream:
            if pattern.search(line):
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def confusion(truth, prediction):
    (truth, prediction) = (np.asarray(truth, dtype=bool), np.asarray(prediction, dtype=bool))
    (tp, fp) = (int((truth & prediction).sum()), int((~truth & prediction).sum()))
    (fn, tn) = (int((truth & ~prediction).sum()), int((~truth & ~prediction).sum()))
    denominator = 2 * tp + fp + fn
    return dict(n=len(truth), tp=tp, fp=fp, fn=fn, tn=tn, f1=2 * tp / denominator if denominator else None)

def expanded_pairs(groups, ids, partition):
    by_size = defaultdict(lambda : [[], []])
    for group in groups:
        if group['partition'] != partition:
            continue
        rids = [r['record_id'] for r in group['members']]
        rids += [f"large_entities_{partition}_{group['draw_index']:03d}_{j:02d}" for j in range(group['target_size'] - 5)]
        ix = ids.get_indexer(rids)
        assert (ix >= 0).all()
        (x, y) = np.triu_indices(len(ix), 1)
        by_size[group['target_size']][0].extend(ix[x])
        by_size[group['target_size']][1].extend(ix[y])
    return {size: tuple((np.array(x, dtype=int) for x in ends)) for (size, ends) in by_size.items()}

def cross_group_new_pairs(labels, groups, is_new):
    (total, reference) = (Counter(), Counter())
    (by_group, initial_group) = (Counter(), Counter())
    for (label, group, new) in zip(labels, groups, is_new):
        total[int(label)] += 1
        by_group[int(label), group] += 1
        if not new:
            reference[int(label)] += 1
            initial_group[int(label), group] += 1
    choose = lambda counts: sum((n * (n - 1) // 2 for n in counts.values()))
    return choose(total) - choose(by_group) - choose(reference) + choose(initial_group)

def evaluate():
    with stage('evaluate') as receipt:
        frozen = read(OUT / 'graphs_frozen.json')
        frame = pd.read_parquet(OUT / 'public_manifest.parquet')
        ids = pd.Index(frame.record_id)
        meta = pd.read_parquet(OUT / 'lineage.parquet').set_index('record_id').loc[ids]
        groups = read(OUT / 'selected_groups.json')
        results = dict(run_id=OUT.name, selection={}, validation=[], test=[], original_expanded_overlap={}, graph_freeze_at=frozen['frozen_at_utc'])
        for partition in ['validation', 'test']:
            opened_at = now()
            pairs = load_reference(partition)
            results[partition + '_first_opened_at'] = opened_at
            known = pairs.visual_relation_mask.to_numpy(dtype=bool)
            truth = pairs.visual_relation.eq('same').to_numpy()
            expected = (9766, 8959, 807) if partition == 'validation' else (9494, 8701, 793)
            assert (int(known.sum()), int((known & truth).sum()), int((known & ~truth).sum())) == expected
            (a, b) = [ids.get_indexer(pairs[c]) for c in ['record_id_a', 'record_id_b']]
            assert min(a.min(), b.min()) >= 0
            assert (meta.partition.to_numpy()[a] == partition).all()
            assert (meta.partition.to_numpy()[b] == partition).all()
            expanded = expanded_pairs(groups, ids, partition)
            assert {k: len(v[0]) for (k, v) in expanded.items()} == {10: 4500, 20: 19000, 40: 78000}
            codes = set()
            for (x, y) in expanded.values():
                codes.update((np.minimum(x, y).astype(np.int64) * len(ids) + np.maximum(x, y)).tolist())
            overlap = np.array([int(min(x, y)) * len(ids) + int(max(x, y)) in codes for (x, y) in zip(a, b)])
            assert not (overlap & known & ~truth).any()
            results['original_expanded_overlap'][partition] = int((overlap & known).sum())
            for cell in frozen['cells']:
                labels = np.load(Path(cell['artifacts']) / 'components.npy')
                connected = labels[a] == labels[b]
                row = dict(model=cell['model'], k=cell['k'], original=confusion(truth[known], connected[known]), unknown_n=int((~known).sum()), unknown_connected=int((~known & connected).sum()))
                by_size = {str(size): dict(connected=int((labels[x] == labels[y]).sum()), n=len(x)) for (size, (x, y)) in expanded.items()}
                for item in by_size.values():
                    item['proportion'] = item['connected'] / item['n']
                row['expanded'] = by_size
                if partition == 'validation':
                    added_tp = sum((r['connected'] for r in by_size.values()))
                    combined = dict(row['original'])
                    combined.update(n=111266, tp=combined['tp'] + added_tp, fn=combined['fn'] + 101500 - added_tp)
                    combined['f1'] = 2 * combined['tp'] / (2 * combined['tp'] + combined['fp'] + combined['fn'])
                    row['combined'] = combined
                else:
                    row['cross_visual_group_with_new'] = {}
                    for scope in ['test', 'validation', 'global']:
                        mask = np.ones(len(meta), dtype=bool) if scope == 'global' else meta.partition.eq(scope).to_numpy()
                        row['cross_visual_group_with_new'][scope] = cross_group_new_pairs(labels[mask], meta.visual_group_id.to_numpy()[mask], meta.is_new.to_numpy()[mask])
                results[partition].append(row)
            if partition == 'validation':
                for model in MODEL_PATHS:
                    rows = [r for r in results['validation'] if r['model'] == model]
                    key = lambda r: (Fraction(2 * r['combined']['tp'], 2 * r['combined']['tp'] + r['combined']['fp'] + r['combined']['fn']), float('inf') if r['k'] == 'inf' else r['k'])
                    selected = max(rows, key=key)
                    selection = dict(model=model, k=selected['k'], frozen_at_utc=now(), exact_f1=str(key(selected)[0]), basis=selected['combined'], validation_cells=16, graph_freeze_at=frozen['frozen_at_utc'])
                    dump(OUT / f'selection_{model}.json', selection)
                    results['selection'][model] = selection
                dump(OUT / 'validation_frozen.json', results)
        results['completed_at_utc'] = now()
        dump(OUT / 'results.json', results)
        dump(REPORT / 'results.json', results)
        rows = []
        for partition in ['validation', 'test']:
            for row in results[partition]:
                flat = dict(partition=partition, model=row['model'], k=row['k'], **row['original'])
                for (size, item) in row['expanded'].items():
                    flat.update({f'size_{size}_{key}': value for (key, value) in item.items()})
                if partition == 'validation':
                    flat.update({'selection_' + k: v for (k, v) in row['combined'].items()})
                else:
                    flat.update({'cross_new_' + k: v for (k, v) in row['cross_visual_group_with_new'].items()})
                rows.append(flat)
        pd.DataFrame(rows).to_csv(REPORT / 'sensitivity.csv', index=False)
        pd.DataFrame(frozen['cells']).drop(columns='artifacts').to_csv(REPORT / 'structure.csv', index=False)
        receipt.update(validation_cells=32, test_cells=32, known_denominators={'validation': 9766, 'test': 9494}, selection={m: r['k'] for (m, r) in results['selection'].items()}, label_access=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['construct', 'encode', 'features', 'retrieve', 'graphs', 'evaluate'])
    args = parser.parse_args()
    {'construct': construct, 'encode': encode, 'features': features, 'retrieve': retrieve, 'graphs': score_graphs, 'evaluate': evaluate}[args.stage]()
if __name__ == '__main__':
    main()
