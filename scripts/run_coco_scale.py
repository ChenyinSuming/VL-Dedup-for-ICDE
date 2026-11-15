from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zipfile import ZipFile
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
RES = RESOURCES
OUT = RES / 'runs/coco_scale'
REPORT = ROOT / str(REPORTS / 'coco_scale')
SOURCE = RES / 'runs/pair_features'
baseline_surfaces = RES / 'runs/baselines_surfaces'
visual_classifier = RES / 'runs/visual_classifier'
DATA = RES / 'datasets/benchmark'

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def read(path):
    return json.loads(Path(path).read_text())

def load_exclusions(path):
    result = set()
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['source'] != 'coco2017':
                continue
            split = re.search('/(train2017|val2017)/', row['image_path']).group(1)
            image_id = int(row['source_asset_id'].split(':')[0])
            assert int(Path(row['image_path']).stem) == image_id
            result.add((split, image_id))
    return result

def keep_annotation(split, annotation, exclusions):
    return (split, int(annotation['image_id'])) not in exclusions

def build_corpus():
    import pandas as pd
    protocol = read(REPORT / 'settings.json')
    with stage('corpus'):
        exclusions = load_exclusions(protocol['sources']['exclusions'])
        counts = dict(source_images={}, source_captions={}, excluded_images=dict(Counter((s for (s, _) in exclusions))))
        counts['excluded_images']['total'] = len(exclusions)
        (rows, sizes, unmatched) = ([], Counter(), set(exclusions))
        with ZipFile(f"{RESOURCES / 'datasets/COCO2017'}/annotations_trainval2017.zip") as archive:
            for split in ('train2017', 'val2017'):
                name = f'annotations/captions_{split}.json'
                raw = archive.read(name)
                path = OUT / 'coco2017_full' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                data = json.loads(raw)
                counts['source_images'][split] = len(data['images'])
                counts['source_captions'][split] = len(data['annotations'])
                images = {x['id']: x['file_name'] for x in data['images']}
                unmatched -= {(split, i) for i in images}
                for annotation in sorted(data['annotations'], key=lambda x: x['id']):
                    if not keep_annotation(split, annotation, exclusions):
                        continue
                    (image_id, ann_id) = (annotation['image_id'], annotation['id'])
                    sizes[split, image_id] += 1
                    rows.append(dict(record_id=f'coco2017:{split}:{image_id}:{ann_id}', source='coco2017', split=split, image_id=image_id, annotation_id=ann_id, image_path=str(OUT / 'coco2017_full' / split / images[image_id]), text=annotation['caption'], visual_group_id=f'{split}:{image_id}', partition='coco_global'))
        counts.update(images=len(sizes), records=len(rows), group_size_counts={str(k): v for (k, v) in sorted(Counter(sizes.values()).items())}, positive_pairs=sum((s * (s - 1) // 2 for s in sizes.values())), all_record_pairs=len(rows) * (len(rows) - 1) // 2, K5_split_groups_lower_bound=sum((s > 5 for s in sizes.values())), K5_FN_lower_bound=sum((s * (s - 1) // 2 - s // 5 * 10 - s % 5 * (s % 5 - 1) // 2 for s in sizes.values())))
        pd.DataFrame(rows).to_parquet(OUT / 'corpus.parquet', index=False)
        dump(OUT / 'corpus_counts.json', counts)

@contextmanager
def stage(name, gpu=False):
    (started, cpu) = (time.perf_counter(), time.process_time())
    item = {}
    try:
        yield item
    finally:
        item.update(wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu)
        dump(OUT / (name + '_stage.json'), item)

def extract_images():
    import pandas as pd
    import shutil
    with stage('image_extraction') as receipt:
        images = pd.read_parquet(OUT / 'corpus.parquet', columns=['split', 'image_id', 'image_path']).drop_duplicates(['split', 'image_id'])
        (count, size) = (0, 0)
        for split in ('train2017', 'val2017'):
            with ZipFile(f"{RESOURCES / 'datasets/COCO2017'}/" + split + '.zip') as archive:
                wanted = {split + '/' + Path(p).name: Path(p) for p in images[images.split.eq(split)].image_path}
                for info in archive.infolist():
                    if info.filename not in wanted:
                        continue
                    path = wanted[info.filename]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, path.open('wb') as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    count += 1
                    size += info.file_size
                    if count % 10000 == 0:
                        print('extracted', count, flush=True)
        receipt.update(images=count, bytes=size, traversal='ZIP entry order')
        assert count == 116091

def stage_a():
    import numpy as np
    import pandas as pd
    import torch
    from scripts import extract_features as extract
    from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoTokenizer
    import imagehash
    with stage('A', gpu=True) as receipt, torch.inference_mode():
        stage_begin = time.perf_counter()
        hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'], text=True)
        print('GPU before model loading:', hardware, flush=True)
        receipt.update(gpu=hardware, threads=8, torch=torch.__version__)
        torch.set_num_threads(8)
        frame = pd.read_parquet(OUT / 'corpus.parquet')
        images = frame.drop_duplicates(['split', 'image_id']).reset_index(drop=True)
        image_positions = pd.Index(images.image_path).get_indexer(frame.image_path)
        assert (image_positions >= 0).all()
        pilot_images = images.iloc[:2000]
        pilot_mask = image_positions < 2000
        pilot_records = frame[pilot_mask]
        (pilot, timings) = ({}, {})

        def encode(name, data, model=None, processor=None):
            n = len(data)
            if name == 'CLIP_image':
                result = np.empty((n, 768), np.float32)
                for offset in range(0, n, 64):
                    (batch, ok, errors) = extract.load_clip_images(data.iloc[offset:offset + 64], DATA)
                    assert all(ok), errors
                    inputs = {k: v.to('cuda') for (k, v) in processor(images=batch, return_tensors='pt').items()}
                    values = extract.clip_feature_tensor(model.get_image_features(**inputs)).float()
                    result[offset:offset + len(values)] = torch.nn.functional.normalize(values, dim=1).cpu().numpy()
                    for image in batch:
                        image.close()
            elif name in {'CLIP_text', 'SimCSE'}:
                batch_size = 256 if name == 'CLIP_text' else 128
                result = np.empty((n, 768), np.float32)
                for offset in range(0, n, batch_size):
                    inputs = processor(text=data.text.iloc[offset:offset + batch_size].astype(str).tolist(), return_tensors='pt', padding=True, truncation=True, max_length=model.config.text_config.max_position_embeddings if name == 'CLIP_text' else 128)
                    inputs = {k: v.to('cuda') for (k, v) in inputs.items()}
                    values = extract.clip_feature_tensor(model.get_text_features(**inputs)).float() if name == 'CLIP_text' else model(**inputs).pooler_output.float()
                    result[offset:offset + len(values)] = torch.nn.functional.normalize(values, dim=1).cpu().numpy()
                    if n > 20000 and offset % (batch_size * 100) == 0:
                        print(name, offset, '/', n, flush=True)
            elif name == 'SSCD':
                result = np.empty((n, 512), np.float32)
                offset = 0
                for (batch, ok) in extract.image_loader(data, DATA, 288, 32, 0):
                    assert bool(ok.all())
                    values = torch.nn.functional.normalize(model(batch.to('cuda')).float(), dim=1).cpu().numpy()
                    result[offset:offset + len(values)] = values
                    offset += len(values)
            elif name == 'pHash':
                values = []
                for path in data.image_path:
                    (image, ok, error) = extract.load_rgb_image(extract.resolve_image_path(DATA, path), 224)
                    assert ok, error
                    values.append(int(str(imagehash.phash(image)), 16))
                    image.close()
                result = np.array(values, dtype=np.uint64)
            return result

        def run(name, data, model=None, processor=None):
            gpu = name != 'pHash'
            if gpu:
                torch.cuda.synchronize()
            begin = time.perf_counter()
            values = encode(name, data, model, processor)
            if gpu:
                torch.cuda.synchronize()
            return (values, time.perf_counter() - begin)

        def load(kind):
            begin = time.perf_counter()
            if kind == 'CLIP':
                path = RES / 'models/clip-vit-large-patch14-336'
                model = CLIPModel.from_pretrained(path, local_files_only=True).to('cuda').eval()
                processor = CLIPProcessor.from_pretrained(path, local_files_only=True)
            elif kind == 'SSCD':
                model = torch.jit.load(str(RES / 'runs/VL-Dedup-runs/current/models/sscd/sscd_disc_mixup.torchscript.pt'), map_location='cuda').eval()
                processor = None
            elif kind == 'SimCSE':
                path = baseline_surfaces / 'models/simcse'
                model = AutoModel.from_pretrained(path, local_files_only=True).to('cuda').eval()
                processor = AutoTokenizer.from_pretrained(path, local_files_only=True)
            else:
                return (None, None, 0.0)
            return (model, processor, time.perf_counter() - begin)
        groups = [('CLIP', ['CLIP_image', 'CLIP_text']), ('SSCD', ['SSCD']), ('SimCSE', ['SimCSE']), ('pHash', ['pHash'])]
        for (kind, names) in groups:
            (model, processor, loading) = load(kind)
            for name in names:
                sample = pilot_images if name in {'CLIP_image', 'SSCD', 'pHash'} else pilot_records
                (pilot[name], seconds) = run(name, sample, model, processor)
                timings[name] = dict(pilot_wall_seconds=seconds, pilot_rows=len(sample), model_load_seconds=loading if name == names[0] else 0.0, batch_size=dict(CLIP_image=64, CLIP_text=256, SSCD=32, SimCSE=128, pHash=1)[name])
                save_begin = time.perf_counter()
                np.save(OUT / (name + '_pilot.npy'), pilot[name])
                timings[name]['pilot_save_seconds'] = time.perf_counter() - save_begin
                print('pilot', name, timings[name], flush=True)
            del model, processor
            torch.cuda.empty_cache()
        for (kind, names) in groups:
            (model, processor, loading) = load(kind)
            for name in names:
                image_encoder = name in {'CLIP_image', 'SSCD', 'pHash'}
                remaining = images.iloc[2000:] if image_encoder else frame[~pilot_mask]
                (values, seconds) = run(name, remaining, model, processor)
                mapping_begin = time.perf_counter()
                unique_save_seconds = 0.0
                if image_encoder:
                    complete = np.concatenate([pilot[name], values])
                    save_begin = time.perf_counter()
                    np.save(OUT / (name + '_unique.npy'), complete)
                    unique_save_seconds = time.perf_counter() - save_begin
                    complete = complete[image_positions]
                else:
                    complete = np.empty((len(frame), values.shape[1]), np.float32)
                    (complete[pilot_mask], complete[~pilot_mask]) = (pilot[name], values)
                mapping_seconds = time.perf_counter() - mapping_begin - unique_save_seconds
                save_begin = time.perf_counter()
                np.save(OUT / (name + '.npy'), complete)
                save_seconds = unique_save_seconds + time.perf_counter() - save_begin
                total = timings[name]['pilot_wall_seconds'] + seconds
                timings[name].update(wall_seconds=total, records=len(frame), unique_images=len(images) if image_encoder else None, records_per_second=len(frame) / total, image_files_per_second=len(images) / total if image_encoder else None, second_load_seconds=loading if name == names[0] else 0.0, gpu=hardware if name != 'pHash' else 'CPU')
                timings[name].update(mapping_seconds=mapping_seconds, save_seconds=save_seconds)
                dump(OUT / 'encoder_timings.json', timings)
                print('completed', name, total, flush=True)
                del values, complete
            del model, processor
            torch.cuda.empty_cache()
        begin = time.perf_counter()
        image = np.load(OUT / 'CLIP_image.npy', mmap_mode='r')
        text = np.load(OUT / 'CLIP_text.npy', mmap_mode='r')
        score = np.empty(len(frame), np.float32)
        for offset in range(0, len(frame), 50000):
            score[offset:offset + 50000] = np.sum(image[offset:offset + 50000] * text[offset:offset + 50000], axis=1)
        seconds = time.perf_counter() - begin
        save_begin = time.perf_counter()
        np.save(OUT / 'CLIPScore.npy', score)
        timings['CLIPScore'] = dict(wall_seconds=seconds, records=len(frame), records_per_second=len(frame) / seconds, batch_size=50000, gpu='CPU', save_seconds=time.perf_counter() - save_begin)
        dump(OUT / 'encoder_timings.json', timings)
        receipt['execution_status'] = 'completed'

def configure_threads(n):
    import pyarrow as pa
    from threadpoolctl import threadpool_limits
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'GRAPH_CPU_THREADS'):
        os.environ[key] = str(n)
    pa.set_cpu_count(n)
    pa.set_io_thread_count(n)
    return threadpool_limits(limits=n)

def predict(frame, candidates, image, text, phashes, sscd, bundle):
    import numpy as np
    from scripts.run_pipeline_scaling import visual_classifier_features
    from scripts.train_hgb_classifier import raw_probability
    features = visual_classifier_features(frame, image, text, phashes, sscd)
    (raw, calibrated) = (np.empty(len(candidates)), np.empty(len(candidates)))
    for offset in range(0, len(candidates), 32768):
        stop = min(offset + 32768, len(candidates))
        x = features.transform(candidates.iloc[offset:stop])
        raw[offset:stop] = raw_probability(bundle['model'], x)
        calibrated[offset:stop] = bundle['calibrator'].predict(raw[offset:stop])
        if offset % (32768 * 50) == 0:
            print('score', offset, '/', len(candidates), flush=True)
    return (raw, calibrated)

def positive_edges(frame, candidates, raw, calibrated, threshold):
    import numpy as np
    import pandas as pd
    ids = pd.Index(sorted(frame.record_id))
    a = ids.get_indexer(candidates.record_id_a)
    b = ids.get_indexer(candidates.record_id_b)
    assert min(a.min(), b.min()) >= 0
    (lo, hi) = (np.minimum(a, b), np.maximum(a, b))
    selected = np.flatnonzero(calibrated >= threshold)
    order = selected[np.lexsort((hi[selected], lo[selected], -raw[selected], -calibrated[selected]))]
    return (ids, lo[order], hi[order], order)

def graph_metrics(labels, groups):
    from scripts.run_graph_baselines import pairwise_counts
    counts = pairwise_counts(labels, groups)
    sizes = Counter(map(int, labels))
    group_components = {}
    for (label, group) in zip(labels, groups):
        group_components.setdefault(group, set()).add(int(label))
    counts.update(clusters=len(sizes), max_cluster=max(sizes.values(), default=0), U=counts['predicted_positive_pairs'], clusters_size_ge2=sum((s >= 2 for s in sizes.values())), records_in_clusters_size_ge2=sum((s for s in sizes.values() if s >= 2)), split_groups=sum((len(v) > 1 for v in group_components.values())), precision=counts['tp'] / (counts['tp'] + counts['fp']) if counts['tp'] + counts['fp'] else 0.0, recall=counts['tp'] / (counts['tp'] + counts['fn']) if counts['tp'] + counts['fn'] else 0.0)
    return counts

def stage_b(threads=8):
    import numpy as np
    import pandas as pd
    from scripts import build_candidates as candidate
    if threads is None:
        threads = 8
    name = 'B_pilot' if False else 'B'
    with stage(name) as receipt, configure_threads(threads):
        candidate.faiss.omp_set_num_threads(threads)
        candidate.tqdm = lambda iterable, *a, **k: iterable
        (begin, cpu) = (time.perf_counter(), time.process_time())
        frame = pd.read_parquet(OUT / 'corpus.parquet')
        frame = frame.reset_index(drop=True)
        partitions = frame.partition.to_numpy()
        (phases, edges, counts) = ({}, {}, {})
        features = OUT / 'features'
        if not (features / 'phash/phash.parquet').exists():
            path = features / 'phash/phash.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            public = pd.read_parquet(OUT / 'corpus.parquet', columns=['record_id'])
            public['phash_int'] = np.load(OUT / 'pHash.npy')
            public.to_parquet(path, index=False)
        mark = time.perf_counter()
        counts['phash'] = candidate.build_phash_edges(frame, features, 2, edges)
        phases['phash_read_generate_merge'] = time.perf_counter() - mark
        for (channel, filename, k) in [('clip_image', 'CLIP_image.npy', 30), ('clip_text', 'CLIP_text.npy', 10)]:
            mark = time.perf_counter()
            embedding = np.array(np.load(OUT / filename, mmap_mode='r')[:len(frame)], dtype=np.float32, order='C', copy=True)
            candidate.faiss.normalize_L2(embedding)
            phases[channel + '_cache_read_normalize'] = time.perf_counter() - mark
            mark = time.perf_counter()
            counts[channel] = candidate.build_embedding_edges(embedding, channel, channel + '_score', k, None, edges, partitions)
            phases[channel + '_index_search_merge'] = time.perf_counter() - mark
            print(name, channel, phases[channel + '_index_search_merge'], 'seconds', flush=True)
            del embedding
        mark = time.perf_counter()
        counts['text_exact'] = candidate.build_text_exact_edges(frame, 20, edges)
        phases['text_exact_generate_merge'] = time.perf_counter() - mark
        mark = time.perf_counter()
        candidates = candidate.edges_to_frame(edges, frame)
        phases['candidate_materialize_sort'] = time.perf_counter() - mark
        mark = time.perf_counter()
        candidates.to_parquet(OUT / ('pilot_candidates.parquet' if False else 'candidates.parquet'), index=False)
        phases['candidate_parquet_write'] = time.perf_counter() - mark
        receipt.update(records=len(frame), candidates=len(candidates), threads=threads, B_wall_seconds=time.perf_counter() - begin, B_cpu_seconds=time.process_time() - cpu, phases=phases, channel_new_edges=counts)
        print(name, receipt, flush=True)

def stage_cd():
    import joblib
    import numpy as np
    import pandas as pd
    from scripts.cluster_with_size_cap import build_graph
    from scripts.baseline_matrix_text import normalize_baseline_text
    with stage('CD') as receipt, configure_threads(8):
        frame = pd.read_parquet(OUT / 'corpus.parquet')
        candidates = pd.read_parquet(OUT / 'candidates.parquet')
        (image, text, sscd, hashes, simcse) = [np.load(OUT / (name + '.npy'), mmap_mode='r') for name in ['CLIP_image', 'CLIP_text', 'SSCD', 'pHash', 'SimCSE']]
        (results, costs) = ([], {})
        for method in ['classifier_LR', 'classifier_HGB', 'B2', 'B6']:
            (begin, cpu) = (time.perf_counter(), time.process_time())
            if method in {'classifier_LR', 'classifier_HGB'}:
                bundle = joblib.load(visual_classifier / (method + '.joblib'))
                (raw, calibrated) = predict(frame, candidates, image, text, hashes, sscd, bundle)
                threshold = 2 / 3 if method == 'classifier_LR' else 1.0
            else:
                row_index = pd.Index(frame.record_id)
                (a, b) = (row_index.get_indexer(candidates.record_id_a), row_index.get_indexer(candidates.record_id_b))
                calibrated = np.empty(len(candidates), np.float64)
                baseline_model = joblib.load(baseline_surfaces / 'B6.joblib') if method == 'B6' else None
                normalized = frame.text.map(normalize_baseline_text).to_numpy() if method == 'B6' else None
                for offset in range(0, len(candidates), 8192):
                    (aa, bb) = (a[offset:offset + 8192], b[offset:offset + 8192])
                    visual = np.sum(sscd[aa] * sscd[bb], axis=1, dtype=np.float32)
                    if method == 'B6':
                        language = np.sum(simcse[aa] * simcse[bb], axis=1, dtype=np.float32)
                        x = np.column_stack([visual, language, (normalized[aa] == normalized[bb]).astype(np.float32)])
                        calibrated[offset:offset + len(aa)] = baseline_model.predict_proba(x)[:, 1]
                    else:
                        calibrated[offset:offset + len(aa)] = visual
                raw = calibrated
                threshold = 0.828769862651825 if method == 'B2' else 0.9259631557561843
            costs[method] = dict(C_wall_seconds=time.perf_counter() - begin, C_cpu_seconds=time.process_time() - cpu, threads=8)
            np.save(OUT / (method + '_raw.npy'), raw)
            np.save(OUT / (method + '_calibrated.npy'), calibrated)
            begin = time.perf_counter()
            (ids, a, b, selected) = positive_edges(frame, candidates, raw, calibrated, threshold)
            costs[method]['edge_selection_sort_wall_seconds'] = time.perf_counter() - begin
            groups = frame.set_index('record_id').loc[ids].visual_group_id.to_numpy()
            for cap in [5, 'inf'] if method in {'classifier_LR', 'classifier_HGB'} else ['inf']:
                begin = time.perf_counter()
                (labels, _, _, graph_counts) = build_graph(len(frame), a, b, cap)
                seconds = time.perf_counter() - begin
                np.save(OUT / (method + '_K' + str(cap) + '_components.npy'), labels)
                stats = graph_metrics(labels, groups)
                row = dict(method=method, K=cap, S=len(selected), D_wall_seconds=costs[method]['edge_selection_sort_wall_seconds'] + seconds, D_union_find_wall_seconds=seconds, D_shared_selection_sort_wall_seconds=costs[method]['edge_selection_sort_wall_seconds'], **stats, **graph_counts)
                if cap == 5:
                    assert stats['max_cluster'] <= 5 and stats['U'] <= 2 * len(frame)
                    assert stats['split_groups'] >= 309 and stats['fn'] >= 1560
                if method in {'classifier_LR', 'classifier_HGB'}:
                    row.update(pair_vlm_gpu_hours=stats['U'] * 0.437 / 3600, record_vlm_gpu_hours=stats['records_in_clusters_size_ge2'] * 0.0661 / 3600)
                results.append(row)
                dump(OUT / 'graph_results.json', dict(rows=results, costs=costs))
                print(method, cap, row, flush=True)
        receipt['rows'] = len(results)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['corpus', 'images', 'A', 'B', 'CD'])
    args = parser.parse_args()
    {'corpus': build_corpus, 'images': extract_images, 'A': stage_a, 'B': stage_b, 'CD': stage_cd}[args.phase]()
if __name__ == '__main__':
    main()
