"""SCALE_BASELINES_RUN CPU-only native baseline workers and frozen-input gates."""
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata as md
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = RESOURCES
RUN = RES / 'runs/scale_baselines'
REPORT = ROOT / 'outputs/reports/scale_baselines'
LOCK = RES / 'locks/cpu_benchmark.lock'
SOURCES = dict(original=RES / 'runs/pair_features',
               coco=RES / 'runs/coco_scale',
               sop=RES / 'runs/sop_scale')
METHODS = ['data_juicer', 'semhash', 'semdedup', 'caption_minhash']

import numpy as np
import pandas as pd


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def groups_to_components(n, groups):
    """Union overlapping native groups; leave absent records as singletons."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for group in groups:
        group = list(group)
        if not group:
            continue
        for i in group:
            if not 0 <= i < n:
                raise ValueError(f'Group index outside corpus: {i}')
        a = find(group[0])
        for i in group[1:]:
            b = find(i)
            parent[max(a, b)] = min(a, b)
            a = min(a, b)
    return np.asarray([find(i) for i in range(n)], dtype=np.int32)


def metrics(labels, groups):
    from scripts.run_coco_scale import graph_metrics
    result = graph_metrics(labels, groups)
    result['delete_if_one_per_cluster'] = len(labels) - result['clusters']
    assert result['tp'] + result['fp'] + result['fn'] + result['tn'] == result['pairs']
    assert result['delete_if_one_per_cluster'] + result['clusters'] == len(labels)
    return result


@contextmanager
def locked(name, threads=4):
    from threadpoolctl import threadpool_limits, threadpool_info
    import pyarrow as pa
    path = RUN / (name + '_receipt.json')
    if path.exists():
        raise FileExistsError(f'Preserve prior attempt: {path}')
    wait = time.perf_counter()
    print('waiting_cpu_lock', name, flush=True)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start, cpu = time.perf_counter(), time.process_time()
        receipt = dict(status='running', name=name, started_utc=datetime.now(timezone.utc).isoformat(),
                       lock_wait_seconds=start-wait, configured_threads=threads, processes=1,
                       execution_head=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip(),
                       cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), phases={})
        dump(path, receipt)
        print('started', name, flush=True)
        try:
            with threadpool_limits(limits=threads):
                pa.set_cpu_count(threads)
                pa.set_io_thread_count(threads)
                yield receipt
                receipt['threadpools'] = threadpool_info()
            receipt['status'] = 'completed'
        except BaseException as exc:
            receipt.update(status='failed', error=repr(exc))
            raise
        finally:
            receipt.update(wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
                           peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           finished_utc=datetime.now(timezone.utc).isoformat())
            dump(path, receipt)
            print('finished', name, receipt['status'], receipt['wall_seconds'], flush=True)


@contextmanager
def phase(receipt, name):
    start, cpu = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        receipt['phases'][name] = dict(wall_seconds=time.perf_counter()-start,
                                      cpu_seconds=time.process_time()-cpu)


def read_input(corpus, embeddings=False, image_only=False):
    source = SOURCES[corpus]
    frame = pd.read_parquet(source / ('public_manifest.parquet' if corpus == 'original' else 'corpus.parquet'))
    assert frame.record_id.is_unique
    image = text = None
    if embeddings:
        cache = source / 'stage_a_amendment2' if corpus == 'sop' else source
        image_path = source / 'features/clip/image_emb.npy' if corpus == 'original' else cache / 'CLIP_image.npy'
        text_path = source / 'features/clip/text_emb.npy' if corpus == 'original' else cache / 'CLIP_text.npy'
        image = np.load(image_path)
        assert image.shape == (len(frame), 768) and np.isfinite(image).all()
        if not image_only:
            text = np.load(text_path)
            assert text.shape == image.shape and np.isfinite(text).all()
    return frame, image, text


def run_method(corpus, method):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    # Load historical helpers before thread limits: graph_baselines sets env=4 on import.
    from scripts import run_native_baselines_50k as e2
    from scripts import run_graph_baselines as g
    threads = 8 if method in ['data_juicer', 'semhash'] else 4
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = str(threads)
    target = RUN / corpus / method
    target.mkdir(parents=True, exist_ok=True)
    with locked(f'{corpus}/{method}', threads) as receipt:
        with phase(receipt, 'read'):
            frame, image, text = read_input(corpus, method in ['semhash', 'semdedup'], method == 'semdedup')
        receipt['records'] = len(frame)
        if method == 'data_juicer':
            with phase(receipt, 'index_or_cluster'):
                assert md.version('py-data-juicer') == '1.5.3'
                op, source = e2.official_dj()
                receipt['operator_source'] = source
            with phase(receipt, 'dedup'):
                rows = e2.NativeDataset()
                for i, row in enumerate(frame.itertuples()):
                    sample = op.compute_hash(dict(record_id=row.record_id, text=row.text, images=[row.image_path]))
                    if not sample.get('__dj__imagehash'):
                        raise ValueError(f'Empty native pHash at {row.record_id}')
                    rows.append(sample)
                    if i % 10000 == 0:
                        print(corpus, method, i, '/', len(frame), flush=True)
                selected, _ = op.process(rows)
                hashes = [r['__dj__imagehash'] for r in rows]
                labels = pd.factorize(hashes, sort=False)[0].astype(np.int32)
                native = pd.DataFrame(dict(record_id=frame.record_id, native_hash=hashes))
                receipt['native_selected'] = len(selected)
            with phase(receipt, 'write'):
                native.to_parquet(target / 'native.parquet', index=False)
        elif method == 'semhash':
            with phase(receipt, 'index_or_cluster'):
                from semhash import SemHash
                assert md.version('semhash') == '0.4.1'
                assert md.version('vicinity') == '0.4.6'
                image /= np.linalg.norm(image, axis=1, keepdims=True)
                text /= np.linalg.norm(text, axis=1, keepdims=True)
                fused = np.concatenate([image/np.sqrt(2), text/np.sqrt(2)], axis=1).astype(np.float32)
                fused /= np.linalg.norm(fused, axis=1, keepdims=True)
                records = [dict(record_id=r.record_id, row_index=i, image=str(r.image_path), text=r.text)
                           for i, r in enumerate(frame.itertuples())]
                index = SemHash.from_embeddings(embeddings=fused, records=records, model=e2.NoEncoding(),
                                                columns=['image', 'text'], ann_backend='basic')
                del image, text, fused
                receipt['native_exact_groups'] = len(index.index.items)
            with phase(receipt, 'dedup'):
                result = index.self_deduplicate(threshold=.9)
                selected = [r['row_index'] for r in result.selected]
                links, exact, filtered = [], [], []
                for duplicate in result.filtered:
                    a = duplicate.record['row_index']
                    filtered.append(a)
                    for other, score in duplicate.duplicates:
                        (exact if duplicate.exact else links).append((a, other['row_index'], float(score)))
                assert sorted(selected + filtered) == list(range(len(frame)))
                labels = groups_to_components(len(frame), ((int(a), int(b)) for a, b, _ in links + exact))
                receipt.update(native_selected=len(selected), native_filtered=len(filtered),
                               native_similarity_links=len(links), exact_relations=len(exact))
            with phase(receipt, 'write'):
                np.savez_compressed(target / 'native.npz', selected=np.asarray(selected), filtered=np.asarray(filtered),
                                    native_similarity_link=np.asarray(links).reshape(-1, 3),
                                    exact_relation=np.asarray(exact).reshape(-1, 3))
        elif method == 'semdedup':
            with phase(receipt, 'index_or_cluster'):
                assert np.allclose(np.linalg.norm(image, axis=1), 1, atol=2e-6)
                assignments, centers = g.kmeans(image, 50)
            with phase(receipt, 'dedup'):
                labels, edges = g.semdedup(image, assignments, .005)
                receipt['threshold_edges'] = edges
            with phase(receipt, 'write'):
                np.save(target / 'assignments.npy', assignments)
                np.save(target / 'centroids.npy', centers)
        else:
            from scripts import run_graph_caption_minhash as mh
            with phase(receipt, 'index_or_cluster'):
                op, source = mh.official_operator()
                receipt['operator_source'] = source
            with phase(receipt, 'dedup'):
                samples = mh.NativeDataset()
                for row in frame.itertuples():
                    samples.append(op.compute_hash(dict(record_id=row.record_id, text=row.text)))
                selected, _ = op.process(samples)
                labels = selected.labels
                receipt.update(native_selected=len(selected), num_bands=op.num_bands, num_rows_per_band=op.num_rows_per_band)
            with phase(receipt, 'write'):
                pd.DataFrame(dict(record_id=[r['record_id'] for r in selected])).to_parquet(target / 'selected.parquet', index=False)
        with phase(receipt, 'write_components'):
            assert len(labels) == len(frame)
            np.save(target / 'components.npy', labels)
            frame[['record_id']].to_parquet(target / 'record_ids.parquet', index=False)
        receipt.update(clusters=len(set(labels)), max_cluster=max(Counter(labels).values()),
                       delete_if_one_per_cluster=len(frame)-len(set(labels)), gpu_required=False, encoding_calls=0)


def check_budget(actual, expected):
    differences = {}
    for key, value in expected.items():
        observed = round(actual[key], 6) if key == 'f1' else actual[key]
        if observed != value:
            differences[key] = dict(expected=value, actual=actual[key])
    return differences


def cached_gates():
    protocol = json.loads((CONFIG / 'scale_baselines.json').read_text())
    result = dict(status='passed', cells=[])
    with locked('cached_gates'):
        for corpus, gate in [('coco', 'coco'), ('sop', 'sop_test')]:
            frame, _, _ = read_input(corpus)
            frame = frame.set_index('record_id').loc[sorted(frame.record_id)]
            mask = np.ones(len(frame), dtype=bool) if corpus == 'coco' else frame.evaluation_partition.eq('test').to_numpy()
            for method, expected in protocol['gates'][gate].items():
                labels = np.load(SOURCES[corpus] / (method + '_components.npy'))
                actual = metrics(labels[mask], frame.visual_group_id.to_numpy()[mask])
                diff = check_budget(actual, expected)
                result['cells'].append(dict(corpus=corpus, method=method, scope=gate, expected=expected, differences=diff, **actual))
                if diff:
                    result['status'] = 'failed'
        dump(RUN / 'cached_gates.json', result)
        if result['status'] != 'passed':
            raise RuntimeError('Cached input gate mismatch; stop')


def evaluate(corpus):
    from scripts import run_graph_baselines as g
    protocol = json.loads((CONFIG / 'scale_baselines.json').read_text())
    result = dict(corpus=corpus, status='passed', cells=[])
    with locked(f'{corpus}/evaluation'):
        frame, _, _ = read_input(corpus)
        if corpus == 'original':
            truth = g.metadata('original', 'test')
            truth = truth[truth.original_level.ne('M')]
            positions = pd.Index(frame.record_id).get_indexer(truth.record_id)
            assert min(positions) >= 0
            scopes = [('test', positions, truth.visual_group_id.to_numpy())]
        else:
            scopes = [('all', np.arange(len(frame)), frame.visual_group_id.to_numpy())]
            if corpus == 'sop':
                mask = frame.evaluation_partition.eq('test').to_numpy()
                scopes.append(('test', np.flatnonzero(mask), frame.visual_group_id.to_numpy()[mask]))
        for method in METHODS:
            target = RUN / corpus / method
            assert pd.read_parquet(target / 'record_ids.parquet').record_id.tolist() == frame.record_id.tolist()
            labels = np.load(target / 'components.npy')
            for scope, positions, groups in scopes:
                actual = metrics(labels[positions], groups)
                diff = check_budget(actual, protocol['gates']['original'][method]) if corpus == 'original' else {}
                result['cells'].append(dict(method=method, scope=scope, differences=diff, **actual))
                if diff:
                    result['status'] = 'failed'
        dump(RUN / corpus / 'evaluation.json', result)
        if result['status'] != 'passed':
            raise RuntimeError('Original 50K native gate mismatch; stop')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['cached-gates', 'run', 'evaluate'])
    parser.add_argument('--corpus', choices=list(SOURCES))
    parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    if args.command == 'cached-gates':
        cached_gates()
    elif args.command == 'run':
        run_method(args.corpus, args.method)
    else:
        evaluate(args.corpus)


if __name__ == '__main__':
    main()
