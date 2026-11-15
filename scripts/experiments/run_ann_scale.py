"""ANN_SCALE_RUN CPU-only ANN scale experiment over immutable RETRIEVAL_RUN/COCO_SCALE_RUN/SOP_SCALE_RUN caches."""
from __future__ import annotations

from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = RESOURCES
OUT = RES / 'runs/ann_scale'
REPORT = ROOT / 'outputs/reports/ann_scale'
COCO = RES / 'runs/coco_scale'
SOP = RES / 'runs/sop_scale'
RETRIEVAL_RUN = RES / 'runs/pair_features'
LOCK = RES / 'locks/cpu_benchmark.lock'
HNSW = dict(hnsw_m=16, ef_construction=40, ef_search=64, build_threads=1, search_threads=8)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS'):
    os.environ[key] = '8'


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def used_time():
    return sum(read(p)['wall_seconds'] for p in OUT.glob('*_stage.json') if 'wall_seconds' in read(p))


@contextmanager
def stage(name):
    path = OUT / (name + '_stage.json')
    if path.exists():
        raise FileExistsError(path)
    wait = time.perf_counter()
    with LOCK.open('a') as lock:
        print('waiting_cpu_lock', name, flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        remaining = 14400 - used_time()
        if remaining <= 0:
            raise TimeoutError('Four-hour cumulative lock budget exhausted')
        begin, cpu = time.perf_counter(), time.process_time()
        receipt = dict(stage=name, status='running', wait_seconds=begin-wait,
                       started_monotonic=begin, start_cpu_seconds=cpu,
                       execution_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())
        dump(path, receipt)

        def expired(signum, frame):
            raise TimeoutError('Four-hour cumulative lock budget exhausted')

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            yield receipt
            receipt['status'] = 'completed'
        except BaseException as exc:
            receipt.update(status='failed', error=repr(exc))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            receipt.update(wall_seconds=time.perf_counter()-begin, cpu_seconds=time.process_time()-cpu,
                           peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            dump(path, receipt)
            print(name, receipt, flush=True)


def metrics(labels, groups):
    from scripts.run_coco_scale import graph_metrics
    result = graph_metrics(labels, groups)
    assert result['tp'] + result['fp'] == result['U']
    assert result['tp'] + result['fn'] == result['true_positive_pairs']
    assert result['tp'] + result['fp'] + result['fn'] + result['tn'] == result['pairs']
    return result


def gate_cached():
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from scripts.run_coco_scale import configure_threads
    protocol = read(CONFIG / 'ann_scale.json')
    with stage('gate_cached') as receipt, configure_threads(8):
        checks, observed = [], {}
        for name, source, caps in [('COCO', COCO, [5, 'inf']), ('SOP', SOP, [8, 'inf'])]:
            frame = pd.read_parquet(source / 'corpus.parquet').sort_values('record_id')
            candidates = source / ('candidates.parquet' if name == 'COCO' else 'stage_a_amendment2/candidates.parquet')
            rows = dict(candidates=pq.ParquetFile(candidates).metadata.num_rows)
            expected = protocol['gates'][name]
            checks.append(dict(check=name + '.candidates', actual=rows['candidates'], expected=expected['candidates'], passed=rows['candidates'] == expected['candidates']))
            for cap in caps:
                labels = np.load(source / f'vl_dedup_lr_K{cap}_components.npy')
                assert labels.shape == (len(frame),)
                rows['K' + str(cap)] = dict(all=metrics(labels, frame.visual_group_id.to_numpy()))
                if name == 'SOP':
                    mask = frame.evaluation_partition.eq('test').to_numpy()
                    rows['K' + str(cap)]['test'] = metrics(labels[mask], frame.visual_group_id.to_numpy()[mask])
            for key, wanted in expected.items():
                if key in ('candidates', 'positive_pairs'):
                    if key == 'positive_pairs':
                        actual = rows['K5']['all']['true_positive_pairs']
                        checks.append(dict(check=name + '.' + key, actual=actual, expected=wanted, passed=actual == wanted))
                    continue
                cap, _, scope = key.partition('_')
                actual_row = rows[cap][scope or 'all']
                for field, value in wanted.items():
                    actual = round(actual_row['f1'], 6) if field == 'f1_rounded6' else actual_row[field]
                    checks.append(dict(check=name + '.' + key + '.' + field, actual=actual, expected=value, passed=actual == value))
            observed[name] = rows
        result = dict(status='passed' if all(c['passed'] for c in checks) else 'failed', checks=checks, observed=observed)
        dump(OUT / 'gate_cached.json', result)
        receipt['gate_status'] = result['status']
        if result['status'] != 'passed':
            raise RuntimeError('Cached graph gate mismatch: see gate_cached.json')


def subset_positions(image_ids, target, seed=20261106):
    """Smallest shuffled whole-image prefix, returned in original row order."""
    import numpy as np
    unique, counts = np.unique(image_ids, return_counts=True)
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(unique))
    stop = int(np.searchsorted(np.cumsum(counts[order]), target, side='left')) + 1
    if target < 1 or target > len(image_ids):
        raise ValueError('Target must be within the corpus')
    return np.flatnonzero(np.isin(image_ids, unique[order[:stop]])), stop


def embedding_edges(builder, embedding, channel, k, edges, partitions, backend):
    """Use the existing builders; instrument exact add/search without replacing it."""
    from scripts.experiments.support import build_hnsw_edges
    if backend == 'hnsw':
        return build_hnsw_edges(builder, embedding, channel, k, edges, partitions, HNSW)
    phases = {}
    original = builder.make_index

    class MeasuredIndex:
        def __init__(self, index):
            self.index = index

        def measure(self, name, function, *args):
            w, p = time.perf_counter(), time.process_time()
            result = function(*args)
            phases[name] = dict(wall_seconds=time.perf_counter()-w, cpu_seconds=time.process_time()-p)
            return result

        def add(self, values):
            return self.measure('index_add', self.index.add, values)

        def search(self, values, count):
            return self.measure('search', self.index.search, values, count)

    builder.make_index = lambda dim, resources: MeasuredIndex(original(dim, resources))
    w, p = time.perf_counter(), time.process_time()
    try:
        added = builder.build_embedding_edges(embedding, channel, channel + '_score', k, None, edges, partitions)
    finally:
        builder.make_index = original
    phases['python_union_and_index_init'] = dict(
        wall_seconds=time.perf_counter()-w-sum(v['wall_seconds'] for v in phases.values()),
        cpu_seconds=time.process_time()-p-sum(v['cpu_seconds'] for v in phases.values()))
    return added, phases


def retrieval(corpus, backend, target, attempt=1):
    import numpy as np
    import pandas as pd
    from scripts import build_candidates as builder
    from scripts.run_coco_scale import configure_threads
    name = f'B_{corpus}_{backend}_{target}'
    if attempt > 1:
        name += f'_attempt{attempt}'
    dest = OUT / name
    dest.mkdir(exist_ok=False)
    with stage(name) as receipt, configure_threads(8):
        builder.faiss.omp_set_num_threads(8)
        builder.tqdm = lambda iterable, *a, **k: iterable
        begin, cpu = time.perf_counter(), time.process_time()
        columns = ['record_id', 'text', 'partition']
        if corpus == 'benchmark':
            source, feature = RETRIEVAL_RUN, RETRIEVAL_RUN / 'features'
            frame = pd.read_parquet(source / 'public_manifest.parquet', columns=['record_id', 'text'])
            member = pd.read_csv(RES / 'runs/retrieval_tradeoff/membership.csv')
            assert frame.record_id.tolist() == member.record_id.tolist()
            metadata = pd.read_parquet(feature / 'clip/metadata.parquet')
            assert metadata.record_id.tolist() == frame.record_id.tolist()
            assert np.array_equal(metadata.row_index, np.arange(len(frame)))
            frame['partition'] = 'global_retrieval_tradeoff'
            positions, images = np.arange(len(frame)), None
            files = ['features/clip/image_emb.npy', 'features/clip/text_emb.npy']
        else:
            source = COCO if corpus == 'coco' else SOP / 'stage_a_amendment2'
            feature = source / 'features'
            corpus_path = (COCO if corpus == 'coco' else SOP) / 'corpus.parquet'
            frame = pd.read_parquet(corpus_path, columns=columns + ['image_id'])
            if corpus == 'coco':
                positions, images = subset_positions(frame.image_id.to_numpy(), target)
            else:
                positions, images = np.arange(len(frame)), len(frame)
            frame = frame.iloc[positions].reset_index(drop=True)
            files = ['CLIP_image.npy', 'CLIP_text.npy']
        assert frame.partition.nunique() == 1
        phases, edges, counts, subphases = {}, {}, {}, {}

        def phase(label, function):
            w, p = time.perf_counter(), time.process_time()
            value = function()
            phases[label] = dict(wall_seconds=time.perf_counter()-w, cpu_seconds=time.process_time()-p)
            return value

        counts['phash'] = phase('phash_read_generate_merge', lambda: builder.build_phash_edges(frame, feature, 2, edges))
        for channel, filename, k in zip(['clip_image', 'clip_text'], files, [30, 10]):
            def load():
                array = np.load(source / filename, mmap_mode='r')
                embedding = np.array(array[positions], dtype=np.float32, order='C', copy=True)
                assert embedding.shape == (len(frame), 768) and np.isfinite(embedding).all()
                builder.faiss.normalize_L2(embedding)
                return embedding

            embedding = phase(channel + '_cache_read_normalize', load)
            counts[channel], subphases[channel] = phase(channel + '_index_search_merge', lambda: embedding_edges(
                builder, embedding, channel, k, edges, frame.partition.to_numpy(), backend))
            print(name, channel, phases[channel + '_index_search_merge'], flush=True)
            del embedding
        counts['text_exact'] = phase('text_exact_generate_merge', lambda: builder.build_text_exact_edges(frame, 20, edges))
        candidates = phase('candidate_materialize_sort', lambda: builder.edges_to_frame(edges, frame))
        phase('candidate_parquet_write', lambda: candidates.to_parquet(dest / 'candidates.parquet', index=False))
        receipt.update(corpus=corpus, backend=backend, target=target, records=len(frame), images=images,
                       candidates=len(candidates), B_wall_seconds=time.perf_counter()-begin,
                       B_cpu_seconds=time.process_time()-cpu, phases=phases, backend_subphases=subphases,
                       channel_new_edges=counts, label_access=False,
                       candidate_file=str(dest / 'candidates.parquet'), threads=8,
                       environment=dict(numpy=np.__version__, faiss=builder.faiss.__version__, python=sys.version))
        assert sum(counts.values()) == len(candidates)
        if corpus == 'benchmark':
            actual, expected = len(candidates), 1508778
            gate = dict(actual_candidates=actual, expected_candidates=expected,
                        status='passed' if actual == expected else 'failed')
            dump(OUT / 'gate_r14.json', gate)
            if actual != expected:
                raise RuntimeError('RETRIEVAL_RUN HNSW candidate gate mismatch')


def candidate_path(corpus):
    n = 580767 if corpus == 'coco' else 97503
    return OUT / f'B_{corpus}_hnsw_{n}/candidates.parquet'


def score(corpus):
    import joblib
    import numpy as np
    import pandas as pd
    from scripts.run_coco_scale import configure_threads, predict
    CLASSIFIER_RUN = RESOURCES / 'runs/scorer_noasset'
    source = COCO if corpus == 'coco' else SOP
    feature = source if corpus == 'coco' else source / 'stage_a_amendment2'
    with stage('C_' + corpus) as receipt, configure_threads(8):
        frame = pd.read_parquet(source / 'corpus.parquet')
        candidates = pd.read_parquet(candidate_path(corpus))
        image, text, phashes, sscd = [np.load(feature / (name + '.npy'), mmap_mode='r')
                                     for name in ['CLIP_image', 'CLIP_text', 'pHash', 'SSCD']]
        bundle = joblib.load(CLASSIFIER_RUN / 'vl_dedup_lr.joblib')
        assert bundle['threshold'] == 2/3
        begin, cpu = time.perf_counter(), time.process_time()
        raw, calibrated = predict(frame, candidates, image, text, phashes, sscd, bundle)
        receipt.update(C_score_wall_seconds=time.perf_counter()-begin, C_score_cpu_seconds=time.process_time()-cpu,
                       candidates=len(candidates), records=len(frame), threshold=2/3, model='vl_dedup_lr')
        np.save(OUT / f'{corpus}_raw.npy', raw)
        np.save(OUT / f'{corpus}_calibrated.npy', calibrated)


def graph(corpus):
    import numpy as np
    import pandas as pd
    from scripts.run_coco_scale import configure_threads, positive_edges
    from scripts.cluster_with_size_cap import build_graph
    source = COCO if corpus == 'coco' else SOP
    with stage('D_' + corpus) as receipt, configure_threads(8):
        frame = pd.read_parquet(source / 'corpus.parquet')
        candidates = pd.read_parquet(candidate_path(corpus))
        raw = np.load(OUT / f'{corpus}_raw.npy', mmap_mode='r')
        calibrated = np.load(OUT / f'{corpus}_calibrated.npy', mmap_mode='r')
        begin, cpu = time.perf_counter(), time.process_time()
        ids, a, b, selected = positive_edges(frame, candidates, raw, calibrated, 2/3)
        receipt.update(selection_sort_wall_seconds=time.perf_counter()-begin,
                       selection_sort_cpu_seconds=time.process_time()-cpu, S=len(selected))
        ordered = frame.set_index('record_id').loc[ids]
        del candidates
        rows = []
        for cap in ([5, 'inf'] if corpus == 'coco' else [8, 'inf']):
            begin, cpu = time.perf_counter(), time.process_time()
            labels, retained, rejected, counts = build_graph(len(frame), a, b, cap)
            cost = dict(union_find_wall_seconds=time.perf_counter()-begin,
                        union_find_cpu_seconds=time.process_time()-cpu)
            np.save(OUT / f'{corpus}_K{cap}_components.npy', labels)
            projections = dict(all=dict(S=len(selected), **metrics(labels, ordered.visual_group_id.to_numpy())))
            if corpus == 'sop':
                mask = ordered.evaluation_partition.eq('test').to_numpy()
                projections['test'] = dict(S=int((mask[a] & mask[b]).sum()),
                                          **metrics(labels[mask], ordered.visual_group_id.to_numpy()[mask]))
            rows.append(dict(K=cap, method='vl_dedup_lr', projections=projections, graph_counts=counts, **cost))
            dump(OUT / f'{corpus}_graph_results.json', dict(rows=rows))
            print(corpus, cap, projections, flush=True)
        receipt['rows'] = len(rows)


def candidate_comparison(corpus):
    import numpy as np
    import pandas as pd
    from scripts.run_coco_scale import configure_threads
    source = COCO if corpus == 'coco' else SOP
    exact_file = (source if corpus == 'coco' else source / 'stage_a_amendment2') / 'candidates.parquet'
    with stage('compare_' + corpus) as receipt, configure_threads(8):
        frame = pd.read_parquet(source / 'corpus.parquet')
        groups = frame.visual_group_id.to_numpy()
        coverage, keys = {}, {}
        for backend, path in [('exact', exact_file), ('hnsw', candidate_path(corpus))]:
            pairs = pd.read_parquet(path, columns=['idx_a', 'idx_b'])
            a, b = pairs.idx_a.to_numpy(), pairs.idx_b.to_numpy()
            assert min(a.min(), b.min()) >= 0 and max(a.max(), b.max()) < len(frame)
            encoded = np.minimum(a, b).astype(np.uint64) * np.uint64(len(frame)) + np.maximum(a, b).astype(np.uint64)
            keys[backend] = np.unique(encoded)
            assert len(keys[backend]) == len(pairs)
            positive = groups[a] == groups[b]
            scopes = {'all': np.ones(len(frame), dtype=bool)}
            if corpus == 'sop':
                scopes['test'] = frame.evaluation_partition.eq('test').to_numpy()
            coverage[backend] = {}
            for scope, mask in scopes.items():
                sizes = pd.Series(groups[mask]).value_counts().to_numpy(dtype=np.int64)
                denominator = int((sizes*(sizes-1)//2).sum())
                covered = int((positive & mask[a] & mask[b]).sum())
                coverage[backend][scope] = dict(covered=covered, positive_pairs=denominator,
                                                missed=denominator-covered, recall=covered/denominator)
        intersection = len(np.intersect1d(keys['exact'], keys['hnsw'], assume_unique=True))
        exact, hnsw = len(keys['exact']), len(keys['hnsw'])
        union = exact + hnsw - intersection
        result = dict(exact_candidates=exact, hnsw_candidates=hnsw, intersection=intersection,
                      exact_only=exact-intersection, hnsw_only=hnsw-intersection, union=union,
                      Jaccard=intersection/union, positive_coverage=coverage)
        assert intersection + result['exact_only'] == exact
        assert intersection + result['hnsw_only'] == hnsw
        dump(OUT / f'{corpus}_candidate_comparison.json', result)
        receipt.update(exact_candidates=exact, hnsw_candidates=hnsw)


def budget(pending):
    """Conservative whole-run projection, updated after each observed stage."""
    import numpy as np
    import pandas as pd
    image_ids = pd.read_parquet(COCO / 'corpus.parquet', columns=['image_id']).image_id.to_numpy()
    estimates = []
    b_rows = [read(path) for path in OUT.glob('B_*_stage.json') if read(path)['status'] == 'completed']
    for phase, corpus, backend, target in pending:
        if phase == 'B':
            n = len(subset_positions(image_ids, target)[0]) if corpus == 'coco' else target
            if backend == 'exact':
                seconds = 6513.7*(n/580767)**2
                for row in b_rows:
                    if row['backend'] == 'exact':
                        seconds = max(seconds, row['B_wall_seconds']*(n/row['records'])**2)
            else:
                seconds = 2*28.468*(n/50000)**1.2
                for row in b_rows:
                    if row['backend'] == 'hnsw':
                        seconds = max(seconds, row['B_wall_seconds']*(n/row['records'])**1.2)
        elif phase == 'C':
            seconds = 2*(653.326 if corpus == 'coco' else 75.771)
        else:
            seconds = 150 if corpus == 'coco' else 50
        estimates.append(dict(phase=phase, corpus=corpus, backend=backend, target=target, seconds=seconds))
    used = used_time()
    projected = used + sum(row['seconds'] for row in estimates)
    result = dict(status='passed' if projected <= 14400 else 'failed', used_lock_wall_seconds=used,
                  remaining=estimates, projected_total_lock_wall_seconds=projected, limit_seconds=14400)
    dest = OUT / 'budget_receipts'
    dest.mkdir(exist_ok=True)
    dump(dest / f'{len(list(dest.glob("*.json"))):02d}.json', result)
    print('budget', used, projected, 'seconds', flush=True)
    if result['status'] != 'passed':
        raise TimeoutError('Projected cumulative lock wall time exceeds four hours')


def run():
    assert read(OUT / 'gate_cached.json')['status'] == 'passed'
    assert read(OUT / 'gate_r14.json')['status'] == 'passed'
    pending = [('B', 'coco', backend, target) for target in [50000, 100000, 200000, 400000, 580767]
               for backend in ['exact', 'hnsw'] if not (target == 580767 and backend == 'exact')]
    pending += [('C', 'coco', None, None), ('D', 'coco', None, None), ('compare', 'coco', None, None),
                ('B', 'sop', 'hnsw', 97503), ('C', 'sop', None, None), ('D', 'sop', None, None),
                ('compare', 'sop', None, None)]
    for index, (phase, corpus, backend, target) in enumerate(pending):
        stage_name = f'B_{corpus}_{backend}_{target}' if phase == 'B' else phase + '_' + corpus
        receipt = OUT / (stage_name + '_stage.json')
        if receipt.exists():
            if read(receipt)['status'] != 'completed':
                raise RuntimeError('Preserved failed stage requires investigation: ' + stage_name)
            continue
        budget(pending[index:])
        command = [sys.executable, '-I', str(Path(__file__).resolve()), '--phase', phase, '--corpus', corpus]
        if phase == 'B':
            command += ['--backend', backend, '--target', str(target)]
        with (OUT / (stage_name + '.log')).open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    if not receipt.exists():
                        continue  # Lock waiting is excluded from the budget.
                    current = read(receipt)
                    if current['status'] != 'running' or 'started_monotonic' not in current:
                        continue
                    elapsed = time.perf_counter() - current['started_monotonic']
                    if used_time() + elapsed <= 14400:
                        continue
                    # A Python alarm can be delayed inside native FAISS. Stop it
                    # from the parent, preserving a receipt before termination.
                    stat = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
                    cpu = (int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK')
                    status = Path(f'/proc/{process.pid}/status').read_text().splitlines()
                    peak = next(int(line.split()[1]) for line in status if line.startswith('VmHWM:'))
                    process.terminate()
                    process.wait()
                    current.update(status='failed', error='Parent enforced four-hour cumulative CPU-lock wall budget',
                                   wall_seconds=elapsed, cpu_seconds=max(0, cpu-current['start_cpu_seconds']), peak_rss_kib=peak)
                    dump(receipt, current)
                    raise TimeoutError(current['error'])
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
    budget([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['gate_cached', 'B', 'C', 'D', 'compare', 'run'])
    parser.add_argument('--corpus', choices=['benchmark', 'coco', 'sop'])
    parser.add_argument('--backend', choices=['exact', 'hnsw'])
    parser.add_argument('--target', type=int)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.phase == 'gate_cached':
        gate_cached()
    elif args.phase == 'B':
        retrieval(args.corpus, args.backend, args.target, args.attempt)
    elif args.phase == 'run':
        run()
    else:
        dict(C=score, D=graph, compare=candidate_comparison)[args.phase](args.corpus)


if __name__ == '__main__':
    main()
