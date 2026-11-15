from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
SIZES = [10000, 20000, 30000, 50000]
SEED = 20_261_025
ORDERS = [SIZES, list(reversed(SIZES)), [20000, 50000, 10000, 30000]]
THREAD_ENV = {key: '8' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'ARROW_NUM_THREADS')}

def stamp():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def process_cpu(who=resource.RUSAGE_SELF):
    usage = resource.getrusage(who)
    return usage.ru_utime + usage.ru_stime

def membership_rows(ids, metadata_ids, row_indices, seed):
    import numpy as np
    ids = np.asarray(ids, dtype=str)
    if len(set(ids)) != len(ids) or not np.array_equal(ids, np.asarray(metadata_ids, dtype=str)):
        raise ValueError('Public record order must match unique cache metadata IDs')
    if not np.array_equal(np.asarray(row_indices), np.arange(len(ids))):
        raise ValueError('Cache metadata row_index must match array order')
    ordered = np.argsort(ids, kind='stable')
    shuffled = np.random.Generator(np.random.PCG64(seed)).permutation(ordered)
    ranks = np.empty(len(ids), dtype=int)
    ranks[shuffled] = np.arange(1, len(ids) + 1)
    return [{'record_id': str(rid), 'cache_row': i, 'selection_rank': int(ranks[i])} for (i, rid) in enumerate(ids)]

def freeze(args):
    import numpy as np
    import pandas as pd
    started = time.perf_counter()
    cache = args.cache.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    public = pd.read_parquet(cache / 'public_manifest.parquet')
    metadata = pd.read_parquet(cache / 'features/clip/metadata.parquet')
    if len(public) != 50000:
        raise ValueError('Expected the named 50K public cache')
    rows = membership_rows(public.record_id, metadata.record_id, metadata.row_index, SEED)
    memberships = output / 'subset_membership.csv'
    with memberships.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['record_id', 'cache_row', 'selection_rank'])
        writer.writeheader()
        writer.writerows(rows)
    arrays = {}
    for name in ('image_emb.npy', 'text_emb.npy'):
        path = cache / 'features/clip' / name
        array = np.load(path, mmap_mode='r')
        if array.shape != (50000, 768) or str(array.dtype) != 'float32':
            raise ValueError(f'Unexpected cache shape/dtype: {name}')
        arrays[name] = {'shape': list(array.shape), 'dtype': str(array.dtype)}
    phash = cache / 'features/phash/phash.parquet'
    if not phash.is_file():
        raise FileNotFoundError(phash)
    previous = set()
    for size in SIZES:
        current = {r['cache_row'] for r in rows if r['selection_rank'] <= size}
        if len(current) != size or not previous.issubset(current):
            raise ValueError('Subset nesting failed')
        previous = current
    contract = {'run_id': 'retrieval_scaling', 'frozen_at_utc': stamp(), 'input_identity': 'benchmark records / pair_features cache', 'cache_root': str(cache), 'output_root': str(output), 'membership_file': str(memberships), 'source_records': 50000, 'arrays': arrays, 'subset_rule': 'Lexically sort public record IDs, PCG64(seed) permutation, take first N, restore original cache row order', 'seed': SEED, 'numpy_version_for_freeze': np.__version__, 'sizes': SIZES, 'repeats': 3, 'jobs': [{'id': f'n{n}_r{r}', 'records': n, 'repeat': r} for (r, order) in enumerate(ORDERS, 1) for n in order], 'parameters': {'cpu_threads': 8, 'phash_radius': 2, 'clip_image_topk': 30, 'clip_text_topk': 10, 'text_exact_neighbors': 20}, 'candidate_scope': 'global within each newly retrieved subset; constant engineering partition, no real split or label access', 'topk_ties': 'Preserve current function: search k+1 then skip self; ties can retain k+1 neighbors if self is absent', 'timing': {'start': 'Immediately before reading membership CSV and public cache metadata in each worker', 'end': 'After candidates.to_parquet returns; no fsync, no receipt/aggregation time', 'included': ['membership/public metadata read and subset selection', 'full pHash cache read and subset candidate generation/merge', 'selected image/text cache rows read, contiguous copy and L2 normalization', 'CPU exact FAISS index/add/search and Python edge union', 'normalized text matching/union', 'candidate frame materialization/sort', 'parquet write to local storage'], 'excluded': ['interpreter/import startup', 'subset freeze preparation', 'feature encoding', 'fitting/calibration', 'model inference', 'quality evaluation', 'manuscript/export work'], 'process_wall_and_cpu': 'Also report full worker process wall and OS child CPU including imports/cleanup', 'peak_rss': 'Linux ru_maxrss for entire independent worker process, KiB converted to GiB', 'io': 'Every repeat reads full 50K public metadata and pHash table; embedding rows selected from read-only mmap. Writes may remain in OS cache.', 'cache_condition': 'OS page cache neither flushed nor explicitly warmed. All three repeats retained, including first; not cold-cache measurements', 'summary': 'Median, minimum, maximum over all 3 successful repeats; no outlier exclusion or result-dependent retries'}, 'claims': 'Cached Stage B stage scaling only; no recall, F1, training, A-F, ANN, matched-quality speed, or natural generalization claim', 'selection_preparation_wall_seconds': time.perf_counter() - started, 'label_access': False, 'gpu_calls': 0}
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    dump(args.protocol, contract)
    print(json.dumps({'protocol': str(args.protocol), 'sizes': SIZES, 'jobs': 12}))

def worker(args):
    contract = read(args.protocol)
    job = next((j for j in contract['jobs'] if j['id'] == args.job))
    out = Path(contract['output_root']) / job['id']
    out.mkdir(exist_ok=False)
    import numpy as np
    import pandas as pd
    import pyarrow
    import build_candidates as candidate
    candidate.faiss.omp_set_num_threads(8)
    pyarrow.set_cpu_count(8)
    candidate.tqdm = lambda iterable, *a, **k: iterable
    cache = Path(contract['cache_root'])
    phases = {}
    (start_wall, start_cpu) = (time.perf_counter(), process_cpu())
    mark = start_wall
    members = pd.read_csv(contract['membership_file'], dtype={'record_id': str})
    selected = members[members.selection_rank <= job['records']].sort_values('cache_row')
    public = pd.read_parquet(cache / 'public_manifest.parquet')
    indices = selected.cache_row.to_numpy(dtype=int)
    frame = public.iloc[indices].copy().reset_index(drop=True)
    if len(frame) != job['records'] or frame.record_id.astype(str).tolist() != selected.record_id.tolist():
        raise ValueError('Frozen subset differs from public cache rows')
    frame['partition'] = 'global_stage_b_timing'
    partitions = frame.partition.to_numpy()
    phases['membership_public_read_select'] = time.perf_counter() - mark
    (edges, counts) = ({}, {})
    mark = time.perf_counter()
    counts['phash'] = candidate.build_phash_edges(frame, cache / 'features', 2, edges)
    phases['phash_read_generate_merge'] = time.perf_counter() - mark
    for (channel, filename, k) in [('clip_image', 'image_emb.npy', 30), ('clip_text', 'text_emb.npy', 10)]:
        mark = time.perf_counter()
        source = np.load(cache / 'features/clip' / filename, mmap_mode='r')
        if source.shape != tuple(contract['arrays'][filename]['shape']):
            raise ValueError('Source embedding shape changed')
        embedding = np.ascontiguousarray(source[indices], dtype=np.float32)
        if not np.isfinite(embedding).all():
            raise ValueError('Nonfinite cached features')
        candidate.faiss.normalize_L2(embedding)
        phases[channel + '_cache_read_normalize'] = time.perf_counter() - mark
        mark = time.perf_counter()
        counts[channel] = candidate.build_embedding_edges(embedding, channel, channel + '_score', k, None, edges, partitions)
        phases[channel + '_index_search_merge'] = time.perf_counter() - mark
        del embedding, source
    mark = time.perf_counter()
    counts['text_exact'] = candidate.build_text_exact_edges(frame, 20, edges)
    phases['text_exact_generate_merge'] = time.perf_counter() - mark
    mark = time.perf_counter()
    candidates = candidate.edges_to_frame(edges, frame)
    phases['candidate_materialize_sort'] = time.perf_counter() - mark
    mark = time.perf_counter()
    artifact = out / 'candidates.parquet'
    candidates.to_parquet(artifact, index=False)
    phases['candidate_parquet_write'] = time.perf_counter() - mark
    (stage_wall, stage_cpu) = (time.perf_counter() - start_wall, process_cpu() - start_cpu)
    receipt = {**job, 'completed_at_utc': stamp(), 'candidate_edges': len(candidates), 'channel_new_edges': counts, 'stage_wall_seconds': stage_wall, 'stage_cpu_seconds': stage_cpu, 'phase_wall_seconds': phases, 'peak_process_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 'candidate_file_bytes': artifact.stat().st_size, 'candidate_file': str(artifact), 'new_model_calls': 0, 'new_human_judgments': 0, 'label_access': False, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'pyarrow': pyarrow.__version__, 'faiss': getattr(candidate.faiss, '__version__', 'unknown'), 'faiss_threads': candidate.faiss.omp_get_max_threads(), 'thread_environment': {k: os.getenv(k) for k in THREAD_ENV}}}
    dump(out / 'receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('id', 'candidate_edges', 'stage_wall_seconds', 'peak_process_rss_gib')}), flush=True)

def run(args):
    contract = read(args.protocol)
    output = Path(contract['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for job in contract['jobs']:
        before_cpu = process_cpu(resource.RUSAGE_CHILDREN)
        before_wall = time.perf_counter()
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--protocol', str(args.protocol.resolve()), '--job', job['id']]
        with (output / (job['id'] + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **THREAD_ENV, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
        jobs.append(dict(**job, receipt=str(output / job['id'] / 'receipt.json'), process_wall_seconds=time.perf_counter() - before_wall, process_cpu_seconds=process_cpu(resource.RUSAGE_CHILDREN) - before_cpu))
    dump(output / 'execution_log.json', dict(jobs=jobs))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'run', 'worker'])
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--job')
    args = parser.parse_args()
    {'freeze': freeze, 'run': run, 'worker': worker}[args.action](args)
if __name__ == '__main__':
    main()
