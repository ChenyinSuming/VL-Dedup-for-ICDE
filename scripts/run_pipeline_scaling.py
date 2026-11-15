from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, REPORTS
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
ROOT = REPO_ROOT
REPORT = ROOT / str(REPORTS / 'pipeline_scaling')
PROTOCOL = REPORT / 'settings.json'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'ARROW_NUM_THREADS', 'GRAPH_CPU_THREADS'):
    os.environ[key] = '8'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

def stamp():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)

def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime

def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2

def review_counts(labels):
    sizes = Counter(map(int, labels))
    return dict(clusters=len(sizes), max_cluster=max(sizes.values(), default=0), U=sum((s * (s - 1) // 2 for s in sizes.values())))

def visual_classifier_features(frame, image, text, phashes, sscd):
    import pandas as pd
    from scripts import train_visual_classifier as visual_classifier
    feature = visual_classifier.Features.__new__(visual_classifier.Features)
    base = visual_classifier.hgb_sscd.Features.__new__(visual_classifier.hgb_sscd.Features)
    base.frame = frame
    base.ids = pd.Index(frame.record_id)
    base.observable = visual_classifier.hgb_sscd.ObservableFeatures(frame, image, text, phashes)
    base.sscd = sscd
    (feature.base, feature.ids) = (base, base.ids)
    return feature

def below_threshold_counts(labels, scores, threshold):
    import numpy as np
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or not np.isfinite(scores).all() or (not np.isfinite(threshold)):
        raise ValueError('Invalid score/label alignment or non-finite threshold')
    sizes = Counter(map(int, labels))
    high = Counter((int(label) for (label, score) in zip(labels, scores) if score >= threshold))
    total = sum((n * (n - 1) // 2 for n in sizes.values()))
    calls = sum((n * (n - 1) // 2 for n in high.values()))
    return dict(U=total, below_t_new=total - calls, vlm_call_pairs=calls)

def worker(n, repeat):
    import joblib
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from threadpoolctl import threadpool_info, threadpool_limits
    from scripts.pair_features import ObservableFeatures
    from scripts.cluster_with_size_cap import build_graph, edge_order
    p = read(PROTOCOL)
    released_model = p.get('released_model', False)
    if released_model:
        from scripts import train_visual_classifier as visual_classifier
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'GRAPH_CPU_THREADS'):
            os.environ[key] = '8'
    out = Path(p['output_root'])
    job = f'n{n}_r{repeat}'
    path = out / f'{job}.json'
    trial = dict(job=job, records=n, repeat=repeat, started_at_utc=stamp(), labels_opened=False)
    dump(path, trial)
    pa.set_cpu_count(8)
    pa.set_io_thread_count(8)
    cache = Path(p['cache_root'])
    bundle = joblib.load(cache / p['model'])
    assert bundle['features'] == p['features']
    if released_model:
        assert bundle['threshold'] == p['threshold'] and visual_classifier.FEATURES == p['features']
    names = list(bundle['model'].classes_)
    positive = [names.index(name) for name in ['duplicate', 'keep_augment_candidate']]
    candidates = pd.read_parquet(Path(p['retrieval_root']) / f'n{n}_r1/candidates.parquet')
    with threadpool_limits(limits=8):
        trial['threadpools'] = threadpool_info()
        trial['arrow_threads'] = dict(cpu=pa.cpu_count(), io=pa.io_thread_count())
        if released_model:
            (wall, cpu) = (time.perf_counter(), cpu_seconds())
            sscd_members = pd.read_csv(Path(p['retrieval_root']) / p['membership'])
            sscd_rows = sscd_members[sscd_members.selection_rank <= n].sort_values('cache_row').cache_row.to_numpy(dtype=int)
            full_sscd = np.load(p['sscd_cache'])
            assert full_sscd.shape == (50000, 512) and full_sscd.dtype == np.float32
            sscd = full_sscd[sscd_rows]
            assert np.allclose(np.linalg.norm(sscd, axis=1), 1.0, atol=2e-06)
            del full_sscd
            trial.update(sscd_load_wall_seconds=time.perf_counter() - wall, sscd_load_cpu_seconds=cpu_seconds() - cpu)
        (wall, cpu) = (time.perf_counter(), cpu_seconds())
        members = pd.read_csv(Path(p['retrieval_root']) / p['membership'], dtype={'record_id': str})
        selected = members[members.selection_rank <= n].sort_values('cache_row')
        indices = selected.cache_row.to_numpy(dtype=int)
        public = pd.read_parquet(cache / 'public_manifest.parquet')
        frame = public.iloc[indices].copy().reset_index(drop=True)
        assert len(frame) == n and frame.record_id.tolist() == selected.record_id.tolist()
        image = np.load(cache / 'features/clip/image_emb.npy', mmap_mode='r')[indices]
        text = np.load(cache / 'features/clip/text_emb.npy', mmap_mode='r')[indices]
        hashes = pd.read_parquet(cache / 'features/phash/phash.parquet')
        assert np.array_equal(hashes.record_id, public.record_id)
        phashes = hashes.phash_int.to_numpy(dtype=np.uint64)[indices]
        if released_model:
            assert np.array_equal(sscd_rows, indices)
            feature = visual_classifier_features(frame, image, text, phashes, sscd)
            (raw, calibrated) = [np.empty(len(candidates), dtype=np.float64) for _ in range(2)]
            chunk = p['prediction_chunk_rows']
            for start in range(0, len(candidates), chunk):
                end = min(start + chunk, len(candidates))
                x = feature.transform(candidates.iloc[start:end])
                raw[start:end] = visual_classifier.hgb_sscd.raw_probability(bundle['model'], x)
                calibrated[start:end] = bundle['calibrator'].predict(raw[start:end])
        else:
            feature = ObservableFeatures(frame, image, text, phashes)
            x = feature.transform(candidates)
            raw = bundle['model'].predict_proba(x)[:, positive].sum(axis=1)
            calibrated = bundle['calibrator'].predict(raw)
        trial.update(C_wall_seconds=time.perf_counter() - wall, C_cpu_seconds=cpu_seconds() - cpu, C_peak_rss_gib=rss_gib())
        (wall, cpu) = (time.perf_counter(), cpu_seconds())
        chosen = np.flatnonzero(calibrated >= p['threshold'])
        ids = sorted(frame.record_id.tolist())
        mapping = dict(zip(ids, range(n)))
        a = candidates.record_id_a.map(mapping).to_numpy(dtype=int)
        b = candidates.record_id_b.map(mapping).to_numpy(dtype=int)
        order = edge_order(chosen, calibrated, a, b, raw[chosen])
        (labels, retained, rejected, graph) = build_graph(n, a[order], b[order], p['cap'])
        trial.update(D_wall_seconds=time.perf_counter() - wall, D_cpu_seconds=cpu_seconds() - cpu, D_peak_rss_gib=rss_gib())
        (wall, cpu) = (time.perf_counter(), cpu_seconds())
        (unlimited, _, _, infinity_graph) = build_graph(n, a[order], b[order], 'inf')
        trial.update(control_wall_seconds=time.perf_counter() - wall, control_cpu_seconds=cpu_seconds() - cpu)
        counts = dict(candidates=len(candidates), S=len(chosen), **review_counts(labels), U_infinity=review_counts(unlimited)['U'])
        assert counts['max_cluster'] <= p['cap']
        assert 2 * counts['U'] <= n * (p['cap'] - 1)
        extra = {}
        if released_model:
            counts.update({k: graph[k] for k in ['union_merges', 'same_root_redundant_edges', 'cap_rejected_edges']})
            extra['clip_scores'] = pd.Series(feature.base.observable.scores, index=frame.record_id).loc[ids].to_numpy()
            if n == 50000 and repeat == 1:
                np.save(out / 'n50000_r1_raw.npy', raw)
                np.save(out / 'n50000_r1_calibrated.npy', calibrated)
        trial.update(counts=counts, graph=graph, infinity_graph=infinity_graph, infinity_counts=review_counts(unlimited))
        np.savez_compressed(out / f'{job}_graph.npz', a=a[order], b=b[order], labels=labels, infinity_labels=unlimited, selected_source_rows=order, **extra)
        dump(path, trial)
    trial.update(finished_at_utc=stamp(), process_cpu_seconds=cpu_seconds(), process_peak_rss_gib=rss_gib())
    dump(path, trial)
    print(job, json.dumps(counts), flush=True)

def run_trials():
    p = read(PROTOCOL)
    out = Path(p['output_root'])
    out.mkdir(parents=True, exist_ok=True)
    trials = []
    for (repeat, sizes) in enumerate(p['orders'], 1):
        for n in sizes:
            job = f'n{n}_r{repeat}'
            with (out / f'{job}.log').open('w') as log:
                subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), 'worker', '--records', str(n), '--repeat', str(repeat), '--protocol', str(PROTOCOL)], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=True)
            trials.append(dict(job=job))
    dump(out / 'execution_receipt.json', dict(trials=trials))
    summarize()

def summarize():
    p = read(PROTOCOL)
    released_model = p.get('released_model', False)
    out = Path(p['output_root'])
    trials = [read(out / f'n{n}_r{r}.json') for (r, sizes) in enumerate(p['orders'], 1) for n in sizes]
    (rows, summaries) = ([], [])
    fields = ['C_wall_seconds', 'C_cpu_seconds', 'C_peak_rss_gib', 'D_wall_seconds', 'D_cpu_seconds', 'D_peak_rss_gib', 'control_wall_seconds', 'control_cpu_seconds', 'process_cpu_seconds', 'process_peak_rss_gib']
    for t in trials:
        rows.append(dict(job=t['job'], records=t['records'], repeat=t['repeat'], **{key: t[key] for key in fields}, **t['counts']))
    for n in sorted(set((t['records'] for t in trials))):
        group = [t for t in trials if t['records'] == n]
        assert all((t['counts'] == group[0]['counts'] for t in group))
        if released_model:
            import numpy as np
            with np.load(out / f'n{n}_r1_graph.npz') as first:
                for repeat in (2, 3):
                    with np.load(out / f'n{n}_r{repeat}_graph.npz') as other:
                        assert np.array_equal(first['labels'], other['labels'])
        summaries.append(dict(records=n, repeats=len(group), counts=group[0]['counts'], timings={key: dict(median=statistics.median((t[key] for t in group)), min=min((t[key] for t in group)), max=max((t[key] for t in group))) for key in fields}))
    with (REPORT / p.get('trials_file', 'trials.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    receipt = read(out / 'execution_receipt.json')
    result = dict(run_id=p['run_id'], trials=12, summaries=summaries, stage_wall_seconds=sum((t['C_wall_seconds'] + t['D_wall_seconds'] for t in trials)), process_cpu_seconds=sum((t['process_cpu_seconds'] for t in trials)), peak_process_rss_gib=max((t['process_peak_rss_gib'] for t in trials)), gpu_seconds=0, vlm_calls=0, api_calls=0, labels_opened=False)
    if released_model:
        result.update(model=p['model_label'], threshold=p['threshold'], cap=p['cap'], sscd_load_wall_seconds=sum((t['sscd_load_wall_seconds'] for t in trials)), sscd_load_cpu_seconds=sum((t['sscd_load_cpu_seconds'] for t in trials)))
        if (out / 'routing_receipt.json').exists():
            result.update(routing=read(out / 'routing_receipt.json'))
    dump(REPORT / p.get('result_file', 'results.json'), result)

def routing(wait=False):
    import numpy as np
    p = read(PROTOCOL)
    out = Path(p['output_root'])
    path = Path(p['routing']['threshold_file'])
    receipt = out / 'routing_receipt.json'
    frozen = read(path)
    threshold = frozen['t_new']
    assert isinstance(threshold, (int, float)) and np.isfinite(threshold)
    scales = []
    for n in sorted(p['budget']['candidates'], key=int):
        with np.load(out / f'n{n}_r1_graph.npz') as archive:
            counts = below_threshold_counts(archive['labels'], archive['clip_scores'], threshold)
        trial = read(out / f'n{n}_r1.json')
        assert counts['U'] == trial['counts']['U']
        scales.append(dict(records=int(n), **counts))
    dump(receipt, dict(source=str(path), t_new=threshold, counted_at_utc=stamp(), rule=p['routing']['rule'], scales=scales))
    summarize()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'worker', 'summarize', 'routing'])
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--records', type=int, choices=[10000, 20000, 30000, 50000])
    parser.add_argument('--repeat', type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    PROTOCOL = args.protocol.resolve()
    if args.command == 'run':
        run_trials()
    elif args.command == 'worker':
        worker(args.records, args.repeat)
    elif args.command == 'routing':
        routing()
    else:
        summarize()
