import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '8'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from scripts import run_native_baselines_50k as native_baselines
CACHE = RESOURCES / 'runs/sop_multiview'
OUT = CACHE / 'external_baselines'
REPORT = ROOT / str(REPORTS / 'sop_multiview/external_baselines')
RUN_ID = 'sop_multiview'
METHODS = ['Data-Juicer', 'SemHash']
dump = native_baselines.dump

def read(path):
    return json.loads(Path(path).read_text())

def fused_embeddings(image, text):
    (image, text) = (np.asarray(image).copy(), np.asarray(text).copy())
    assert image.ndim == text.ndim == 2 and image.shape == text.shape
    assert np.isfinite(image).all() and np.isfinite(text).all()
    assert (np.linalg.norm(image, axis=1) > 0).all() and (np.linalg.norm(text, axis=1) > 0).all()
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    fused = np.concatenate([image / np.sqrt(2), text / np.sqrt(2)], axis=1).astype(np.float32)
    fused /= np.linalg.norm(fused, axis=1, keepdims=True)
    return fused

def hash_components(hashes, selected):
    assert all((isinstance(h, str) and h for h in hashes))
    roots = pd.factorize(pd.Series(hashes), sort=False)[0]
    first = ~pd.Series(hashes).duplicated().to_numpy()
    assert np.array_equal(np.asarray(selected, dtype=bool), first), 'Official selection and native hash groups disagree'
    return roots

def native_links(result, n):
    selected = np.asarray([r['row_index'] for r in result.selected], dtype=np.int64)
    (filtered, exact, links, journal) = ([], [], [], [])
    for duplicate in result.filtered:
        a = duplicate.record['row_index']
        filtered.append(a)
        entries = []
        for (other, score) in duplicate.duplicates:
            b = other['row_index']
            (exact if duplicate.exact else links).append((a, b, float(score)))
            entries.append(dict(row_index=b, score=float(score)))
        journal.append(dict(row_index=a, exact=bool(duplicate.exact), duplicates=entries))
    filtered = np.asarray(filtered, dtype=np.int64)
    assert np.array_equal(np.sort(np.r_[selected, filtered]), np.arange(n))
    exact = np.asarray(exact, dtype=np.float64).reshape(-1, 3)
    links = np.asarray(links, dtype=np.float64).reshape(-1, 3)
    combined = np.concatenate([links[:, :2], exact[:, :2]]).astype(np.int64)
    assert ((combined >= 0) & (combined < n)).all()
    roots = native_baselines.components(n, combined[:, 0], combined[:, 1])
    return (roots, dict(selected=selected, filtered=filtered, native_similarity_link=links, exact_relation=exact), journal)

def structure(components, positions=None):
    labels = np.asarray(components) if positions is None else np.asarray(components)[positions]
    sizes = np.unique(labels, return_counts=True)[1]
    return dict(records=len(labels), components=len(sizes), largest_component=int(sizes.max()), U=sum((int(n) * (int(n) - 1) // 2 for n in sizes)))

def load_test_truth(parent):
    base = parent.reference.read_partition(parent.reference.DATA / 'private/lineage.jsonl', 'test')
    truth = parent.combine_records(base, pd.read_parquet(CACHE / 'new_manifest.parquet'), 'test')
    pairs = parent.reference.read_partition(parent.reference.DATA / 'sealed/relation_applicability.jsonl', 'test')
    return (truth, pairs)

def known_metrics(components, pairs, ids, parent):
    mask = pairs.visual_relation_mask.to_numpy(dtype=bool)
    (a, b) = [ids.get_indexer(pairs[c]) for c in parent.ENDS]
    assert (a >= 0).all() and (b >= 0).all()
    connected = components[a] == components[b]
    return dict(component=parent.hgb_sscd.confusion(pairs.visual_relation.eq('same').to_numpy()[mask], connected[mask]), unknown=dict(n=int((~mask).sum()), connected=int(connected[~mask].sum())))

def measure(components, truth, pairs, ids, parent):
    non_m = truth[truth.original_level.ne('M')]
    sop = truth[truth.source_dataset.eq('stanford_online_products')]
    return dict(primary_pairwise=parent.visual_classifier.pairwise_counts(components, truth, ids), pairwise_sop=parent.visual_classifier.pairwise_counts(components, sop, ids), original_known=known_metrics(components, pairs, ids, parent), structure=structure(components), nonM_test_structure=structure(components, ids.get_indexer(non_m.record_id)), SOP_test_structure=structure(components, ids.get_indexer(sop.record_id)))

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(CACHE / 'public_manifest.parquet')
    ids = pd.Index(frame.record_id)
    assert len(frame) == 58550 and ids.is_unique and ids.is_monotonic_increasing
    meta = pd.read_parquet(CACHE / 'embedding_metadata.parquet')
    assert np.array_equal(meta.record_id, ids) and np.array_equal(meta.row_index, np.arange(len(frame)))
    frame.assign(row_index=np.arange(len(frame))).to_parquet(OUT / 'input_mapping.parquet', index=False)

def data_juicer():
    with threadpool_limits(limits=8):
        OUT.mkdir(parents=True, exist_ok=True)
        (operator, source) = native_baselines.official_dj()
        frame = pd.read_parquet(OUT / 'input_mapping.parquet')
        rows = native_baselines.NativeDataset()
        journal = OUT / 'data_juicer_rows.jsonl'
        with journal.open('w') as stream:
            for (i, row) in enumerate(frame.itertuples()):
                sample = dict(record_id=row.record_id, text=row.text, images=[row.image_path])
                try:
                    native = operator.compute_hash(sample)
                    if not native.get('__dj__imagehash'):
                        raise ValueError('Empty native image hash')
                except Exception as exc:
                    raise
                rows.append(native)
                stream.write(json.dumps(native) + '\n')
                if i % 5000 == 0:
                    print('Data-Juicer', i, '/', len(frame), flush=True)
        (selected, _) = operator.process(rows)
        selected_ids = {r['record_id'] for r in selected}
        native = pd.DataFrame(dict(record_id=frame.record_id, native_hash=[r['__dj__imagehash'] for r in rows]))
        native['selected'] = native.record_id.isin(selected_ids)
        labels = hash_components(native.native_hash.tolist(), native.selected.to_numpy())
        assert len(labels) == len(frame) == 58550
        native.to_parquet(OUT / 'data_juicer_native.parquet', index=False)
        np.save(OUT / 'data_juicer_components.npy', labels)

def semhash():
    from semhash import SemHash
    with threadpool_limits(limits=8):
        OUT.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(OUT / 'input_mapping.parquet')
        records = [dict(record_id=r.record_id, row_index=i, image=str(r.image_path), text=r.text) for (i, r) in enumerate(frame.itertuples())]
        fused = fused_embeddings(np.load(CACHE / 'image.npy', mmap_mode='r'), np.load(CACHE / 'text.npy', mmap_mode='r'))
        index = SemHash.from_embeddings(embeddings=fused, records=records, model=native_baselines.NoEncoding(), columns=['image', 'text'], ann_backend='basic')
        print('SemHash index', len(index.index.items), flush=True)
        result = index.self_deduplicate(threshold=0.9)
        (labels, native, journal) = native_links(result, len(frame))
        np.savez_compressed(OUT / 'semhash_0.9_native.npz', **native)
        with (OUT / 'semhash_0.9_filtered.jsonl').open('w') as stream:
            for row in journal:
                stream.write(json.dumps(row) + '\n')
        np.save(OUT / 'semhash_0.9_components.npy', labels)

def freeze():
    frame = pd.read_parquet(OUT / 'input_mapping.parquet')
    paths = dict(zip(METHODS, ['data_juicer_components.npy', 'semhash_0.9_components.npy']))
    for (name, filename) in paths.items():
        assert np.load(OUT / filename, mmap_mode='r').shape == (len(frame),)
    dump(OUT / 'graphs_frozen.json', dict(all_external_outputs_frozen=True, records=len(frame), graphs=paths, frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))

def evaluate():
    from scripts import run_multiview_classifier as parent
    frozen = read(OUT / 'graphs_frozen.json')
    with threadpool_limits(limits=8):
        OUT.mkdir(parents=True, exist_ok=True)
        ids = pd.Index(pd.read_parquet(OUT / 'input_mapping.parquet').record_id)
        (truth, pairs) = load_test_truth(parent)
        rows = []
        for (name, filename) in frozen['graphs'].items():
            value = measure(np.load(OUT / filename), truth, pairs, ids, parent)
            rows.append(dict(name=name, role='native', k='unbounded', graph_artifacts=str(OUT / filename), **value))
        previous = read(CACHE / 'visual_classifier_classifier/results.json')
        for name in ['B2', 'B6', 'visual_classifier_classifier_LR']:
            row = next((r for r in previous['rows'] if r['name'] == name and r['role'] == 'K_dagger'))
            value = measure(np.load(Path(row['graph_artifacts']) / 'components.npy'), truth, pairs, ids, parent)
            assert value['primary_pairwise'] == row['primary_pairwise'] and value['pairwise_sop'] == row['pairwise_sop']
            rows.append(dict(name=name, role='frozen_K5_reference', k=5, graph_artifacts=row['graph_artifacts'], **value))
        for row in rows:
            for (key, n, positive) in [('primary_pairwise', 8044, 11685), ('pairwise_sop', 3154, 2984)]:
                m = row[key]
                assert m['records'] == n and m['pairs'] == n * (n - 1) // 2 and (m['tp'] + m['fn'] == positive)
            assert row['original_known']['component']['n'] == 9494 and row['original_known']['unknown']['n'] == 543
        dump(OUT / 'results.json', dict(run_id=RUN_ID, scope='Fixed native external comparator global graphs on frozen COCO+MV inputs', records=58550, new_multiview_records_reused=8550, rows=rows, primary_definition='Exclude original_level=M records only from evaluation, retaining global M bridge nodes', inputs=str(OUT / 'input_mapping.parquet')))
    dump(REPORT / 'results.json', read(OUT / 'results.json'))

def main():
    import pyarrow as pa
    pa.set_cpu_count(8)
    pa.set_io_thread_count(8)
    parser = argparse.ArgumentParser(description=__doc__)
    phases = {'prepare': prepare, 'dj': data_juicer, 'semhash': semhash, 'freeze': freeze, 'evaluate': evaluate}
    parser.add_argument('--phase', required=True, choices=list(phases))
    args = parser.parse_args()
    phases[args.phase]()
if __name__ == '__main__':
    main()
