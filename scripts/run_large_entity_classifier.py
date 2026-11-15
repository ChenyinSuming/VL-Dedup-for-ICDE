import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['GRAPH_CPU_THREADS'] = '4'
import argparse
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import time
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scripts import train_visual_classifier as visual_classifier
from scripts import large_entity_metrics as base
from scripts.cluster_with_size_cap import build_graph, edge_order, same_partition
from scripts.run_graph_repair import kwikcluster
(RES, REPORT, LARGE) = (base.RES, base.REPORT, base.LARGE)
classifier_run = RES / 'runs/visual_classifier'
SOURCE = RES / 'runs/pair_features'
SSCD_REFERENCE = RES / 'runs/edit_stress/original/sscd.npy'
SSCD_MODEL = RES / 'runs/VL-Dedup-runs/current/models/sscd/sscd_disc_mixup.torchscript.pt'
OUT = LARGE / 'augmentation_d_classifier_LR'
GRID = base.GRID
ENDS = ['record_id_a', 'record_id_b']
(dump, read, now) = (base.dump, base.read, base.now)

@contextmanager
def stage(name, gpu=False):
    (started, cpu) = (time.perf_counter(), time.process_time())
    item = {}
    try:
        yield item
    finally:
        item.update(wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu)
        dump(OUT / (name + '_stage.json'), item)

def structure(labels):
    sizes = np.unique(labels, return_counts=True)[1]
    return dict(global_components=len(sizes), largest_component=int(sizes.max()), U=sum((int(n) * (int(n) - 1) // 2 for n in sizes)))

def truth(graph, partition):
    return pd.read_parquet(LARGE / 'pairwise' / f'metadata_{graph}_{partition}.parquet')

def counts(labels, metadata, ids):
    return visual_classifier.pairwise_counts(labels, metadata, ids, exclude_m=True)

def original_cell(model, cap):
    return classifier_run / 'graphs/original' / model / str(cap)

def bundles():
    result = {m: joblib.load(classifier_run / (m + '.joblib')) for m in ['classifier_LR', 'classifier_HGB']}
    assert all((b['features'] == visual_classifier.FEATURES for b in result.values()))
    assert result['classifier_LR']['threshold'] == 2 / 3 and result['classifier_HGB']['threshold'] == 1.0
    return result

def encode():
    with stage('encode', gpu=True) as receipt:
        receipt['nvidia_smi_before_load'] = subprocess.check_output(['nvidia-smi']).decode()
        dump(OUT / 'encode_receipt.json', receipt)
        import torch
        from scripts.extract_features import image_loader
        torch.set_num_threads(4)
        frame = pd.read_json(LARGE / 'new_records.jsonl', lines=True)
        model = torch.jit.load(str(SSCD_MODEL), map_location='cuda').eval()
        batches = []
        with torch.inference_mode():
            for (images, ok) in image_loader(frame, ROOT, 288, 32, 0):
                assert all((bool(v) for v in ok))
                vectors = model(images.to('cuda'))
                if isinstance(vectors, (tuple, list)):
                    vectors = vectors[0]
                batches.append(torch.nn.functional.normalize(vectors.float(), dim=1).cpu().numpy())
                if len(batches) % 32 == 0:
                    print('SSCD', len(batches) * 32, '/', len(frame), flush=True)
        encoded = np.concatenate(batches)
        assert encoded.shape == (11000, 512) and encoded.dtype == np.float32 and np.isfinite(encoded).all()
        assert np.allclose(np.linalg.norm(encoded, axis=1), 1.0, atol=2e-06)
        np.save(OUT / 'sscd_new.npy', encoded)
        pd.DataFrame({'record_id': frame.record_id, 'row_index': np.arange(len(frame))}).to_parquet(OUT / 'sscd_new_row_ids.parquet', index=False)
        torch.cuda.synchronize()
        receipt.update(images=len(frame), checkpoint=str(SSCD_MODEL), dtype='float32', shape=list(encoded.shape), cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())

class ExpandedFeatures(visual_classifier.hgb_sscd.Features):

    def __init__(self, frame, image, text, phashes, sscd):
        self.frame = frame
        self.ids = pd.Index(frame.record_id)
        self.observable = visual_classifier.hgb_sscd.ObservableFeatures(frame, image, text, phashes)
        self.sscd = sscd

    def transform(self, pairs):
        return super().transform(pairs)[:, visual_classifier.FEATURE_COLUMNS]

def load_features():
    frame = pd.read_parquet(LARGE / 'public_manifest.parquet')
    ids = pd.Index(frame.record_id)
    original_ids = pd.Index(pd.read_parquet(SOURCE / 'public_manifest.parquet', columns=['record_id']).record_id)
    sidecar = pd.read_parquet(OUT / 'sscd_new_row_ids.parquet')
    assert np.array_equal(sidecar.row_index, np.arange(11000))
    full = np.empty((61000, 512), dtype=np.float32)
    for (row_ids, vectors) in [(original_ids, np.load(SSCD_REFERENCE, mmap_mode='r')), (pd.Index(sidecar.record_id), np.load(OUT / 'sscd_new.npy', mmap_mode='r'))]:
        positions = ids.get_indexer(row_ids)
        assert (positions >= 0).all() and vectors.shape == (len(row_ids), 512)
        full[positions] = vectors
    assert len(set(original_ids) | set(sidecar.record_id)) == 61000 and (not set(original_ids) & set(sidecar.record_id))
    assert np.isfinite(full).all() and np.allclose(np.linalg.norm(full, axis=1), 1.0, atol=2e-06)
    np.save(OUT / 'sscd_61000.npy', full)
    return ExpandedFeatures(frame, np.load(LARGE / 'features/clip/image_emb.npy', mmap_mode='r'), np.load(LARGE / 'features/clip/text_emb.npy', mmap_mode='r'), pd.read_parquet(LARGE / 'features/phash/phash.parquet').phash_int.to_numpy(dtype=np.uint64), full)

def graphs():
    with stage('graphs') as receipt:
        feature = load_features()
        bundle = bundles()['classifier_LR']
        pairs = pd.read_parquet(LARGE / 'candidates.parquet', columns=ENDS)
        (a, b) = [feature.ids.get_indexer(pairs[c]) for c in ENDS]
        assert len(pairs) == 1731972 and min(a.min(), b.min()) >= 0
        raw = np.empty(len(pairs), dtype=np.float64)
        calibrated = np.empty(len(pairs), dtype=np.float64)
        calls = 0
        for start in range(0, len(pairs), 32768):
            stop = min(start + 32768, len(pairs))
            x = feature.transform(pairs.iloc[start:stop])
            raw[start:stop] = visual_classifier.hgb_sscd.raw_probability(bundle['model'], x)
            calibrated[start:stop] = bundle['calibrator'].predict(raw[start:stop])
            calls += 1
            if start % (32768 * 10) == 0:
                print('score', start, '/', len(pairs), flush=True)
        assert np.isfinite(raw).all() and np.isfinite(calibrated).all()
        np.save(OUT / 'raw.npy', raw)
        np.save(OUT / 'calibrated.npy', calibrated)
        positive = np.flatnonzero(calibrated >= bundle['threshold'])
        order = edge_order(positive, calibrated, a, b, raw[positive])
        np.save(OUT / 'ordered_positive_rows.npy', order)
        cells = []
        for cap in GRID:
            (labels, kept, rejected, receipt) = build_graph(61000, a[order], b[order], cap)
            path = OUT / 'graphs' / str(cap)
            path.mkdir(parents=True)
            np.save(path / 'components.npy', labels)
            np.save(path / 'retained_candidate_indices.npy', order[kept])
            np.save(path / 'rejected_sizes.npy', rejected)
            if cap == 'inf':
                assert kept.all()
            cell = dict(model='classifier_LR', k=cap, structure=structure(labels), counts=receipt, artifacts=str(path))
            cells.append(cell)
            print('graph', cap, cell['structure'], flush=True)
        dump(OUT / 'graphs_frozen.json', dict(frozen_at_utc=now(), cells=cells))
        receipt.update(candidate_rows=len(pairs), positive_edges=len(positive), predict_calls=calls, model_rows=len(pairs), fit_calls=0, threshold=bundle['threshold'])

def select():
    with stage('select') as receipt:
        frozen = read(OUT / 'graphs_frozen.json')
        ids = pd.Index(pd.read_parquet(LARGE / 'public_manifest.parquet', columns=['record_id']).record_id)
        metadata = truth('large_entities', 'validation')
        rows = []
        opened = now()
        for cell in frozen['cells']:
            metric = counts(np.load(Path(cell['artifacts']) / 'components.npy'), metadata, ids)
            rows.append(dict(k=cell['k'], pairwise=metric, structure=cell['structure']))
        cap = visual_classifier.select_k(rows)
        selected = next((r for r in rows if r['k'] == cap))
        dump(OUT / 'validation.json', dict(rows=rows, first_computed_at_utc=opened))
        dump(OUT / 'selection_frozen.json', dict(k=cap, validation_cells=16, selected=selected, frozen_at_utc=now(), graphs_frozen_at_utc=frozen['frozen_at_utc']))
        receipt.update(selected_k=cap, validation_cells=16)
        print('selected', cap, selected['pairwise'], flush=True)

def bridge_blocks(n, a, b):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(zip(a.tolist(), b.tolist()))
    bridges = list(nx.bridges(graph))
    full = np.empty(n, dtype=np.int32)
    for component in nx.connected_components(graph):
        full[list(component)] = min(component)
    graph.remove_edges_from(bridges)
    blocks = np.empty(n, dtype=np.int32)
    for component in nx.connected_components(graph):
        blocks[list(component)] = min(component)
    return (full, blocks, bridges)

def repair_original(models):
    frame = pd.read_parquet(SOURCE / 'public_manifest.parquet')
    ids = pd.Index(frame.record_id)
    assert ids.is_monotonic_increasing
    pairs = pd.read_parquet(SOURCE / 'candidates.parquet', columns=ENDS)
    assert len(pairs) == 1502683
    (a, b) = [ids.get_indexer(pairs[c]) for c in ENDS]
    metadata = truth('original', 'test')
    (rows, bridges_report) = ([], [])
    for (model, bundle) in models.items():
        calibrated = np.load(classifier_run / ('original_' + model + '_calibrated.npy'), mmap_mode='r')
        assert calibrated.shape == (1502683,)
        selected = np.flatnonzero(calibrated >= bundle['threshold'])
        stored = np.load(original_cell(model, 'inf') / 'retained_candidate_indices.npy')
        assert np.array_equal(np.sort(stored), selected)
        (labels_kwik, rejected) = kwikcluster(50000, a[selected], b[selected])
        assert rejected == 0
        np.save(OUT / (model + '_kwik.npy'), labels_kwik)
        rows.append(dict(model=model, method='KwikCluster_ascending', k=None, pairwise=counts(labels_kwik, metadata, ids), structure=structure(labels_kwik)))
        for cap in [5, 'inf']:
            labels = np.load(original_cell(model, cap) / 'components.npy')
            rows.append(dict(model=model, method='union_find', k=cap, pairwise=counts(labels, metadata, ids), structure=structure(labels)))
        (original_labels, blocks, bridges) = bridge_blocks(50000, a[selected], b[selected])
        frozen = np.load(original_cell(model, 'inf') / 'components.npy')
        assert same_partition(original_labels, frozen)
        (full_counts, block_counts) = (counts(frozen, metadata, ids), counts(blocks, metadata, ids))
        separable = full_counts['fp'] - block_counts['fp']
        assert separable >= 0
        np.save(OUT / (model + '_bridge_blocks.npy'), blocks)
        np.save(OUT / (model + '_bridges.npy'), np.asarray(bridges, dtype=np.int64).reshape(-1, 2))
        bridges_report.append(dict(model=model, positive_edges=len(selected), global_bridges=len(bridges), false_pairs_union_find_inf=full_counts['fp'], false_pairs_inside_2edge_blocks=block_counts['fp'], false_pairs_separable_by_one_edge=separable, proportion=separable / full_counts['fp'] if full_counts['fp'] else 0.0))
    return (rows, bridges_report)

def test():
    selected = read(OUT / 'selection_frozen.json')
    ids = pd.Index(pd.read_parquet(LARGE / 'public_manifest.parquet', columns=['record_id']).record_id)
    metadata = truth('large_entities', 'test')
    rows = []
    for cell in read(OUT / 'graphs_frozen.json')['cells']:
        metric = counts(np.load(Path(cell['artifacts']) / 'components.npy'), metadata, ids)
        rows.append(dict(model='classifier_LR', k=cell['k'], pairwise=metric, structure=cell['structure']))
    (repairs, bridge) = repair_original(bundles())
    result = dict(selection=selected, validation=read(OUT / 'validation.json')['rows'], large_test=rows, original_repairs=repairs, single_edge=bridge)
    dump(OUT / 'results.json', result)
    dump(REPORT / 'augmentation_d_results.json', result)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['encode', 'graphs', 'select', 'test'])
    args = parser.parse_args()
    {'encode': encode, 'graphs': graphs, 'select': select, 'test': test}[args.phase]()
if __name__ == '__main__':
    main()
