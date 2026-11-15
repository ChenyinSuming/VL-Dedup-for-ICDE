import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['GRAPH_CPU_THREADS'] = '4'
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
RES = RESOURCES
RUNS = RES / 'runs'
OUT = RUNS / 'graph_baselines'
REPORT = ROOT / str(REPORTS / 'graph_baselines')
SOURCE = RUNS / 'pair_features'
visual_classifier = RUNS / 'visual_classifier'
MV = RUNS / 'sop_multiview'
LARGE = RUNS / 'large_entities'
native_baselines = RUNS / 'native_baselines'
edit_surfaces = RUNS / 'edit_stress'
ENDS = ['record_id_a', 'record_id_b']

def now():
    return datetime.now(timezone.utc).isoformat()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def read(path):
    return json.loads(Path(path).read_text())

@contextmanager
def stage(name):
    (started, cpu) = (time.perf_counter(), time.process_time())
    item = {}
    try:
        yield item
    finally:
        item.update(wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu)
        dump(OUT / (name + '_stage.json'), item)

def choose_two(counts):
    return sum((int(n) * (int(n) - 1) // 2 for n in counts))

def pairwise_counts(labels, groups):
    assert len(labels) == len(groups)
    tp = choose_two(Counter(zip(labels, groups)).values())
    predicted = choose_two(Counter(labels).values())
    positive = choose_two(Counter(groups).values())
    (fp, fn) = (predicted - tp, positive - tp)
    pairs = len(labels) * (len(labels) - 1) // 2
    tn = pairs - tp - fp - fn
    assert min(tp, fp, fn, tn) >= 0 and tp + fp + fn + tn == pairs
    f1 = Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0)
    return dict(records=len(labels), pairs=pairs, true_positive_pairs=positive, predicted_positive_pairs=predicted, tp=tp, fp=fp, fn=fn, tn=tn, f1=float(f1), exact_f1=str(f1))

def structure(labels):
    sizes = Counter(labels).values()
    return dict(clusters=len(sizes), largest_cluster=max(sizes), U=choose_two(sizes))

def metadata(surface, partition):
    if surface != 'MV':
        name = 'large_entities' if surface == 'large_entities' else 'original'
        return pd.read_parquet(LARGE / 'pairwise' / f'metadata_{name}_{partition}.parquet')
    pattern = re.compile('"partition"\\s*:\\s*"' + partition + '"')
    rows = []
    with (RES / 'datasets/benchmark/private/lineage.jsonl').open() as stream:
        for line in stream:
            if pattern.search(line):
                row = json.loads(line)
                rows.append({k: row[k] for k in ['record_id', 'partition', 'visual_group_id', 'original_level', 'source_dataset']})
    new = pd.read_parquet(MV / 'new_manifest.parquet')
    added = new[new.partition.eq(partition)].copy().assign(original_level='SOP_multiview', source_dataset='stanford_online_products')
    result = pd.concat([pd.DataFrame(rows), added], ignore_index=True)
    assert result.record_id.is_unique
    return result

def evaluate(labels, truth, ids):
    truth = truth[truth.original_level.ne('M')]
    pos = ids.get_indexer(truth.record_id)
    assert (pos >= 0).all()
    return pairwise_counts(labels[pos], truth.visual_group_id.to_numpy())

def surface_path(surface):
    return dict(original=SOURCE, MV=MV, large_entities=LARGE, mixed=native_baselines / 'mixed', mild=edit_surfaces / 'mild', strong=edit_surfaces / 'strong')[surface]

def inputs(surface, model):
    path = surface_path(surface)
    manifest_path = path / 'public_manifest.parquet' if surface in {'original', 'MV', 'large_entities'} else SOURCE / 'public_manifest.parquet'
    ids = pd.Index(pd.read_parquet(manifest_path, columns=['record_id']).record_id)
    assert ids.is_unique and ids.is_monotonic_increasing
    candidates = pd.read_parquet(path / 'candidates.parquet', columns=ENDS)
    (a, b) = [ids.get_indexer(candidates[c]) for c in ENDS]
    assert min(a.min(), b.min()) >= 0
    (lo, hi) = (np.minimum(a, b), np.maximum(a, b))
    if surface == 'large_entities':
        if model == 'classifier_LR':
            (score_path, raw_path) = (LARGE / 'augmentation_d_classifier_LR/calibrated.npy', LARGE / 'augmentation_d_classifier_LR/raw.npy')
        else:
            (score_path, raw_path) = (OUT / 'large_HGB_calibrated.npy', OUT / 'large_HGB_raw.npy')
    elif surface == 'MV':
        (score_path, raw_path) = (MV / 'visual_classifier_classifier' / f'visual_classifier_{model}_score.npy', MV / 'visual_classifier_classifier' / f'visual_classifier_{model}_raw.npy')
    else:
        score_path = visual_classifier / f'{surface}_{model}_calibrated.npy'
        raw_path = visual_classifier / f'{surface}_{model}_raw.npy'
    (score, raw) = (np.load(score_path, mmap_mode='r'), np.load(raw_path, mmap_mode='r'))
    assert score.shape == raw.shape == a.shape and np.isfinite(score).all() and np.isfinite(raw).all()
    selected = np.flatnonzero(score >= (2 / 3 if model == 'classifier_LR' else 1.0))
    selected = selected[np.lexsort((hi[selected], lo[selected], -raw[selected], -score[selected]))]
    return (ids, lo[selected], hi[selected], np.asarray(score[selected]), selected)

def canonical(labels):
    minima = {}
    for (i, label) in enumerate(labels):
        minima.setdefault(int(label), i)
    return np.asarray([minima[int(v)] for v in labels], dtype=np.int32)

def center(n, a, b, merge=False):
    labels = np.full(n, -1, dtype=np.int32)
    centers = np.zeros(n, dtype=bool)
    parent = list(range(n))

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for (u, v) in zip(a, b):
        (u, v) = (int(u), int(v))
        if labels[u] < 0 and labels[v] < 0:
            (u, v) = (min(u, v), max(u, v))
            centers[u] = True
            labels[u] = labels[v] = u
        elif labels[u] < 0 or labels[v] < 0:
            if labels[v] < 0:
                (u, v) = (v, u)
            if centers[v]:
                labels[u] = labels[v]
            else:
                centers[u] = True
                labels[u] = u
                if merge:
                    (x, y) = (find(u), find(int(labels[v])))
                    parent[max(x, y)] = min(x, y)
        elif merge and (centers[u] or centers[v]):
            (x, y) = (find(int(labels[u])), find(int(labels[v])))
            parent[max(x, y)] = min(x, y)
    for i in range(n):
        if labels[i] < 0:
            labels[i] = i
    return canonical([find(int(v)) for v in labels])

def mcl(n, a, b, weights, inflation):
    graph = sparse.csc_matrix((np.r_[weights, weights], (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    (_, components) = connected_components(graph, directed=False)
    order = np.argsort(components, kind='stable')
    cuts = np.r_[0, np.flatnonzero(np.diff(components[order])) + 1, n]
    labels = np.arange(n, dtype=np.int32)
    analysiss = dict(components=0, converged=0, unconverged=0, max_iterations=0)
    for (start, stop) in zip(cuts[:-1], cuts[1:]):
        nodes = np.sort(order[start:stop])
        size = len(nodes)
        if size == 1:
            continue
        analysiss['components'] += 1
        if size == 2:
            labels[nodes] = nodes[0]
            analysiss['converged'] += 1
            analysiss['max_iterations'] = max(analysiss['max_iterations'], 1)
            continue
        matrix = graph[nodes][:, nodes].tocsc()
        matrix.setdiag(np.asarray(matrix.max(axis=0).toarray()).ravel())
        matrix = matrix @ sparse.diags(1 / np.asarray(matrix.sum(axis=0)).ravel())
        converged = False
        dense = size <= 128
        current = matrix.toarray() if dense else matrix.tocsc()
        for iteration in range(1, 101):
            if dense:
                value = (current @ current) ** inflation
                value /= value.sum(axis=0)
                maximum = value.argmax(axis=0)
                keep = value >= 0.0001
                keep[maximum, np.arange(size)] = True
                value *= keep
                value /= value.sum(axis=0)
                converged = bool(np.allclose(value, current, rtol=1e-05, atol=1e-08))
            else:
                blocks = []
                for col in range(0, size, 128):
                    value = (current @ current[:, col:col + 128]).tocsc()
                    value.sort_indices()
                    value.data **= inflation
                    sums = np.add.reduceat(value.data, value.indptr[:-1])
                    value.data /= np.repeat(sums, np.diff(value.indptr))
                    keep = value.data >= 0.0001
                    for (left, right) in zip(value.indptr[:-1], value.indptr[1:]):
                        keep[left + value.data[left:right].argmax()] = True
                    value.data[~keep] = 0
                    value.eliminate_zeros()
                    sums = np.add.reduceat(value.data, value.indptr[:-1])
                    value.data /= np.repeat(sums, np.diff(value.indptr))
                    blocks.append(value)
                value = sparse.hstack(blocks, format='csc')
                difference = abs(value - current) - 1e-05 * abs(current)
                converged = not difference.nnz or difference.max() <= 1e-08
            current = value
            if converged:
                break
        analysiss['max_iterations'] = max(analysiss['max_iterations'], iteration)
        analysiss['converged' if converged else 'unconverged'] += 1
        attractors = np.flatnonzero(current.diagonal() > 0)
        assert len(attractors), 'MCL component has no attractor'
        if dense:
            flow = current[attractors]
            assert (flow.max(axis=0) > 0).all()
            assigned = attractors[flow.argmax(axis=0)]
        else:
            flow = current[attractors].tocsc()
            assert (flow.max(axis=0).toarray() > 0).all()
            assigned = attractors[np.asarray(flow.argmax(axis=0)).ravel()]
        labels[nodes] = nodes[assigned]
    return (canonical(labels), analysiss)

def cluster(n, a, b, weights, algorithm, parameter=None):
    if algorithm in {'CC', 'Kdagger'}:
        from scripts.cluster_with_size_cap import build_graph
        return (build_graph(n, a, b, 'inf' if algorithm == 'CC' else parameter)[0], {})
    if algorithm == 'KwikCluster':
        from scripts.run_graph_repair import kwikcluster
        return (kwikcluster(n, a, b)[0], {})
    if algorithm in {'Center', 'Merge-Center'}:
        return (center(n, a, b, merge=algorithm == 'Merge-Center'), {})
    if algorithm == 'MCL':
        return mcl(n, a, b, weights, parameter)
    if algorithm == 'Louvain':
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_weighted_edges_from(zip(map(int, a), map(int, b), map(float, weights)))
        groups = nx.community.louvain_communities(graph, weight='weight', resolution=parameter, seed=20_261_047, threshold=1e-07)
        labels = np.empty(n, dtype=np.int32)
        for group in groups:
            labels[list(group)] = min(group)
        return (labels, {})
    raise ValueError(algorithm)

def prepare_hgb():
    import joblib
    from scripts.run_large_entity_classifier import ExpandedFeatures
    from scripts.train_hgb_classifier import raw_probability
    with stage('large_HGB_scoring') as receipt:
        bundle = joblib.load(visual_classifier / 'classifier_HGB.joblib')
        frame = pd.read_parquet(LARGE / 'public_manifest.parquet')
        feature = ExpandedFeatures(frame, np.load(LARGE / 'features/clip/image_emb.npy', mmap_mode='r'), np.load(LARGE / 'features/clip/text_emb.npy', mmap_mode='r'), pd.read_parquet(LARGE / 'features/phash/phash.parquet').phash_int.to_numpy(dtype=np.uint64), np.load(LARGE / 'augmentation_d_classifier_LR/sscd_61000.npy', mmap_mode='r'))
        pairs = pd.read_parquet(LARGE / 'candidates.parquet', columns=ENDS)
        raw = np.empty(len(pairs))
        for start in range(0, len(pairs), 32768):
            stop = min(start + 32768, len(pairs))
            raw[start:stop] = raw_probability(bundle['model'], feature.transform(pairs.iloc[start:stop]))
        calibrated = bundle['calibrator'].predict(raw)
        np.save(OUT / 'large_HGB_raw.npy', raw)
        np.save(OUT / 'large_HGB_calibrated.npy', calibrated)
        receipt.update(candidate_rows=len(pairs), features=bundle['features'], new_fits=0, new_encodings=0)

def select(rows):
    return min(enumerate(rows), key=lambda item: (-Fraction(item[1]['metrics']['exact_f1']), item[1]['metrics']['predicted_positive_pairs'], item[0]))[1]

def validation_a1():
    protocol = read(REPORT / 'settings.json')
    for surface in protocol['edges']['surfaces']:
        truth = metadata(surface, 'validation')
        for model in ['classifier_LR', 'classifier_HGB']:
            name = f'validation_{surface}_{model}'
            stage_name = name
            with stage(stage_name):
                (ids, a, b, weights, ordered) = inputs(surface, model)
                output = dict(surface=surface, model=model, validation=[], selection={})
                for (algorithm, grid) in [('MCL', protocol['A1']['MCL']['inflation_grid']), ('Louvain', protocol['A1']['Louvain']['resolution_grid'])]:
                    rows = []
                    for parameter in grid:
                        (start, cpu) = (time.perf_counter(), time.process_time())
                        (labels, analysiss) = cluster(len(ids), a, b, weights, algorithm, parameter)
                        timing = dict(wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu)
                        target = OUT / 'graphs' / surface / model / algorithm / str(parameter)
                        target.mkdir(parents=True, exist_ok=True)
                        np.save(target / 'components.npy', labels)
                        row = dict(algorithm=algorithm, parameter=parameter, metrics=evaluate(labels, truth, ids), structure=structure(labels), analysiss=analysiss, timing=timing, components=str(target / 'components.npy'))
                        rows.append(row)
                        print(name, algorithm, parameter, row['metrics']['f1'], timing, flush=True)
                    output['validation'].extend(rows)
                    output['selection'][algorithm] = select(rows)
                dump(OUT / (name + '.json'), output)

def image_input(surface):
    if surface == 'MV':
        ids = pd.Index(pd.read_parquet(MV / 'public_manifest.parquet', columns=['record_id']).record_id)
        image = np.load(MV / 'image.npy', mmap_mode='r')
        assert np.array_equal(pd.read_parquet(MV / 'embedding_metadata.parquet').record_id, ids)
    else:
        ids = pd.Index(pd.read_parquet(SOURCE / 'public_manifest.parquet', columns=['record_id']).record_id)
        path = SOURCE / 'features/clip/image_emb.npy' if surface == 'original' else surface_path(surface) / 'clip.npy'
        image = np.load(path, mmap_mode='r')
    assert image.shape == (len(ids), 768) and np.isfinite(image).all()
    assert np.allclose(np.linalg.norm(image, axis=1), 1, atol=2e-06)
    return (ids, np.ascontiguousarray(image, dtype=np.float32))

def kmeans(image, k):
    import faiss
    faiss.omp_set_num_threads(4)
    model = faiss.Kmeans(image.shape[1], k, niter=50, nredo=1, seed=20_261_048, spherical=True, gpu=False, max_points_per_centroid=len(image), min_points_per_centroid=1)
    model.train(image)
    (_, assignments) = model.index.search(image, 1)
    return (assignments.ravel(), model.centroids)

def semdedup(image, assignments, epsilon):
    n = len(image)
    parent = list(range(n))

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    edges = 0
    for group in np.unique(assignments):
        nodes = np.flatnonzero(assignments == group)
        vectors = np.ascontiguousarray(image[nodes])
        for start in range(0, len(nodes), 512):
            similarity = vectors[start:start + 512] @ vectors.T
            (x, y) = np.nonzero(similarity >= 1 - epsilon)
            x += start
            keep = x < y
            for (u, v) in zip(nodes[x[keep]], nodes[y[keep]]):
                (a, b) = (find(int(u)), find(int(v)))
                parent[max(a, b)] = min(a, b)
            edges += int(keep.sum())
    return (canonical([find(i) for i in range(n)]), edges)

def validation_a2():
    protocol = read(REPORT / 'settings.json')['A2']['SemDeDup']
    (ids, image) = image_input('original')
    truth = metadata('original', 'validation')
    output = dict(validation=[], selection=None)
    for k in protocol['kmeans']['k_grid']:
        with stage('validation_SemDeDup_k' + str(k)) as receipt:
            (start, cpu) = (time.perf_counter(), time.process_time())
            (assignment, centroids) = kmeans(image, k)
            kt = dict(wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu)
            target = OUT / 'SemDeDup' / 'original' / str(k)
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / 'assignments.npy', assignment)
            np.save(target / 'centroids.npy', centroids)
            for epsilon in protocol['epsilon_grid']:
                (start, cpu) = (time.perf_counter(), time.process_time())
                (labels, edges) = semdedup(image, assignment, epsilon)
                timing = dict(wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu)
                path = target / (str(epsilon) + '_components.npy')
                np.save(path, labels)
                row = dict(k=k, epsilon=epsilon, metrics=evaluate(labels, truth, ids), structure=structure(labels), threshold_edges=edges, kmeans_timing=kt, threshold_timing=timing, components=str(path))
                output['validation'].append(row)
                dump(OUT / 'validation_SemDeDup.json', output)
                print('validation_SemDeDup', k, epsilon, row['metrics']['f1'], timing, flush=True)
            receipt['kmeans_timing'] = kt
    output['selection'] = select(output['validation'])
    dump(OUT / 'validation_SemDeDup.json', output)

def freeze():
    protocol = read(REPORT / 'settings.json')
    result = dict(run_id=OUT.name, frozen_at_utc=now(), A1={}, A2={})
    for surface in protocol['edges']['surfaces']:
        for model in ['classifier_LR', 'classifier_HGB']:
            data = read(OUT / f'validation_{surface}_{model}.json')
            assert len(data['validation']) == 14
            result['A1'][surface + '/' + model] = data['selection']
    result['A2']['SemDeDup'] = read(OUT / 'validation_SemDeDup.json')['selection']
    dump(REPORT / 'selection_frozen.json', result)

def test_a1():
    frozen = read(REPORT / 'selection_frozen.json')
    result = dict(run_id=OUT.name, cells=[])
    for surface in ['original', 'strong', 'MV', 'large_entities']:
        truth = metadata(surface, 'test')
        for model in ['classifier_LR', 'classifier_HGB']:
            with stage(f'test_{surface}_{model}'):
                (ids, a, b, weights, ordered) = inputs(surface, model)
                selection = frozen['A1'][surface + '/' + model]
                cap = 64 if surface == 'large_entities' else 5
                for algorithm in ['CC', 'Kdagger', 'KwikCluster', 'Center', 'Merge-Center', 'MCL', 'Louvain']:
                    parameter = cap if algorithm == 'Kdagger' else selection[algorithm]['parameter'] if algorithm in selection else None
                    if algorithm in selection:
                        path = Path(selection[algorithm]['components'])
                        labels = np.load(path)
                        (analysiss, timing) = (selection[algorithm]['analysiss'], selection[algorithm]['timing'])
                    else:
                        (start, cpu) = (time.perf_counter(), time.process_time())
                        (labels, analysiss) = cluster(len(ids), a, b, weights, algorithm, parameter)
                        timing = dict(wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu)
                        path = OUT / 'graphs' / surface / model / algorithm / 'components.npy'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(path, labels)
                    row = dict(surface=surface, model=model, algorithm=algorithm, parameter=parameter, metrics=evaluate(labels, truth, ids), structure=structure(labels), analysiss=analysiss, build_timing=timing, components=str(path))
                    row['known'] = known_test(labels, ids)
                    result['cells'].append(row)
                    dump(OUT / 'test_A1.json', result)
                    print('test', surface, model, algorithm, row['metrics']['f1'], flush=True)
    dump(OUT / 'test_A1.json', result)

def timing_a1():
    frozen = read(REPORT / 'selection_frozen.json')
    result = dict(runs=[])
    for model in ['classifier_LR', 'classifier_HGB']:
        (ids, a, b, weights, ordered) = inputs('original', model)
        selection = frozen['A1']['original/' + model]
        for algorithm in ['CC', 'Kdagger', 'KwikCluster', 'Center', 'Merge-Center', 'MCL', 'Louvain']:
            parameter = 5 if algorithm == 'Kdagger' else selection[algorithm]['parameter'] if algorithm in selection else None
            with stage(f'timing_{model}_{algorithm}'):
                reference = next((r for r in read(OUT / 'test_A1.json')['cells'] if r['surface'] == 'original' and r['model'] == model and (r['algorithm'] == algorithm)))
                expected = canonical(np.load(reference['components']))
                samples = []
                for repeat in range(3):
                    (start, cpu) = (time.perf_counter(), time.process_time())
                    (labels, analysiss) = cluster(len(ids), a, b, weights, algorithm, parameter)
                    samples.append(dict(wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu))
                    assert np.array_equal(canonical(labels), expected), 'Timing repeat changed partition'
                result['runs'].append(dict(model=model, algorithm=algorithm, parameter=parameter, samples=samples, median_cpu_seconds=float(np.median([r['cpu_seconds'] for r in samples]))))
                dump(OUT / 'timing_A1.json', result)

def known_test(labels, ids):
    pattern = re.compile('"partition"\\s*:\\s*"test"')
    with (RES / 'datasets/benchmark/sealed/relation_applicability.jsonl').open() as stream:
        pairs = pd.DataFrame([json.loads(line) for line in stream if pattern.search(line)])
    (a, b) = [ids.get_indexer(pairs[c]) for c in ENDS]
    assert min(a.min(), b.min()) >= 0
    connected = labels[a] == labels[b]
    mask = pairs.visual_relation_mask.to_numpy(dtype=bool)
    same = pairs.visual_relation.eq('same').to_numpy()
    assert mask.sum() == 9494 and (~mask).sum() == 543
    return dict(n=9494, fp=int((mask & ~same & connected).sum()), fn=int((mask & same & ~connected).sum()), unknown_n=543, unknown_connected=int((~mask & connected).sum()))

def corpus_metrics(labels, ids, surface):
    truth = metadata(surface, 'test')
    result = dict(metrics=evaluate(labels, truth, ids), structure=structure(labels), known=known_test(labels, ids))
    if surface == 'MV':
        result['sop_projection'] = evaluate(labels, truth[truth.source_dataset.eq('stanford_online_products')], ids)
    return result

def test_a2():
    frozen = read(REPORT / 'selection_frozen.json')
    selection = frozen['A2']['SemDeDup']
    result = dict(cells=[])
    for surface in ['original', 'mixed', 'mild', 'strong', 'MV']:
        (ids, image) = image_input(surface)
        if surface == 'original':
            labels = np.load(selection['components'])
            timing = dict(wall_seconds=selection['kmeans_timing']['wall_seconds'] + selection['threshold_timing']['wall_seconds'], cpu_seconds=selection['kmeans_timing']['cpu_seconds'] + selection['threshold_timing']['cpu_seconds'])
            path = Path(selection['components'])
        else:
            with stage('SemDeDup_' + surface):
                (start, cpu) = (time.perf_counter(), time.process_time())
                (assignment, centroids) = kmeans(image, selection['k'])
                (labels, edges) = semdedup(image, assignment, selection['epsilon'])
                timing = dict(wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu)
                target = OUT / 'SemDeDup' / surface
                target.mkdir(parents=True, exist_ok=True)
                np.save(target / 'assignments.npy', assignment)
                np.save(target / 'centroids.npy', centroids)
                path = target / 'components.npy'
                np.save(path, labels)
        row = dict(method='SemDeDup', surface=surface, k=selection['k'], epsilon=selection['epsilon'], components=str(path), timing=timing, **corpus_metrics(labels, ids, surface))
        result['cells'].append(row)
        dump(OUT / 'test_A2_SemDeDup.json', result)
        print('test SemDeDup', surface, row['metrics']['f1'], flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare_hgb', 'validation_a1', 'validation_a2', 'freeze', 'test_a1', 'timing_a1', 'test_a2'])
    args = parser.parse_args()
    {'prepare_hgb': prepare_hgb, 'validation_a1': validation_a1, 'validation_a2': validation_a2, 'freeze': freeze, 'test_a1': test_a1, 'timing_a1': timing_a1, 'test_a2': test_a2}[args.phase]()
if __name__ == '__main__':
    main()
