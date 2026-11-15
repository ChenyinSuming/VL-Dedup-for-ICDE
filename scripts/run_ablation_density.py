import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, REPORTS
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['GRAPH_CPU_THREADS'] = '4'
import argparse
from contextlib import contextmanager
from fractions import Fraction
import json
from pathlib import Path
import re
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from scripts import train_visual_classifier as visual_classifier
from scripts import run_multiview_classifier as mv
from scripts import run_large_entity_classifier as large
hgb_sscd = visual_classifier.hgb_sscd
RES = visual_classifier.RES
OUT = RES / 'runs/ablation_density'
REPORT = ROOT / str(REPORTS / 'ablation_density')
PROTOCOL = json.loads((REPO_ROOT / 'scripts/config/ablation_density.json').read_text())
(FEATURES, GRID, ENDS) = (visual_classifier.FEATURES, visual_classifier.GRID, visual_classifier.ENDS)
GROUPS = PROTOCOL['groups']
(dump, read) = (visual_classifier.dump, visual_classifier.read)

def feature_columns():
    groups = {k: [FEATURES.index(c) for c in v] for (k, v) in GROUPS.items()}
    all_columns = list(range(len(FEATURES)))
    result = {'classifier': all_columns}
    for (group, columns) in groups.items():
        result['without_' + group] = [c for c in all_columns if c not in columns]
    result['image_only'] = sorted(groups['SSCD'] + groups['pHash'] + groups['CLIP_image'])
    result['text_only'] = groups['text']
    assert list(result) == PROTOCOL['variants']
    return result

def model_input(x, columns):
    return x if columns == list(range(len(FEATURES))) else x[:, columns]

@contextmanager
def stage(name):
    (started, cpu) = (time.perf_counter(), time.process_time())
    item = {}
    try:
        yield item
    finally:
        item.update(wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu)
        dump(OUT / (name + '_stage.json'), item)

def reference(partition, lineage=False):
    path = visual_classifier.DATA / ('private/lineage.jsonl' if lineage else 'sealed/relation_applicability.jsonl')
    pattern = re.compile('"partition"\\s*:\\s*"' + partition + '"')
    return pd.DataFrame([json.loads(line) for line in path.open() if pattern.search(line)])

def records(surface, partition):
    base = reference(partition, True)
    if surface == 'MV':
        return mv.combine_records(base, pd.read_parquet(mv.CACHE / 'new_manifest.parquet'), partition)
    if surface == 'expanded':
        return pd.read_parquet(large.LARGE / 'pairwise' / f'metadata_large_entities_{partition}.parquet')
    return base

def features(surface):
    if surface == 'MV':
        return mv.Features()
    if surface == 'expanded':
        frame = pd.read_parquet(large.LARGE / 'public_manifest.parquet')
        return large.ExpandedFeatures(frame, np.load(large.LARGE / 'features/clip/image_emb.npy', mmap_mode='r'), np.load(large.LARGE / 'features/clip/text_emb.npy', mmap_mode='r'), pd.read_parquet(large.LARGE / 'features/phash/phash.parquet').phash_int, np.load(large.OUT / 'sscd_61000.npy', mmap_mode='r'))
    return visual_classifier.Features(surface)

def candidate_path(surface):
    if surface == 'MV':
        return mv.CACHE / 'candidates.parquet'
    if surface == 'expanded':
        return large.LARGE / 'candidates.parquet'
    return hgb_sscd.source_path(surface) / 'candidates.parquet'

def candidates(surface, ids):
    pairs = pd.read_parquet(candidate_path(surface), columns=ENDS)
    (a, b) = [ids.get_indexer(pairs[c]) for c in ENDS]
    assert min(a.min(), b.min()) >= 0 and (a != b).all()
    code = np.minimum(a, b).astype(np.int64) * len(ids) + np.maximum(a, b)
    assert len(np.unique(code)) == len(code)
    expected = dict(original=1502683, mixed=1503362, mild=1504724, strong=1502221, MV=1771452, expanded=1731972)
    assert len(pairs) == expected[surface], (surface, len(pairs))
    return (pairs, a, b, code)

def score(surface, feature, pairs, bundles):
    arrays = {name: {kind: np.lib.format.open_memmap(OUT / f'{surface}_{name}_{kind}.npy', mode='w+', dtype=np.float64, shape=(len(pairs),)) for kind in ['raw', 'calibrated']} for name in bundles}
    for start in range(0, len(pairs), 32768):
        x = feature.transform(pairs.iloc[start:start + 32768])
        for (name, bundle) in bundles.items():
            raw = hgb_sscd.raw_probability(bundle['model'], model_input(x, bundle['columns']))
            arrays[name]['raw'][start:start + len(x)] = raw
            arrays[name]['calibrated'][start:start + len(x)] = bundle['calibrator'].predict(raw)
        if start % (32768 * 10) == 0:
            print('score', surface, start, flush=True)
    for item in arrays.values():
        for array in item.values():
            array.flush()
    return arrays

def ordered_edges(bundle, arrays, a, b):
    (raw, calibrated) = (arrays['raw'], arrays['calibrated'])
    selected = np.flatnonzero(calibrated >= bundle['threshold'])
    return hgb_sscd.edge_order(selected, calibrated, a, b, raw[selected])

def save_graph(surface, name, cap, n, a, b, order):
    labels = hgb_sscd.build_graph(n, a[order], b[order], cap)[0]
    path = OUT / 'graphs' / surface / name / str(cap)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'components.npy', labels)
    return labels

def graph(surface, name, cap):
    return np.load(OUT / 'graphs' / surface / name / str(cap) / 'components.npy')

def counts(labels, truth, ids):
    return visual_classifier.pairwise_counts(labels, truth, ids, exclude_m=True)

def universe(truth):
    truth = truth[truth.original_level.ne('M')]
    n = len(truth)
    return dict(records=n, pairs=n * (n - 1) // 2, positives=sum((int(c) * (int(c) - 1) // 2 for c in truth.visual_group_id.value_counts())))

def train():
    with stage('ablation_train') as receipt:
        (train, val) = (reference('train'), reference('validation'))
        assert len(train) == 46425 and val.visual_relation_mask.sum() == 9766
        feature = features('original')
        truth = records('original', 'validation')
        assert universe(truth) == dict(records=7079, pairs=25052581, positives=8959)
        (pairs, a, b, codes) = candidates('original', feature.ids)
        (xt, xv) = (feature.transform(train), feature.transform(val))
        mask = train.graded_relation_mask.to_numpy(bool)
        known = val.visual_relation_mask.to_numpy(bool)
        (va, vb) = [feature.ids.get_indexer(val[c]) for c in ENDS]
        recalled = np.isin(np.minimum(va, vb).astype(np.int64) * len(feature.ids) + np.maximum(va, vb), codes)[known]
        yv = val.visual_relation.eq('same').to_numpy()[known]
        (bundles, selection) = ({}, {})
        for (name, columns) in feature_columns().items():
            print('fit', name, columns, flush=True)
            model = hgb_sscd.make_models()['LR'].fit(model_input(xt[mask], columns), train.relation.to_numpy()[mask])
            raw = hgb_sscd.raw_probability(model, model_input(xv[known], columns))
            calibrator = IsotonicRegression(out_of_bounds='clip').fit(raw, yv)
            tau = hgb_sscd.choose_threshold(yv, calibrator.predict(raw), recalled)
            bundle = dict(model=model, calibrator=calibrator, threshold=tau, columns=columns)
            joblib.dump(bundle, OUT / (name + '.joblib'))
            bundles[name] = bundle
            selection[name] = dict(threshold=tau, columns=columns, training_pairs=int(mask.sum()), calibration_pairs=int(known.sum()), accepted_validation=hgb_sscd.confusion(yv, recalled & (calibrator.predict(raw) >= tau)))
        arrays = score('original', feature, pairs, bundles)
        for (name, bundle) in bundles.items():
            order = ordered_edges(bundle, arrays[name], a, b)
            rows = []
            for cap in GRID:
                labels = save_graph('original', name, cap, len(feature.ids), a, b, order)
                rows.append(dict(k=cap, pairwise=counts(labels, truth, feature.ids)))
            selection[name].update(K_dagger=visual_classifier.select_k(rows), original_validation=rows)
        feature = features('MV')
        (pairs, a, b, codes) = candidates('MV', feature.ids)
        arrays = score('MV', feature, pairs, bundles)
        truth = records('MV', 'validation')
        for (name, bundle) in bundles.items():
            order = ordered_edges(bundle, arrays[name], a, b)
            rows = []
            for cap in GRID:
                labels = save_graph('MV', name, cap, len(feature.ids), a, b, order)
                rows.append(dict(k=cap, pairwise=counts(labels, truth, feature.ids)))
            selection[name].update(MV_K_dagger=visual_classifier.select_k(rows), MV_validation=rows)
        frozen = dict(run_id=PROTOCOL['run_id'], frozen_at_utc=visual_classifier.now(), models=selection)
        dump(OUT / 'ablation_selection_frozen.json', frozen)
        dump(REPORT / 'ablation_selection_frozen.json', frozen)
        receipt.update(classifier_fits=8, calibrator_fits=8)

def ablation_test():
    with stage('ablation_test'):
        selection = read(OUT / 'ablation_selection_frozen.json')['models']
        bundles = {name: joblib.load(OUT / (name + '.joblib')) for name in selection}
        test = reference('test')
        assert hgb_sscd.denominators(test) == dict(total=10037, graded=10037, known=9494, positive=8701, negative=793, unknown=543)
        known = test.visual_relation_mask.to_numpy(bool)
        results = []
        for surface in ['original', 'strong', 'MV']:
            feature = features(surface)
            (pairs, a, b, codes) = candidates(surface, feature.ids)
            if surface == 'strong':
                arrays = score(surface, feature, pairs, bundles)
            truth = records(surface, 'test')
            if surface == 'original':
                assert universe(truth) == dict(records=5998, pairs=17985003, positives=8701)
            if surface == 'MV':
                assert universe(truth) == dict(records=8044, pairs=32348946, positives=11685)
            for (name, bundle) in bundles.items():
                cap = selection[name]['MV_K_dagger' if surface == 'MV' else 'K_dagger']
                labels = save_graph(surface, name, cap, len(feature.ids), a, b, ordered_edges(bundle, arrays[name], a, b)) if surface == 'strong' else graph(surface, name, cap)
                metric = counts(labels, truth, feature.ids)
                item = dict(surface=surface, variant=name, threshold=bundle['threshold'], K_dagger=cap, pairwise=metric, structure=visual_classifier.structure(labels), universe=universe(truth))
                if surface == 'original':
                    x = feature.transform(test)
                    raw = hgb_sscd.raw_probability(bundle['model'], model_input(x, bundle['columns']))
                    calibrated = bundle['calibrator'].predict(raw)
                    (ta, tb) = [feature.ids.get_indexer(test[c]) for c in ENDS]
                    recalled = np.isin(np.minimum(ta, tb).astype(np.int64) * len(feature.ids) + np.maximum(ta, tb), codes)
                    pred = recalled & (calibrated >= bundle['threshold'])
                    item['direct'] = hgb_sscd.confusion(test.visual_relation.eq('same').to_numpy()[known], pred[known])
                    item['unknown_selected'] = int(pred[~known].sum())
                    if name == 'classifier':
                        assert [metric[k] for k in ['tp', 'fp', 'fn']] == [8696, 106, 5]
                        assert item['structure']['U'] == 71930
                results.append(item)
        dump(REPORT / 'ablation_results.json', results)

def density_graph(n, a, b, cap, rho):
    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)
    adjacent = [dict() for _ in range(n)]
    for (x, y) in zip(a, b):
        (x, y) = (int(x), int(y))
        adjacent[x][y] = adjacent[x].get(y, 0) + 1
        adjacent[y][x] = adjacent[y].get(x, 0) + 1
    (numerator, denominator) = Fraction(str(rho)).as_integer_ratio()

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)
    (accepted, rejected) = (0, 0)
    for (x, y) in zip(a, b):
        (x, y) = (find(x), find(y))
        if x == y:
            continue
        if size[x] + size[y] > cap and adjacent[x].get(y, 0) * denominator < numerator * int(size[x]) * int(size[y]):
            rejected += 1
            continue
        if size[x] < size[y]:
            (x, y) = (y, x)
        parent[y] = x
        size[x] += size[y]
        adjacent[x].pop(y, None)
        for (z, count) in adjacent[y].items():
            if z == x:
                continue
            total = adjacent[x].get(z, 0) + count
            adjacent[x][z] = total
            adjacent[z].pop(y)
            adjacent[z][x] = total
        adjacent[y].clear()
        accepted += 1
    labels = np.array([find(i) for i in range(n)], dtype=np.int64)
    return (labels, dict(union_merges=accepted, rejected_merges=rejected))

def density_inputs(surface):
    feature = features(surface)
    (pairs, a, b, codes) = candidates(surface, feature.ids)
    if surface in ['original', 'MV']:
        arrays = {kind: np.load(OUT / f'{surface}_classifier_{kind}.npy', mmap_mode='r') for kind in ['raw', 'calibrated']}
    else:
        path = OUT / f'{surface}_classifier_raw.npy'
        if not path.exists():
            score(surface, feature, pairs, {'classifier': joblib.load(OUT / 'classifier.joblib')})
        arrays = {kind: np.load(OUT / f'{surface}_classifier_{kind}.npy', mmap_mode='r') for kind in ['raw', 'calibrated']}
    order = ordered_edges(joblib.load(OUT / 'classifier.joblib'), arrays, a, b)
    return (feature, a[order], b[order])

def density_select():
    with stage('density_selection'):
        (feature, a, b) = density_inputs('original')
        truth = records('original', 'validation')
        rows = []
        for cap in PROTOCOL['density']['K']:
            for rho in PROTOCOL['density']['rho']:
                (labels, merges) = density_graph(len(feature.ids), a, b, cap, rho)
                path = OUT / 'density/original' / f'{cap}_{rho}'
                path.mkdir(parents=True, exist_ok=True)
                np.save(path / 'components.npy', labels)
                rows.append(dict(K=cap, rho=rho, pairwise=counts(labels, truth, feature.ids), structure=visual_classifier.structure(labels), **merges))
        selected = min(rows, key=lambda r: (-Fraction(r['pairwise']['exact_f1']), -r['rho'], r['K']))
        frozen = dict(run_id=PROTOCOL['run_id'], frozen_at_utc=visual_classifier.now(), selected=selected, validation_grid=rows, test_access=False, selection_population='original non-M validation only')
        dump(OUT / 'density_selection_frozen.json', frozen)
        dump(REPORT / 'density_selection_frozen.json', frozen)

def density_test():
    with stage('density_test'):
        selected = read(OUT / 'density_selection_frozen.json')['selected']
        (rows, curve, baseline_validation) = ([], [], [])
        for surface in PROTOCOL['density']['test_surfaces']:
            (feature, a, b) = density_inputs(surface)
            truth = records(surface, 'test')
            ordinary = {}
            for (role, cap) in [('K5', 5), ('Kdagger', 64 if surface == 'expanded' else 5), ('density', selected['K'])]:
                if role == 'density':
                    (labels, _) = density_graph(len(feature.ids), a, b, cap, selected['rho'])
                else:
                    if cap not in ordinary:
                        ordinary[cap] = hgb_sscd.build_graph(len(feature.ids), a, b, cap)[0]
                    labels = ordinary[cap]
                path = OUT / 'density_test' / surface / role
                path.mkdir(parents=True, exist_ok=True)
                np.save(path / 'components.npy', labels)
                item = dict(surface=surface, role=role, K=cap, rho=selected['rho'] if role == 'density' else None, pairwise=counts(labels, truth, feature.ids), structure=visual_classifier.structure(labels), universe=universe(truth))
                rows.append(item)
                if surface == 'expanded':
                    assert universe(truth) == dict(records=11498, pairs=66096253, positives=107201)
                    if role == 'K5':
                        assert [item['pairwise'][k] for k in ['tp', 'fp', 'fn']] == [18895, 106, 88306]
                    if role == 'Kdagger':
                        assert [item['pairwise'][k] for k in ['tp', 'fp', 'fn']] == [107201, 407, 0]
                        assert item['structure']['U'] == 312229
            if surface in ['original', 'expanded']:
                val = records(surface, 'validation')
                if surface == 'expanded':
                    assert universe(val) == dict(records=12579, pairs=79109331, positives=107459)
                    for (role, expected) in [('K5', 0.300753), ('Kdagger', 0.932108)]:
                        metric = counts(np.load(OUT / 'density_test' / surface / role / 'components.npy'), val, feature.ids)
                        assert round(metric['f1'], 6) == expected, (role, metric)
                        baseline_validation.append(dict(surface=surface, role=role, pairwise=metric))
                for rho in PROTOCOL['density']['rho']:
                    (labels, _) = density_graph(len(feature.ids), a, b, selected['K'], rho)
                    curve.append(dict(surface=surface, K=selected['K'], rho=rho, pairwise=counts(labels, val, feature.ids)))
        dump(REPORT / 'density_results.json', rows)
        dump(REPORT / 'expanded_baseline_validation.json', baseline_validation)
        dump(REPORT / 'density_curves.json', curve)
        pd.json_normalize(curve).to_csv(REPORT / 'density_curves.csv', index=False)

def retrieval():
    with stage('retrieval'):
        rows = []
        for surface in ['original', 'strong', 'MV']:
            feature = features(surface)
            pairs = pd.read_parquet(candidate_path(surface))
            assert 'channels' in pairs, 'Frozen provenance missing; reconstruct channels before evaluation'
            truth = records(surface, 'test')
            test_ids = set(truth.record_id)
            test_candidates = pairs[ENDS[0]].isin(test_ids).to_numpy() & pairs[ENDS[1]].isin(test_ids).to_numpy()
            truth = truth[truth.original_level.ne('M')].set_index('record_id')
            a = truth.visual_group_id.reindex(pairs[ENDS[0]]).to_numpy()
            b = truth.visual_group_id.reindex(pairs[ENDS[1]]).to_numpy()
            positive = pd.notna(a) & pd.notna(b) & (a == b)
            channels = pairs.channels.fillna('').astype(str)
            for name in ['phash', 'clip_image', 'clip_text', 'text_exact', 'all']:
                mask = np.ones(len(pairs), dtype=bool) if name == 'all' else channels.str.split(';', regex=False).map(lambda xs: name in xs).to_numpy()
                n = int((positive & mask).sum())
                denominator = universe(truth.reset_index())['positives']
                rows.append(dict(surface=surface, channel=name, candidates=int(mask.sum()), recalled_positive=n, test_candidates=int((test_candidates & mask).sum()), non_M_test_candidates=int((pd.notna(a) & pd.notna(b) & mask).sum()), positive_denominator=denominator, recall=n / denominator, provenance='reused frozen channels column', channel_values=sorted(channels.unique().tolist())))
        dump(REPORT / 'retrieval_results.json', rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['train', 'ablation_test', 'density_select', 'density_test', 'retrieval'])
    args = parser.parse_args()
    {'train': train, 'ablation_test': ablation_test, 'density_select': density_select, 'density_test': density_test, 'retrieval': retrieval}[args.phase]()
if __name__ == '__main__':
    main()
