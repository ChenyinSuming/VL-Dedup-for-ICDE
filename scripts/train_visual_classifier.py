import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, REPORTS
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['GRAPH_CPU_THREADS'] = '4'
import argparse
from contextlib import contextmanager
from fractions import Fraction
import json
from pathlib import Path
import re
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import sklearn
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from scripts import train_hgb_classifier as hgb_sscd
(RES, DATA, SOURCE) = (hgb_sscd.RES, hgb_sscd.DATA, hgb_sscd.SOURCE)
RUN_ID = 'visual_classifier'
OUT = RES / 'runs' / RUN_ID
REPORT = ROOT / str(REPORTS / 'visual_classifier')
FEATURES = [name for name in hgb_sscd.FEATURES if name != 'same_image_asset']
FEATURE_COLUMNS = [i for (i, name) in enumerate(hgb_sscd.FEATURES) if name != 'same_image_asset']
(GRID, MODELS, SURFACES, ENDS) = (hgb_sscd.GRID, ['LR', 'HGB'], hgb_sscd.SURFACES, hgb_sscd.ENDS)
(dump, read, now) = (hgb_sscd.dump, hgb_sscd.read, hgb_sscd.now)
baseline_surfaces = RES / 'runs/baselines_surfaces'

@contextmanager
def stage(name, gpu=False):
    (started, cpu) = (time.perf_counter(), time.process_time())
    item = {}
    try:
        yield item
    finally:
        item.update(wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu)
        dump(OUT / (name + '_stage.json'), item)

def load_partition(partition, lineage=False):
    path = DATA / ('private/lineage.jsonl' if lineage else 'sealed/relation_applicability.jsonl')
    pattern = re.compile('"partition"\\s*:\\s*"' + partition + '"')
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if pattern.search(line)]
    frame = pd.DataFrame(rows)
    assert len(frame) and frame.partition.eq(partition).all()
    if lineage:
        assert frame.record_id.is_unique and frame.visual_group_id.notna().all()
    return frame

def pairwise_counts(components, lineage, ids, exclude_m=True):
    truth = lineage[lineage.original_level.ne('M')] if exclude_m else lineage
    positions = ids.get_indexer(truth.record_id)
    assert (positions >= 0).all()
    return hgb_sscd.pairwise_counts(np.asarray(components)[positions], truth.visual_group_id)

def select_k(rows):
    assert len(rows) == len(GRID) and {r['k'] for r in rows} == set(GRID)
    return min(rows, key=lambda r: (-Fraction(r['pairwise']['exact_f1']), GRID.index(r['k'])))['k']

class Features:

    def __init__(self, surface='original'):
        self.base = hgb_sscd.Features(surface)
        self.ids = self.base.ids

    def transform(self, pairs):
        result = self.base.transform(pairs)[:, FEATURE_COLUMNS]
        assert result.shape == (len(pairs), 11) and np.isfinite(result).all()
        return result

def build_cell(surface, name, cap, a, b, ordered):
    (labels, retained, rejected, counts) = hgb_sscd.build_graph(50000, a[ordered], b[ordered], cap)
    path = OUT / 'graphs' / surface / name / str(cap)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'components.npy', labels)
    np.save(path / 'retained_candidate_indices.npy', ordered[retained])
    sizes = np.unique(labels, return_counts=True)[1]
    structure = dict(global_components=len(sizes), largest_component=int(sizes.max()), U=sum((int(n) * (int(n) - 1) // 2 for n in sizes)), **counts)
    assert structure['global_components'] == 50000 - counts['union_merges']
    cell = dict(surface=surface, name=name, k=cap, structure=structure, artifacts=str(path))
    dump(path / 'cell.json', cell)
    return cell

def score_and_build(surface, bundles, caps):
    feature = Features(surface)
    (candidates, a, b, codes) = hgb_sscd.candidate_input(surface, feature.ids)
    arrays = {name: {kind: np.lib.format.open_memmap(OUT / f'{surface}_{name}_{kind}.npy', mode='w+', dtype=np.float64, shape=(len(candidates),)) for kind in ['raw', 'calibrated']} for name in bundles}
    for start in range(0, len(candidates), 32768):
        x = feature.transform(candidates.iloc[start:start + 32768])
        for (name, bundle) in bundles.items():
            raw = hgb_sscd.raw_probability(bundle['model'], x)
            arrays[name]['raw'][start:start + len(x)] = raw
            arrays[name]['calibrated'][start:start + len(x)] = bundle['calibrator'].predict(raw)
        if start % (32768 * 10) == 0:
            print('score', surface, start, '/', len(candidates), flush=True)
    cells = []
    for (name, bundle) in bundles.items():
        for array in arrays[name].values():
            array.flush()
        (raw, calibrated) = (arrays[name]['raw'], arrays[name]['calibrated'])
        selected = np.flatnonzero(calibrated >= bundle['threshold'])
        ordered = hgb_sscd.edge_order(selected, calibrated, a, b, raw[selected])
        for cap in caps[name]:
            cells.append(build_cell(surface, name, cap, a, b, ordered))
    return cells

def train():
    with stage('train_select') as receipt:
        protocol = read(REPORT / 'settings.json')
        assert protocol['features'] == FEATURES and protocol['k_grid'] == GRID
        (train_pairs, val) = (load_partition('train'), load_partition('validation'))
        assert hgb_sscd.denominators(train_pairs)['total'] == 46425
        assert hgb_sscd.denominators(val)['known'] == 9766
        base = Features()
        (xtrain, xval) = (base.transform(train_pairs), base.transform(val))
        known = val.visual_relation_mask.to_numpy(dtype=bool)
        yval = val.visual_relation.eq('same').to_numpy()[known]
        (_, _, _, codes) = hgb_sscd.candidate_input('original', base.ids)
        (va, vb) = [base.ids.get_indexer(val[c]) for c in ENDS]
        recalled = np.isin(np.minimum(va, vb).astype(np.int64) * 50000 + np.maximum(va, vb), codes)[known]
        mask = train_pairs.graded_relation_mask.to_numpy(dtype=bool)
        (x, y) = (xtrain[mask], train_pairs.relation.to_numpy()[mask])
        (bundles, selections) = ({}, {})
        models = hgb_sscd.make_models()
        for classifier in MODELS:
            model = models[classifier]
            name = 'classifier_' + classifier
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(x, y)
            raw = hgb_sscd.raw_probability(model, xval[known])
            calibrator = IsotonicRegression(out_of_bounds='clip').fit(raw, yval)
            score = calibrator.predict(raw)
            threshold = hgb_sscd.choose_threshold(yval, score, recalled)
            bundle = dict(model=model, calibrator=calibrator, threshold=threshold, features=FEATURES, run_id=RUN_ID)
            joblib.dump(bundle, OUT / (name + '.joblib'))
            bundles[name] = bundle
            selections[name] = dict(threshold=threshold, fitted_training_pairs=len(x), calibration_pairs=len(yval), accepted_validation=hgb_sscd.confusion(yval, recalled & (score >= threshold)), warnings=[str(w.message) for w in caught], effective_parameters={k: v for (k, v) in model.get_params().items() if isinstance(v, (str, int, float, bool, tuple, type(None)))})
        cells = score_and_build('original', bundles, {name: GRID for name in bundles})
        dump(OUT / 'original_graphs_frozen.json', dict(cells=cells, frozen_at_utc=now()))
        truth = load_partition('validation', True)
        rows = []
        for cell in cells:
            labels = np.load(Path(cell['artifacts']) / 'components.npy')
            rows.append(dict(name=cell['name'], k=cell['k'], pairwise=pairwise_counts(labels, truth, base.ids)))
        for name in bundles:
            selections[name]['K_dagger'] = select_k([r for r in rows if r['name'] == name])
        frozen = dict(run_id=RUN_ID, frozen_at_utc=now(), models=selections, validation=rows, sklearn_version=sklearn.__version__)
        dump(OUT / 'selection_frozen.json', frozen)
        dump(REPORT / 'selection_frozen.json', frozen)
        receipt.update(classifier_fits=2, calibrator_fits=2)

def surfaces():
    selection = read(OUT / 'selection_frozen.json')
    bundles = {'classifier_' + model: joblib.load(OUT / ('classifier_' + model + '.joblib')) for model in MODELS}
    cells = read(OUT / 'original_graphs_frozen.json')['cells']
    for surface in SURFACES[1:]:
        with stage('surface_' + surface):
            caps = {name: sorted(set(['inf', 8, 5, selection['models'][name]['K_dagger']]), key=GRID.index) for name in bundles}
            cells.extend(score_and_build(surface, bundles, caps))
    dump(OUT / 'graphs_frozen.json', dict(cells=cells, frozen_at_utc=now()))

def structure(labels):
    sizes = np.unique(labels, return_counts=True)[1]
    return dict(global_components=len(sizes), largest_component=int(sizes.max()), U=sum((int(n) * (int(n) - 1) // 2 for n in sizes)))

def evaluate():
    with stage('test_evaluate'):
        selection = read(OUT / 'selection_frozen.json')
        frozen = read(OUT / 'graphs_frozen.json')
        (pairs, truth) = (load_partition('test'), load_partition('test', True))
        assert hgb_sscd.denominators(pairs) == dict(total=10037, graded=10037, known=9494, positive=8701, negative=793, unknown=543)
        (rows, analysiss) = ([], {})
        for surface in SURFACES:
            ids = pd.Index(hgb_sscd.load_public(surface).record_id)
            (candidates, a, b, codes) = hgb_sscd.candidate_input(surface, ids)
            variant = 'classifier'
            for classifier in MODELS:
                name = variant + '_' + classifier
                roles = dict(inf='inf', K_dagger=selection['models'][name]['K_dagger'], K8=8, K5=5)
                for (role, cap) in roles.items():
                    cell = next((c for c in frozen['cells'] if (c['surface'], c['name'], c['k']) == (surface, name, cap)))
                    labels = np.load(Path(cell['artifacts']) / 'components.npy')
                    retained = np.load(Path(cell['artifacts']) / 'retained_candidate_indices.npy')
                    (pa_, pb) = [ids.get_indexer(pairs[c]) for c in ENDS]
                    known = pairs.visual_relation_mask.to_numpy(dtype=bool)
                    y = pairs.visual_relation.eq('same').to_numpy()[known]
                    paircodes = np.minimum(pa_, pb).astype(np.int64) * 50000 + np.maximum(pa_, pb)
                    direct = np.isin(paircodes, codes[retained])
                    co = labels[pa_] == labels[pb]
                    row = dict(surface=surface, name=name, variant=variant, classifier=classifier, role=role, k=cap, pairwise=pairwise_counts(labels, truth, ids), direct=hgb_sscd.confusion(y, direct[known]), component=hgb_sscd.confusion(y, co[known]), M_connected=int(co[~known].sum()), M_n=int((~known).sum()), structure=cell['structure'])
                    assert row['direct']['n'] == row['component']['n'] == 9494 and row['M_n'] == 543
                    rows.append(row)
            if surface == 'strong':
                feature = Features('strong')
                positives = pairs[pairs.visual_relation_mask & pairs.visual_relation.eq('same')]
                x = feature.transform(positives)
                quantiles = read(REPORT / 'settings.json')['s4_analysiss']['quantiles']
                variant = 'classifier'
                for classifier in MODELS:
                    name = variant + '_' + classifier
                    bundle = joblib.load(OUT / (name + '.joblib'))
                    raw = hgb_sscd.raw_probability(bundle['model'], x)
                    (scores, t) = (bundle['calibrator'].predict(raw), bundle['threshold'])
                    analysiss[name] = dict(n=len(scores), threshold=t, quantiles=quantiles, calibrated_quantiles=np.quantile(scores, quantiles).tolist(), raw_quantiles=np.quantile(raw, quantiles).tolist(), rejected=int((scores < t).sum()), rejected_within_005=int(((scores < t) & (scores >= t - 0.05)).sum()))
        result = dict(run_id=RUN_ID, rows=rows, s4_analysiss=analysiss, selection=selection['models'], freeze=dict(selection=selection['frozen_at_utc'], graphs=frozen['frozen_at_utc']))
        dump(OUT / 'results.json', result)
        dump(REPORT / 'results.json', result)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['train', 'surfaces', 'evaluate'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'train': train, 'surfaces': surfaces, 'evaluate': evaluate}[args.phase]()
if __name__ == '__main__':
    main()
