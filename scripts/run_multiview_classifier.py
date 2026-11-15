import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['GRAPH_CPU_THREADS'] = '4'
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
from threadpoolctl import threadpool_limits
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from scripts import run_sop_multiview as multiview_base
from scripts import train_visual_classifier as visual_classifier
from scripts import train_hgb_classifier as hgb_sscd
CACHE = multiview_base.OUT
OUT = CACHE / 'visual_classifier_classifier'
REPORT = multiview_base.REPORT / 'visual_classifier_classifier'
classifier_run = visual_classifier.OUT
FEATURES = visual_classifier.FEATURES
(GRID, ENDS) = (visual_classifier.GRID, multiview_base.ENDS)
SCORERS = ['visual_classifier_classifier_LR', 'visual_classifier_classifier_HGB']
(read, dump, now) = (multiview_base.read, multiview_base.dump, multiview_base.now)

def require_freezes():
    return (read(OUT / 'selection_frozen.json'), read(OUT / 'graphs_frozen.json'))

def reference(partition, lineage=False):
    return multiview_base.read_partition(multiview_base.DATA / ('private/lineage.jsonl' if lineage else 'sealed/relation_applicability.jsonl'), partition)

def combine_records(base, new, partition):
    columns = ['record_id', 'partition', 'visual_group_id', 'original_level', 'source_dataset']
    added = new[new.partition.eq(partition)].copy().assign(original_level='SOP_multiview', source_dataset='stanford_online_products')
    result = pd.concat([base[columns], added[columns]], ignore_index=True).sort_values('record_id')
    assert result.record_id.is_unique and result.partition.eq(partition).all()
    return result.reset_index(drop=True)

def records(partition):
    return combine_records(reference(partition, True), pd.read_parquet(CACHE / 'new_manifest.parquet'), partition)

class Features:

    def __init__(self):
        self.base = multiview_base.Features()
        (self.ids, self.frame) = (self.base.ids, self.base.frame)
        assert len(self.ids) == 58550 and self.ids.is_monotonic_increasing

    def transform(self, pairs):
        result = self.base.transform(pairs)[:, visual_classifier.FEATURE_COLUMNS]
        assert result.shape == (len(pairs), 11) and np.isfinite(result).all()
        return result

def coverage(pairs, candidate_codes, ids):
    added = pairs[pairs.origin.eq('multiview_SOP')]
    called = np.isin(multiview_base.codes_for(added, ids), candidate_codes)
    positive = added.visual_relation.eq('same').to_numpy()
    return dict(positive=[int((called & positive).sum()), int(positive.sum())], negative=[int((called & ~positive).sum()), int((~positive).sum())])

def build_cell(name, cap, a, b, ordered, n):
    (labels, retained, rejected, counts) = hgb_sscd.build_graph(n, a[ordered], b[ordered], cap)
    path = OUT / 'graphs' / name / str(cap)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'components.npy', labels)
    np.save(path / 'retained_candidate_indices.npy', ordered[retained])
    np.save(path / 'rejections.npy', rejected)
    sizes = np.unique(labels, return_counts=True)[1]
    structure = dict(global_components=len(sizes), largest_component=int(sizes.max()), U=sum((int(s) * (int(s) - 1) // 2 for s in sizes)), **counts)
    assert n - structure['union_merges'] == structure['global_components']
    cell = dict(name=name, k=cap, structure=structure, artifacts=str(path), source='new_visual_classifier_scoring')
    dump(path / 'cell.json', cell)
    return cell

def score_and_build(feature, candidates, bundles, selections):
    n = len(candidates)
    arrays = {name: {kind: np.lib.format.open_memmap(OUT / f'{name}_{kind}.npy', mode='w+', dtype=np.float64, shape=(n,)) for kind in ['raw', 'score']} for name in bundles}
    calls = {name: 0 for name in bundles}
    for start in range(0, n, 32768):
        batch = candidates.iloc[start:start + 32768]
        x = feature.transform(batch)
        for (name, bundle) in bundles.items():
            raw = hgb_sscd.raw_probability(bundle['model'], x)
            arrays[name]['raw'][start:start + len(batch)] = raw
            arrays[name]['score'][start:start + len(batch)] = bundle['calibrator'].predict(raw)
            calls[name] += 1
        if start % (32768 * 10) == 0:
            print('score', start, '/', n, flush=True)
    (a, b) = feature.base.indices(candidates)
    cells = []
    for name in SCORERS:
        for vector in arrays[name].values():
            vector.flush()
            assert np.isfinite(vector).all()
        (raw, score) = (arrays[name]['raw'], arrays[name]['score'])
        positive = np.flatnonzero(score >= selections[name]['threshold'])
        ordered = hgb_sscd.edge_order(positive, score, a, b, raw[positive])
        np.save(OUT / (name + '_ordered.npy'), ordered)
        for cap in GRID:
            cells.append(build_cell(name, cap, a, b, ordered, len(feature.ids)))
    dump(OUT / 'scoring_frozen.json', dict(candidates=n, features=FEATURES, new_predict_calls=calls, new_model_rows={name: n for name in bundles}, new_graphs=len(SCORERS) * len(GRID), frozen_at_utc=now()))
    return cells

def train_select():
    with threadpool_limits(limits=4):
        OUT.mkdir(parents=True, exist_ok=True)
        feature = Features()
        candidate = pd.read_parquet(CACHE / 'candidates.parquet', columns=ENDS)
        (bundles, selections) = ({}, {})
        for name in SCORERS:
            source_name = name.removeprefix('visual_classifier_')
            path = classifier_run / (source_name + '.joblib')
            bundle = joblib.load(path)
            bundles[name] = bundle
            selections[name] = dict(threshold=float(bundle['threshold']), feature_count=11, source=str(path))
        cells = score_and_build(feature, candidate, bundles, selections)
        val_records = records('validation')
        validation = []
        for cell in cells:
            components = np.load(Path(cell['artifacts']) / 'components.npy')
            primary = visual_classifier.pairwise_counts(components, val_records, feature.ids)
            full = visual_classifier.pairwise_counts(components, val_records, feature.ids, exclude_m=False)
            sop = val_records[val_records.source_dataset.eq('stanford_online_products')]
            validation.append(dict(name=cell['name'], k=cell['k'], pairwise=primary, pairwise_all=full, pairwise_sop=visual_classifier.pairwise_counts(components, sop, feature.ids)))
        assert len(validation) == len(SCORERS) * len(GRID)
        for name in SCORERS:
            rows = [r for r in validation if r['name'] == name]
            selections[name]['K_dagger'] = visual_classifier.select_k(rows)
        frozen = dict(run_id=multiview_base.RUN_ID, scorers=selections, validation=validation, k_grid=GRID, feature_names=FEATURES, primary='validation all-record pairwise excluding original_level=M; global graph unchanged', frozen_at_utc=now())
        dump(OUT / 'selection_frozen.json', frozen)
        dump(REPORT / 'selection_frozen.json', frozen)
        requested = [c for c in cells if c['k'] in ['inf', 8, 5, selections[c['name']]['K_dagger']]]
        dump(OUT / 'validation_graphs.json', dict(cells=cells, frozen_at_utc=now()))
        dump(OUT / 'graphs_frozen.json', dict(cells=requested, frozen_at_utc=now()))

def labeled_metrics(pairs, ids, components, retained_codes):
    known = pairs.visual_relation_mask.fillna(False).to_numpy(dtype=bool)
    y = pairs.visual_relation.eq('same').to_numpy()
    (a, b) = [ids.get_indexer(pairs[c]) for c in ENDS]
    assert (a >= 0).all() and (b >= 0).all()
    direct = np.isin(multiview_base.codes_for(pairs, ids), retained_codes)
    co = components[a] == components[b]
    return dict(direct=hgb_sscd.confusion(y[known], direct[known]), component=hgb_sscd.confusion(y[known], co[known]), unknown=dict(n=int((~known).sum()), selected=int(direct[~known].sum()), connected=int(co[~known].sum())))

def evaluate():
    (selection, graph_manifest) = require_freezes()
    with threadpool_limits(limits=4):
        OUT.mkdir(parents=True, exist_ok=True)
        public = pd.read_parquet(CACHE / 'public_manifest.parquet')
        ids = pd.Index(public.record_id)
        candidate = pd.read_parquet(CACHE / 'candidates.parquet', columns=ENDS)
        codes = multiview_base.codes_for(candidate, ids)
        truth = records('test')
        original = reference('test')
        labeled = pd.read_parquet(CACHE / 'test_pairs.parquet')
        sop_labeled = labeled[labeled.origin.eq('multiview_SOP')]
        sop_records = truth[truth.source_dataset.eq('stanford_online_products')]
        values = {}
        for cell in graph_manifest['cells']:
            components = np.load(Path(cell['artifacts']) / 'components.npy')
            retained = np.load(Path(cell['artifacts']) / 'retained_candidate_indices.npy')
            primary = visual_classifier.pairwise_counts(components, truth, ids)
            full = visual_classifier.pairwise_counts(components, truth, ids, exclude_m=False)
            sop = visual_classifier.pairwise_counts(components, sop_records, ids)
            assert primary['records'] == 8044 and primary['tp'] + primary['fn'] == 11685
            assert sop['records'] == 3154 and sop['tp'] + sop['fn'] == 2984
            values[cell['name'], cell['k']] = dict(primary_pairwise=primary, pairwise_all=full, pairwise_sop=sop, labeled=labeled_metrics(labeled, ids, components, codes[retained]), sop_labeled=labeled_metrics(sop_labeled, ids, components, codes[retained]), original_known=labeled_metrics(original, ids, components, codes[retained]), structure=cell['structure'], graph_artifacts=cell['artifacts'], graph_source=cell['source'])
        rows = []
        for name in SCORERS:
            for (role, cap) in [('inf', 'inf'), ('K_dagger', selection['scorers'][name]['K_dagger']), ('K8', 8), ('K5', 5)]:
                rows.append(dict(name=name, role=role, k=cap, **values[name, cap]))
        result = dict(run_id=multiview_base.RUN_ID, scope='CPU-only cached-surface scoring and global graph evaluation', records=58550, candidates=1771452, features=FEATURES, selections=selection['scorers'], rows=rows, primary_definition='All test unordered pairs with both original_level !=M; M remains in global graph', labeled_test=dict(total=6127, positive=2984, negative=3143, SOP_only=5968, original_HN=159), original_known=dict(total=9494, positive=8701, negative=793, unknown_separate=543), test_evaluated_at_utc=now())
        dump(OUT / 'results.json', result)
    dump(REPORT / 'results.json', read(OUT / 'results.json'))

def main():
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    parser = argparse.ArgumentParser(description=__doc__)
    phases = {'train-select': train_select, 'evaluate': evaluate}
    parser.add_argument('--phase', required=True, choices=list(phases))
    args = parser.parse_args()
    phases[args.phase]()
if __name__ == '__main__':
    main()
