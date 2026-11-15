"""Run Stages C and D of VL-Dedup on the original surface from the released features and models.

Usage:
    python scripts/recompute_stage_cd.py

Inputs:
    artifacts/features/original_candidates.npz   the 1,502,683 Stage B candidate pairs of the
                                                 50,000 records and their 11 Stage C features
    artifacts/models/vl_dedup_lr.joblib           frozen classifier, calibrator and threshold
    artifacts/models/hgb.joblib                   the HGB variant of the graph table

For each model the script scores every candidate, calibrates the score, keeps the pairs at or
above the threshold, adds them in a fixed order under the size cap (K = 5 and no cap), and checks
that the clusters equal the released labels in artifacts/clusters/original.parquet. It
then prints pairwise F1 on the test records. The models need the scikit-learn version listed in
environment/requirements.txt.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
MODELS = {'vl_dedup': 'vl_dedup_lr.joblib', 'hgb': 'hgb.joblib'}
POSITIVE = ['duplicate', 'keep_augment_candidate']


def raw_probability(model, x):
    """Sum of the class probabilities of the two same-entity relations."""
    prob = model.predict_proba(x)
    columns = [list(model.classes_).index(c) for c in POSITIVE]
    return prob[:, columns].sum(axis=1)


def edge_order(selected, calibrated, a, b, raw):
    """Calibrated score descending, then raw score descending, then record indices ascending."""
    keys = (np.maximum(a[selected], b[selected]), np.minimum(a[selected], b[selected]), -raw)
    return selected[np.lexsort((*keys, -calibrated[selected]))]


def build_graph(n, a, b, cap):
    """Union-find over the ordered pairs; a merge is rejected when the cluster would exceed cap."""
    parent, sizes = list(range(n)), [1] * n
    limit = n if cap == 'inf' else cap

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in zip(a, b):
        x, y = find(int(u)), find(int(v))
        if x == y or sizes[x] + sizes[y] > limit:
            continue
        if sizes[x] < sizes[y] or (sizes[x] == sizes[y] and x > y):
            x, y = y, x
        parent[y] = x
        sizes[x] += sizes[y]
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def same_partition(x, y):
    """True when two label arrays group the records in the same way."""
    return len(np.unique(x)) == len(np.unique(y)) == len(np.unique(np.column_stack([x, y]), axis=0))


def pairwise_f1(labels, groups):
    def choose2(counts):
        return sum(int(c) * (int(c) - 1) // 2 for c in counts)
    _, c, cs = np.unique(labels, return_inverse=True, return_counts=True)
    _, g, gs = np.unique(groups, return_inverse=True, return_counts=True)
    _, cells = np.unique(np.column_stack([c, g]), axis=0, return_counts=True)
    tp = choose2(cells)
    fp, fn = choose2(cs) - tp, choose2(gs) - tp
    return 2 * tp / (2 * tp + fp + fn)


def main():
    data = np.load(ART / 'features' / 'original_candidates.npz')
    ids, a, b, x = data['record_id'], data['a'], data['b'], data['features']
    released = pd.read_parquet(ART / 'clusters' / 'original.parquet').set_index('record_id').loc[ids]
    records = pd.read_csv(ROOT / 'benchmark' / 'records.csv.gz').set_index('record_id').loc[ids]
    test = (records.partition.eq('test') & records.level.ne('M')).to_numpy()
    groups = records.visual_group_id.to_numpy()
    print(f'{len(ids)} records, {len(a)} candidate pairs, {x.shape[1]} features')
    ok = True
    for name, file in MODELS.items():
        bundle = joblib.load(ART / 'models' / file)
        raw = raw_probability(bundle['model'], x)
        calibrated = bundle['calibrator'].predict(raw)
        selected = np.flatnonzero(calibrated >= bundle['threshold'])
        ordered = edge_order(selected, calibrated, a, b, raw[selected])
        print(f'{name}: threshold {bundle["threshold"]:.4f}, {len(selected)} selected pairs')
        for cap in [5, 'inf']:
            labels = build_graph(len(ids), a[ordered], b[ordered], cap)
            column = f'{name}_k{cap}'
            match = same_partition(labels, released[column].to_numpy())
            ok &= match
            print(f'  K={cap}: clusters equal released labels: {match}; '
                  f'test pairwise F1 {pairwise_f1(labels[test], groups[test]):.4f}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
