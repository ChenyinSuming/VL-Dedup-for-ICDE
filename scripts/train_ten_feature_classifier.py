"""Fit the ten-feature logistic classifier used by the larger-entity surface."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from scripts.vl_paths import RESOURCES
from scripts.pair_features import FEATURES, RELATIONS, candidate, choose_threshold
MEMBERSHIP_FIELDS = ['record_id_a', 'record_id_b', 'partition', 'graded_relation_mask', 'visual_relation_mask']

def project_features(cache, names):
    if names != FEATURES or cache.ndim != 2 or cache.shape[1] != len(names):
        raise ValueError('Original eleven-feature cache required')
    return cache[:, :names.index('same_image_asset')]

def masks(membership):
    graded = membership.graded_relation_mask.to_numpy(dtype=bool)
    visual = membership.visual_relation_mask.to_numpy(dtype=bool)
    train = membership.partition.eq('train').to_numpy() & graded
    val = membership.partition.eq('validation').to_numpy() & graded
    test = membership.partition.eq('test').to_numpy() & graded
    return (train, val, test, visual)

def load_membership(path):
    with path.open() as handle:
        return pd.DataFrame([{key: row[key] for key in MEMBERSHIP_FIELDS} for row in map(json.loads, handle)])

def load_fit_labels(path, membership):
    y = np.full(len(membership), None, dtype=object)
    vy = np.zeros(len(membership), dtype=bool)
    with path.open() as handle:
        for (index, line) in enumerate(handle):
            row = json.loads(line)
            if row['partition'] != 'test':
                y[index] = row['relation']
                vy[index] = row['visual_relation'] == 'same'
    return (y, vy)

def fit(source: Path, input_revision: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    label_path = input_revision / 'sealed/relation_applicability.jsonl'
    membership = load_membership(label_path)
    (train, val, test, visual) = masks(membership)
    x = project_features(np.load(source / 'pair_features.npy', mmap_mode='r'), FEATURES)
    candidates = pd.read_parquet(source / 'candidates.parquet', columns=['edge_id', 'record_id_a', 'record_id_b'])
    keys = set(candidates.edge_id)
    recalled = np.array([candidate.canonical_edge_id(a, b) in keys for (a, b) in membership[['record_id_a', 'record_id_b']].itertuples(index=False, name=None)])
    source_bundle = joblib.load(source / 'logistic.joblib')
    assert source_bundle['features'] == FEATURES
    model = clone(source_bundle['model'])
    (y, vy) = load_fit_labels(label_path, membership)
    model.fit(x[train], y[train])
    assert list(model.classes_) == RELATIONS
    prob = model.predict_proba(x)
    vcolumns = [list(model.classes_).index(name) for name in ('duplicate', 'keep_augment_candidate')]
    raw = prob[:, vcolumns].sum(axis=1)
    calibrator = IsotonicRegression(out_of_bounds='clip').fit(raw[val & visual], vy[val & visual])
    scores = calibrator.predict(raw)
    threshold = choose_threshold(vy[val & visual], scores[val & visual], recalled[val & visual])
    features = FEATURES[:FEATURES.index('same_image_asset')]
    joblib.dump({'model': model, 'calibrator': calibrator, 'threshold': threshold, 'features': features}, output / 'logistic.joblib')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=RESOURCES / 'runs/pair_features')
    parser.add_argument('--input-revision', type=Path, default=RESOURCES / 'datasets/benchmark')
    parser.add_argument('--output', type=Path, default=RESOURCES / 'runs/ten_feature_classifier')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        fit(args.source, args.input_revision, args.output)
if __name__ == '__main__':
    main()
