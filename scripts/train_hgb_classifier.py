import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
os.environ['GRAPH_CPU_THREADS'] = '4'
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from scripts.pair_features import FEATURES as REFERENCE_FEATURES, RELATIONS, ObservableFeatures, choose_threshold
from scripts.cluster_with_size_cap import build_graph, edge_order, GRID
RES = RESOURCES
RUN_ID = 'hgb_sscd'
OUT = RES / 'runs' / RUN_ID
REPORT = ROOT / str(REPORTS / 'hgb_sscd')
SOURCE = RES / 'runs/pair_features'
DATA = RES / 'datasets/benchmark'
native_baselines = RES / 'runs/native_baselines'
edit_surfaces = RES / 'runs/edit_stress'
EDITS = REPORTS / 'edit_stress/stress_manifest.csv'
FEATURES = [*REFERENCE_FEATURES, 'sscd_cosine']
MODELS = ['LR', 'HGB', 'MLP']
SURFACES = ['original', 'mixed', 'mild', 'strong']
ENDS = ['record_id_a', 'record_id_b']
EXPECTED = dict(original=1502683, mixed=1503362, mild=1504724, strong=1502221)

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)

def load_partition(partition, lineage=False):
    path = DATA / ('private/lineage.jsonl' if lineage else 'sealed/relation_applicability.jsonl')
    pattern = re.compile('"partition"\\s*:\\s*"' + partition + '"')
    rows = []
    with path.open() as stream:
        for line in stream:
            if pattern.search(line):
                row = json.loads(line)
                rows.append({k: row[k] for k in ['record_id', 'partition', 'visual_group_id']} if lineage else row)
    frame = pd.DataFrame(rows)
    assert len(frame) and frame.partition.eq(partition).all()
    if lineage:
        assert frame.record_id.is_unique and frame.visual_group_id.notna().all()
    return frame

def denominators(pairs):
    known = pairs.visual_relation_mask.to_numpy(dtype=bool)
    same = pairs.visual_relation.eq('same').to_numpy()
    return dict(total=len(pairs), graded=int(pairs.graded_relation_mask.sum()), known=int(known.sum()), positive=int((known & same).sum()), negative=int((known & ~same).sum()), unknown=int((~known).sum()))

def pairwise_counts(components, groups):
    (components, groups) = (np.asarray(components), np.asarray(groups))
    if components.ndim != 1 or groups.ndim != 1 or len(components) != len(groups):
        raise ValueError('Components and visual groups must align')
    if pd.isna(components).any() or pd.isna(groups).any():
        raise ValueError('Missing component or visual group')
    (_, c, cs) = np.unique(components, return_inverse=True, return_counts=True)
    (_, g, gs) = np.unique(groups, return_inverse=True, return_counts=True)
    (_, cells) = np.unique(np.column_stack([c, g]), axis=0, return_counts=True)
    choose2 = lambda ns: sum((int(n) * (int(n) - 1) // 2 for n in ns))
    tp = choose2(cells)
    (fp, fn) = (choose2(cs) - tp, choose2(gs) - tp)
    total = len(c) * (len(c) - 1) // 2
    return metrics(tp, fp, fn, total - tp - fp - fn, records=len(c), pairs=total)

def metrics(tp, fp, fn, tn, **extra):
    assert min(tp, fp, fn, tn) >= 0
    exact = Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0)
    return dict(n=tp + fp + fn + tn, tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn), f1=float(exact), exact_f1=str(exact), **extra)

def confusion(y, predicted):
    (y, predicted) = (np.asarray(y, dtype=bool), np.asarray(predicted, dtype=bool))
    return metrics(int((y & predicted).sum()), int((~y & predicted).sum()), int((y & ~predicted).sum()), int((~y & ~predicted).sum()))

def select_k(rows, objective):
    if len(rows) != len(GRID) or {r['k'] for r in rows} != set(GRID):
        raise ValueError('Complete frozen K grid required')
    if objective == 'pairwise':
        return min(rows, key=lambda r: (-Fraction(r['pairwise']['exact_f1']), GRID.index(r['k'])))['k']
    if objective == 'capped_graphs':
        return max(rows, key=lambda r: (Fraction(r['component']['exact_f1']), GRID.index(r['k'])))['k']
    raise ValueError(objective)

def source_path(surface):
    return SOURCE if surface == 'original' else (native_baselines if surface == 'mixed' else edit_surfaces) / surface

def load_public(surface):
    original = pd.read_parquet(SOURCE / 'public_manifest.parquet')
    assert len(original) == original.record_id.nunique() == 50000
    assert original.record_id.is_monotonic_increasing
    if surface == 'mixed':
        frame = pd.read_parquet(native_baselines / 'mixed/input_mapping.parquet')
        assert np.array_equal(frame.record_id, original.record_id)
        for column in ['row_index', 'cache_row']:
            assert np.array_equal(frame[column], np.arange(50000))
        assert frame.text.equals(original.text)
    else:
        frame = original.copy()
        if surface != 'original':
            edits = pd.read_csv(EDITS, keep_default_na=False)
            edits = edits[edits.level.eq(surface)].set_index('record_id')
            assert len(edits) == 5709 and edits.edit_status.eq('completed').all()
            frame.image_path = [edits.at[r, 'new_path'] if r in edits.index else p for (r, p) in zip(frame.record_id, frame.image_path)]
    return frame

class Features:

    def __init__(self, surface):
        self.frame = load_public(surface)
        self.ids = pd.Index(self.frame.record_id)
        source = source_path(surface)
        metadata = pd.read_parquet(SOURCE / 'features/clip/metadata.parquet')
        assert np.array_equal(metadata.record_id, self.frame.record_id)
        assert np.array_equal(metadata.row_index, np.arange(50000))
        hashes = pd.read_parquet(source / 'features/phash/phash.parquet')
        assert np.array_equal(hashes.record_id, self.frame.record_id)
        image_path = source / ('features/clip/image_emb.npy' if surface == 'original' else 'clip.npy')
        image = np.load(image_path, mmap_mode='r')
        text = np.load(SOURCE / 'features/clip/text_emb.npy', mmap_mode='r')
        assert image.shape == text.shape == (50000, 768)
        self.observable = ObservableFeatures(self.frame, image, text, hashes.phash_int)
        self.sscd = np.load(edit_surfaces / ('original' if surface == 'mixed' else surface) / 'sscd.npy', mmap_mode='r')
        assert self.sscd.shape == (50000, 512) and self.sscd.dtype == np.float32
        if surface == 'mixed':
            self.sscd = self.sscd.copy()
            selected = self.frame.image_cache_version.eq('light').to_numpy()
            assert selected.sum() == 2901
            self.sscd[selected] = np.load(edit_surfaces / 'light/sscd.npy', mmap_mode='r')[selected]
        assert np.allclose(np.linalg.norm(self.sscd, axis=1), 1.0, atol=2e-06)

    def transform(self, pairs):
        (a, b) = [self.ids.get_indexer(pairs[c]) for c in ENDS]
        assert (a >= 0).all() and (b >= 0).all()
        x = self.observable.transform(pairs)
        sscd = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), 8192):
            end = start + 8192
            sscd[start:end] = np.sum(self.sscd[a[start:end]] * self.sscd[b[start:end]], axis=1)
        result = np.column_stack([x, sscd])
        assert result.shape == (len(pairs), 12) and np.isfinite(result).all()
        return result

def candidate_input(surface, ids):
    pairs = pd.read_parquet(source_path(surface) / 'candidates.parquet', columns=ENDS)
    assert len(pairs) == EXPECTED[surface]
    (a, b) = [ids.get_indexer(pairs[c]) for c in ENDS]
    assert (a >= 0).all() and (b >= 0).all() and (a != b).all()
    codes = np.minimum(a, b).astype(np.int64) * len(ids) + np.maximum(a, b)
    assert len(np.unique(codes)) == len(codes)
    return (pairs, a, b, codes)

def make_models():
    return dict(LR=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1, random_state=23)), HGB=HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, l2_regularization=1, early_stopping=False, random_state=23), MLP=make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=160, early_stopping=False, random_state=23)))

def raw_probability(model, x):
    prob = model.predict_proba(x)
    columns = [list(model.classes_).index(c) for c in ['duplicate', 'keep_augment_candidate']]
    return prob[:, columns].sum(axis=1)
