import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import re
import sys
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(4)
pa.set_io_thread_count(4)
from sklearn.linear_model import LogisticRegression
import joblib
from scripts.baseline_matrix_text import normalize_baseline_text, _word_shingles
from scripts.cluster_with_size_cap import build_graph
RES = RESOURCES
RUN_ID = 'baselines_surfaces'
OUT = RES / 'runs' / RUN_ID
REPORT = ROOT / str(REPORTS / 'baselines_surfaces')
SOURCE = RES / 'runs/pair_features'
edit_surfaces = RES / 'runs/edit_stress'
edit_surfaces_REPORT = REPORTS / 'edit_stress'
native_baselines = RES / 'runs/native_baselines'
SEALED = RES / 'datasets/benchmark/sealed/relation_applicability.jsonl'
METHODS = ['B0', 'B1a', 'B1b', 'B2', 'B5a', 'B3', 'B4', 'B5b', 'B5c', 'B6']
SURFACES = ['original', 'mixed', 'mild', 'strong']
GRID = [2, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024, 'inf']
ENDS = ['record_id_a', 'record_id_b']
SIMCSE_REV = '2d82fab19ac3a73a20dd20333d27eb8a52d6e97f'
SIMCSE_ID = 'princeton-nlp/sup-simcse-bert-base-uncased'

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def manifest():
    frame = pd.read_parquet(SOURCE / 'public_manifest.parquet')
    assert len(frame) == frame.record_id.nunique() == 50000
    return frame

def mapping(surface):
    frame = manifest()
    if surface == 'mixed':
        mapped = pd.read_parquet(native_baselines / 'mixed/input_mapping.parquet')
        assert np.array_equal(mapped.record_id, frame.record_id)
        assert np.array_equal(mapped.text, frame.text)
        assert np.array_equal(mapped.row_index, np.arange(50000))
        assert np.array_equal(mapped.cache_row, np.arange(50000))
        return mapped
    if surface != 'original':
        edits = pd.read_csv(edit_surfaces_REPORT / 'stress_manifest.csv', keep_default_na=False)
        rows = edits[edits.level.eq(surface)].set_index('record_id')
        assert len(rows) == 5709 and rows.edit_status.eq('completed').all()
        frame.image_path = [rows.at[r, 'new_path'] if r in rows.index else p for (r, p) in zip(frame.record_id, frame.image_path)]
    return frame

def pdq_one(path):
    import pdqhash
    from PIL import Image
    with Image.open(path) as image:
        (bits, quality) = pdqhash.compute(np.asarray(image.convert('RGB')))
    return (''.join((str(int(x)) for x in bits)), int(quality))

def prepare_pdq():
    assert importlib.metadata.version('pdqhash') == '0.2.8'
    frames = {s: mapping(s) for s in SURFACES}
    paths = list(dict.fromkeys((p for f in frames.values() for p in f.image_path)))
    with ProcessPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(pdq_one, paths, chunksize=32))
    by_path = dict(zip(paths, values))
    for (surface, frame) in frames.items():
        rows = [by_path[p] for p in frame.image_path]
        pd.DataFrame(dict(record_id=frame.record_id, bits=[r[0] for r in rows], quality=[r[1] for r in rows])).to_parquet(OUT / f'{surface}_pdq.parquet', index=False)

def prepare_simcse():
    import torch
    from transformers import AutoTokenizer, AutoModel
    model_dir = OUT / 'models/simcse'
    assert (model_dir / 'config.json').exists(), 'Download pinned model assets first'
    torch.set_num_threads(4)
    torch.manual_seed(20_260_713)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to('cuda').eval()
    frame = manifest()
    texts = frame.text.astype(str).tolist()
    vectors = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(texts), 128):
            inputs = tok(texts[start:start + 128], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
            output = model(**inputs).pooler_output
            vectors[start:start + len(output)] = torch.nn.functional.normalize(output.float(), dim=1).cpu().numpy()
            if start % 12800 == 0:
                print('simcse', start, '/', len(texts), flush=True)
    np.save(OUT / 'simcse.npy', vectors)
    frame[['record_id']].to_parquet(OUT / 'simcse_metadata.parquet', index=False)

def load_partition(partition):
    pattern = re.compile('"partition"\\s*:\\s*"' + partition + '"')
    rows = []
    with SEALED.open() as stream:
        for line in stream:
            if pattern.search(line):
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    assert frame.partition.eq(partition).all() and (not frame.pair_id.duplicated().any())
    return frame

def metric(y, pred):
    (y, pred) = (np.asarray(y, dtype=bool), np.asarray(pred, dtype=bool))
    (tp, fp) = (int((y & pred).sum()), int((~y & pred).sum()))
    (fn, tn) = (int((y & ~pred).sum()), int((~y & ~pred).sum()))
    return dict(n=len(y), tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn) if tp + fn else 0.0, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)

def exact_f1(row):
    return Fraction(2 * row['tp'], 2 * row['tp'] + row['fp'] + row['fn'])

def select_threshold(scores, y, recalled, direction='higher'):
    (scores, y) = (np.asarray(scores, dtype=np.float64), np.asarray(y, dtype=bool))
    utility = scores if direction == 'higher' else -scores
    valid = np.isfinite(utility)
    unique = np.unique(utility[valid])[::-1]
    assert len(unique)
    order = np.argsort(-utility[valid], kind='stable')
    (values, truth) = (utility[valid][order], y[valid][order])
    active = np.asarray(recalled, dtype=bool)[valid][order]
    ends = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values) - 1]
    tp = np.cumsum(truth & active)[ends]
    fp = np.cumsum(~truth & active)[ends]
    positive = int(y.sum())
    fs = [Fraction(2 * int(t), int(t + f + positive)) for (t, f) in zip(tp, fp)]
    best = max(fs)
    indices = np.array([i for (i, f) in enumerate(fs) if f == best])
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)

    def group_key(g):
        (t, f) = (int(tp[g[0]]), int(fp[g[0]]))
        return (Fraction(t, t + f) if t + f else Fraction(0), float(unique[g[0]]))
    chosen = max(groups, key=group_key)
    (lo, hi) = (float(unique[chosen[-1]]), float(unique[chosen[0]]))
    threshold = (lo + hi) / 2
    if direction == 'lower':
        threshold = -threshold
    pred = threshold_mask(scores, threshold, direction) & recalled
    assert exact_f1(metric(y, pred)) == best
    return dict(threshold=threshold, direction=direction, validation=metric(y, pred), best_f1_exact=str(best), plateau_score_endpoints=sorted([lo, hi] if direction == 'higher' else [-lo, -hi]), maximizing_intervals=len(groups), unique_validation_scores=len(unique))

def threshold_mask(scores, threshold, direction):
    return np.isfinite(scores) & (scores >= threshold if direction == 'higher' else scores <= threshold)

class Scorer:

    def __init__(self, frame):
        self.frame = frame
        self.ids = pd.Index(frame.record_id)
        self.text = np.load(SOURCE / 'features/clip/text_emb.npy', mmap_mode='r')
        meta = pd.read_parquet(SOURCE / 'features/clip/metadata.parquet')
        assert np.array_equal(meta.record_id, frame.record_id)
        assert np.array_equal(meta.row_index, np.arange(50000))
        sim_meta = pd.read_parquet(OUT / 'simcse_metadata.parquet')
        assert np.array_equal(sim_meta.record_id, frame.record_id)
        self.simcse = np.load(OUT / 'simcse.npy', mmap_mode='r')
        self.shingles = [_word_shingles(t) for t in frame.text]
        self.normalized = np.array([normalize_baseline_text(t) for t in frame.text])
        self.models = {}
        self.surface = None

    def set_surface(self, surface):
        self.surface = surface
        source = SOURCE if surface == 'original' else (native_baselines if surface == 'mixed' else edit_surfaces) / surface
        self.clip = np.load(source / ('features/clip/image_emb.npy' if surface == 'original' else 'clip.npy'), mmap_mode='r')
        self.dino = np.load(edit_surfaces / ('original' if surface == 'mixed' else surface) / 'dinov2.npy', mmap_mode='r')
        self.sscd = np.load(edit_surfaces / ('original' if surface == 'mixed' else surface) / 'sscd.npy', mmap_mode='r')
        if surface == 'mixed':
            mapped = mapping('mixed')
            edited = mapped.image_cache_version.eq('light').to_numpy()
            assert edited.sum() == 2901
            self.dino = self.dino.copy()
            self.sscd = self.sscd.copy()
            for (attr, name) in [('dino', 'dinov2'), ('sscd', 'sscd')]:
                getattr(self, attr)[edited] = np.load(edit_surfaces / 'light' / (name + '.npy'), mmap_mode='r')[edited]
        hashes = pd.read_parquet(source / 'features/phash/phash.parquet')
        assert np.array_equal(hashes.record_id, self.frame.record_id)
        self.phash = hashes.phash_int.to_numpy(dtype=np.uint64)
        pdq = pd.read_parquet(OUT / f'{surface}_pdq.parquet')
        assert np.array_equal(pdq.record_id, self.frame.record_id)
        self.pdq = [int(x, 2) for x in pdq.bits]
        self.quality = pdq.quality.to_numpy()
        for vectors in (self.clip, self.dino, self.sscd, self.text, self.simcse):
            assert vectors.shape[0] == 50000 and np.isfinite(vectors).all()

    @staticmethod
    def cosine(vectors, a, b):
        result = np.empty(len(a), dtype=np.float32)
        for start in range(0, len(a), 8192):
            stop = start + 8192
            result[start:stop] = np.sum(vectors[a[start:stop]] * vectors[b[start:stop]], axis=1, dtype=np.float32)
        return result

    def score(self, method, a, b):
        vectors = {'B0': self.dino, 'B2': self.sscd, 'B5a': self.clip, 'B4': self.simcse, 'B5b': self.text}
        if method in vectors:
            return self.cosine(vectors[method], a, b)
        if method == 'B1a':
            return np.fromiter((int(x).bit_count() for x in self.phash[a] ^ self.phash[b]), dtype=np.float64, count=len(a))
        if method == 'B1b':
            scores = np.fromiter(((self.pdq[u] ^ self.pdq[v]).bit_count() for (u, v) in zip(a, b)), dtype=np.float64, count=len(a))
            scores[np.minimum(self.quality[a], self.quality[b]) < 50] = np.nan
            return scores
        if method == 'B3':

            def jaccard(u, v):
                (x, y) = (self.shingles[u], self.shingles[v])
                union = len(x | y)
                return len(x & y) / union if union else 0.0
            return np.fromiter((jaccard(u, v) for (u, v) in zip(a, b)), dtype=np.float32, count=len(a))
        features = ['B5a', 'B5b'] if method == 'B5c' else ['B2', 'B4']
        matrix = np.column_stack([self.score(m, a, b) for m in features])
        if method == 'B6':
            matrix = np.column_stack([matrix, (self.normalized[a] == self.normalized[b]).astype(np.float32)])
        return matrix

    def fused_score(self, method, a, b):
        values = self.score(method, a, b)
        return self.models[method].predict_proba(values)[:, 1] if method in self.models else values

def indices(frame, ids):
    (a, b) = (ids.get_indexer(frame[c]) for c in ENDS)
    assert min(a.min(), b.min()) >= 0 and np.all(a != b)
    return (a, b)

def codes(a, b):
    return np.minimum(a, b).astype(np.int64) * 50000 + np.maximum(a, b)

def check_denominator(frame, partition):
    known = frame.visual_relation_mask.to_numpy(dtype=bool)
    y = frame.visual_relation.eq('same').to_numpy()[known]
    result = dict(total=len(frame), known=int(known.sum()), positive=int(y.sum()), negative=int((~y).sum()), unknown=int((~known).sum()))
    return (known, y, result)

def graph_cell(n, a, b, scores, selection, cap, ids):
    good = np.flatnonzero(threshold_mask(scores, selection['threshold'], selection['direction']))
    utility = scores[good] if selection['direction'] == 'higher' else -scores[good]
    (u, v) = (ids[a[good]].to_numpy(), ids[b[good]].to_numpy())
    order = np.lexsort((np.maximum(u, v), np.minimum(u, v), -utility))
    edge = good[order]
    (labels, retained, rejected, counts) = build_graph(n, a[edge], b[edge], cap)
    counts.update(threshold_edges=len(good), components=n - counts['union_merges'])
    return (labels, edge[retained], counts)

def evaluate():
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    frame = manifest()
    scorer = Scorer(frame)
    scorer.set_surface('original')
    candidates = pd.read_parquet(SOURCE / 'candidates.parquet', columns=ENDS)
    assert len(candidates) == 1502683
    (a, b) = indices(candidates, scorer.ids)
    candidate_codes = pd.Index(codes(a, b))
    assert candidate_codes.is_unique
    (train, val) = (load_partition('train'), load_partition('validation'))
    (train_known, train_y, train_counts) = check_denominator(train, 'train')
    (val_known, val_y, val_counts) = check_denominator(val, 'validation')
    (ta, tb) = indices(train[train_known], scorer.ids)
    (va, vb) = indices(val[val_known], scorer.ids)
    positions = candidate_codes.get_indexer(codes(va, vb))
    recalled = positions >= 0
    print('denominators', train_counts, val_counts, flush=True)
    for method in ['B5c', 'B6']:
        model = LogisticRegression(class_weight='balanced', solver='lbfgs', C=1.0, max_iter=1000, random_state=20_260_713)
        model.fit(scorer.score(method, ta, tb), train_y)
        assert model.n_iter_.max() < 1000
        scorer.models[method] = model
        joblib.dump(model, OUT / (method + '.joblib'))
    (selections, validation_rows) = ({}, [])
    for method in METHODS:
        scores = scorer.fused_score(method, va, vb)
        selected = select_threshold(scores, val_y, recalled, 'lower' if method in ['B1a', 'B1b'] else 'higher')
        all_scores = scorer.fused_score(method, a, b)
        np.save(OUT / f'original_{method}_scores.npy', all_scores)
        rows = []
        for cap in GRID:
            (labels, edges, counts) = graph_cell(50000, a, b, all_scores, selected, cap, scorer.ids)
            co = metric(val_y, labels[va] == labels[vb])
            row = dict(method=method, k=cap, co_membership=co, **counts)
            rows.append(row)
        winner = max(rows, key=lambda r: (exact_f1(r['co_membership']), float('inf') if r['k'] == 'inf' else r['k']))
        selected.update(k_star=winner['k'], validation_co_membership=winner['co_membership'])
        selections[method] = selected
        validation_rows.extend(rows)
        print('selected', method, selected['threshold'], selected['k_star'], flush=True)
    frozen = dict(frozen_at=datetime.now(timezone.utc).isoformat(), selections=selections, test_labels_opened=False)
    dump(OUT / 'selection_frozen.json', frozen)
    dump(REPORT / 'selection_frozen.json', frozen)
    dump(REPORT / 'validation_grid.json', validation_rows)
    for surface in SURFACES:
        scorer.set_surface(surface)
        for method in METHODS:
            path = OUT / f'{surface}_{method}_scores.npy'
            scores = scorer.fused_score(method, a, b)
            np.save(path, scores)
            for (role, cap) in [('inf', 'inf'), ('selected', selections[method]['k_star'])]:
                prefix = OUT / f'{surface}_{method}_{role}'
                (labels, edges, counts) = graph_cell(50000, a, b, scores, selections[method], cap, scorer.ids)
                np.save(str(prefix) + '_components.npy', labels)
                np.save(str(prefix) + '_edges.npy', edges)
                dump(Path(str(prefix) + '_structure.json'), counts)
        print('graphs frozen', surface, flush=True)
    dump(OUT / 'graphs_frozen.json', dict(cells=80, frozen_at=datetime.now(timezone.utc).isoformat(), test_labels_opened=False))
    test = load_partition('test')
    (known, y, test_counts) = check_denominator(test, 'test')
    (qa, qb) = indices(test, scorer.ids)
    test_pos = candidate_codes.get_indexer(codes(qa, qb))
    rows = []
    for surface in SURFACES:
        pdq = pd.read_parquet(OUT / f'{surface}_pdq.parquet')
        assert np.array_equal(pdq.record_id, frame.record_id)
        quality = pdq.quality.to_numpy()
        pdq_valid = np.minimum(quality[qa], quality[qb]) >= 50
        for method in METHODS:
            scores = np.load(OUT / f'{surface}_{method}_scores.npy', mmap_mode='r')
            valid = test_pos >= 0
            feature_valid = np.zeros(len(test), dtype=bool)
            feature_valid[valid] = np.isfinite(scores[test_pos[valid]])
            for (role, cap) in [('inf', 'inf'), ('selected', selections[method]['k_star'])]:
                prefix = OUT / f'{surface}_{method}_{role}'
                labels = np.load(str(prefix) + '_components.npy')
                edges = np.load(str(prefix) + '_edges.npy')
                co = labels[qa] == labels[qb]
                direct = valid & np.isin(test_pos, edges)
                counts = json.loads(Path(str(prefix) + '_structure.json').read_text())
                rows.append(dict(surface=surface, method=method, role=role, k=cap, direct=metric(y, direct[known]), co_membership=metric(y, co[known]), unknown_n=int((~known).sum()), unknown_direct=int(direct[~known].sum()), unknown_co_membership=int(co[~known].sum()), feature_valid_known=int(pdq_valid[known].sum()) if method == 'B1b' else int(known.sum()), feature_invalid_known=int((~pdq_valid[known]).sum()) if method == 'B1b' else 0, feature_valid_unknown=int(pdq_valid[~known].sum()) if method == 'B1b' else int((~known).sum()), candidate_feature_valid_known=int(feature_valid[known].sum()), **counts))
    result = dict(run_id=RUN_ID, denominators=dict(train=train_counts, validation=val_counts, test=test_counts), candidates=1502683, selections=selections, rows=rows, resources=str(OUT))
    dump(OUT / 'results.json', result)
    dump(REPORT / 'results.json', result)
    print('completed', len(rows), 'cells', test_counts, flush=True)
PER_SURFACE = OUT / 'per_surface_candidates'
SURFACE_COUNTS = {'mixed': 1503362, 'mild': 1504724, 'strong': 1502221}

def reuse_candidate_scores(existing, positions, score_missing):
    missing = positions < 0
    scores = np.empty(len(positions), dtype=existing.dtype)
    scores[~missing] = existing[positions[~missing]]
    if missing.any():
        scores[missing] = score_missing(np.flatnonzero(missing))
    return scores

def evaluate_per_surface():
    frozen = json.loads((REPORT / 'selection_frozen.json').read_text())
    assert frozen == json.loads((OUT / 'selection_frozen.json').read_text())
    selections = frozen['selections']
    frame = manifest()
    scorer = Scorer(frame)
    scorer.models = {m: joblib.load(OUT / (m + '.joblib')) for m in ['B5c', 'B6']}
    initial_candidates = pd.read_parquet(SOURCE / 'candidates.parquet', columns=ENDS)
    (oa, ob) = indices(initial_candidates, scorer.ids)
    initial_codes = pd.Index(codes(oa, ob))
    assert len(initial_codes) == 1502683 and initial_codes.is_unique
    inputs = {}
    for (surface, expected) in SURFACE_COUNTS.items():
        source = (native_baselines if surface == 'mixed' else edit_surfaces) / surface / 'candidates.parquet'
        candidates = pd.read_parquet(source, columns=ENDS)
        assert len(candidates) == expected, (surface, len(candidates), expected)
        (a, b) = indices(candidates, scorer.ids)
        current_codes = pd.Index(codes(a, b))
        assert current_codes.is_unique
        positions = initial_codes.get_indexer(current_codes)
        inputs[surface] = dict(candidate_path=str(source), candidate_rows=len(a), common_original_pairs=int((positions >= 0).sum()), new_pairs=int((positions < 0).sum()), removed_original_pairs=int(len(initial_codes) - (positions >= 0).sum()))
        scorer.set_surface(surface)
        for method in METHODS:
            existing = np.load(OUT / f'{surface}_{method}_scores.npy', mmap_mode='r')
            assert len(existing) == len(initial_codes)
            scores = reuse_candidate_scores(existing, positions, lambda rows: scorer.fused_score(method, a[rows], b[rows]))
            np.save(PER_SURFACE / f'{surface}_{method}_scores.npy', scores)
            selected = selections[method]
            for (role, cap) in [('inf', 'inf'), ('selected', selected['k_star'])]:
                if role == 'inf' or cap != 'inf':
                    (labels, edges, counts) = graph_cell(50000, a, b, scores, selected, cap, scorer.ids)
                prefix = PER_SURFACE / f'{surface}_{method}_{role}'
                np.save(str(prefix) + '_components.npy', labels)
                np.save(str(prefix) + '_edges.npy', edges)
                dump(Path(str(prefix) + '_structure.json'), counts)
        print('per-surface graphs frozen', surface, inputs[surface], flush=True)
    dump(PER_SURFACE / 'graphs_frozen.json', dict(cells=60, frozen_at=datetime.now(timezone.utc).isoformat(), test_labels_opened=False))
    test = load_partition('test')
    (known, y, denominator) = check_denominator(test, 'test')
    (qa, qb) = indices(test, scorer.ids)
    test_codes = codes(qa, qb)
    (rows, coverage) = ([], {})
    for surface in SURFACE_COUNTS:
        candidates = pd.read_parquet(inputs[surface]['candidate_path'], columns=ENDS)
        (a, b) = indices(candidates, scorer.ids)
        test_pos = pd.Index(codes(a, b)).get_indexer(test_codes)
        recalled = test_pos >= 0
        coverage[surface] = dict(positive=int((recalled[known] & y).sum()), positive_denominator=int(y.sum()), negative=int((recalled[known] & ~y).sum()), negative_denominator=int((~y).sum()), unknown=int(recalled[~known].sum()), unknown_denominator=int((~known).sum()))
        for method in METHODS:
            for (role, cap) in [('inf', 'inf'), ('selected', selections[method]['k_star'])]:
                prefix = PER_SURFACE / f'{surface}_{method}_{role}'
                labels = np.load(str(prefix) + '_components.npy')
                edges = np.load(str(prefix) + '_edges.npy')
                counts = json.loads(Path(str(prefix) + '_structure.json').read_text())
                direct = recalled & np.isin(test_pos, edges)
                co = labels[qa] == labels[qb]
                rows.append(dict(surface=surface, method=method, role=role, k=cap, direct=metric(y, direct[known]), co_membership=metric(y, co[known]), unknown_n=int((~known).sum()), unknown_direct=int(direct[~known].sum()), unknown_co_membership=int(co[~known].sum()), **counts))
    result = dict(run_id=RUN_ID, surface=1, selections=selections, inputs=inputs, test_denominator=denominator, candidate_coverage=coverage, rows=rows, resource_directory=str(PER_SURFACE))
    dump(PER_SURFACE / 'results.json', result)
    dump(REPORT / 'per_surface_results.json', result)
    print('per-surface evaluation completed', coverage, flush=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PER_SURFACE.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    phases = {'pdq': prepare_pdq, 'simcse': prepare_simcse, 'evaluate': evaluate, 'per-surface': evaluate_per_surface}
    parser.add_argument('stage', choices=list(phases))
    args = parser.parse_args()
    phases[args.stage]()
if __name__ == '__main__':
    main()
