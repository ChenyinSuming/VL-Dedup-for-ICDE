import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import argparse
import ast
from collections import Counter, defaultdict
import importlib
import importlib.metadata as md
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
RES = RESOURCES
SOURCE = RES / 'runs/pair_features'
edit_surfaces = RES / 'runs/edit_stress'
edit_surfaces_REPO = REPO_ROOT
edit_surfaces_REPORT = edit_surfaces_REPO / str(REPORTS / 'edit_stress')
RUN = RES / 'runs/native_baselines'
REPORT = ROOT / str(REPORTS / 'native_baselines')
SEALED = RES / 'datasets/benchmark/sealed/relation_applicability.jsonl'
GRID = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995]
SURFACES = ['original', 'mixed', 'mild', 'strong']
MODELS = ['logistic', 'hist_gradient_boosting']

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def assignment(eligible):
    ordered = sorted(map(str, eligible))
    if len(ordered) != len(set(ordered)):
        raise ValueError('Duplicate eligible ID')
    selected = np.random.Generator(np.random.PCG64(20_261_027)).random(len(ordered)) < 0.5
    return dict(zip(ordered, np.where(selected, 'light', 'original')))

def exact_groups(frame):
    groups = defaultdict(list)
    for (i, row) in enumerate(frame.itertuples()):
        groups[str(row.image_path), row.text].append(i)
    return list(groups.values())

def components(n, a, b):
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for (x, y) in zip(a, b):
        (x, y) = (find(x), find(y))
        if x != y:
            parent[y] = x
    return np.array([find(i) for i in range(n)])

def pair_codes(a, b, n):
    return np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)

def confusion(truth, pred, known):
    (truth, pred, known) = map(lambda x: np.asarray(x, dtype=bool), (truth, pred, known))
    tp = int((truth & pred & known).sum())
    fp = int((~truth & pred & known).sum())
    fn = int((truth & ~pred & known).sum())
    tn = int((~truth & ~pred & known).sum())
    return dict(n=int(known.sum()), tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None, unknown=int((~known).sum()))

def version_layers(frame, pairs):
    versions = frame.set_index('record_id').version
    (x, y) = [pairs[c].map(versions).to_numpy() for c in ['record_id_a', 'record_id_b']]
    return np.where((x == 'original') & (y == 'original'), 'original_original', np.where((x == 'light') & (y == 'light'), 'light_light', 'original_light'))

def prepare():
    with threadpool_limits(limits=8):
        RUN.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(SOURCE / 'public_manifest.parquet')
        frame.record_id = frame.record_id.astype(str)
        meta = pd.read_parquet(SOURCE / 'features/clip/metadata.parquet')
        assert len(frame) == 50000 and frame.record_id.is_unique
        assert meta.record_id.astype(str).tolist() == frame.record_id.tolist()
        assert meta.row_index.tolist() == list(range(50000))
        edits = pd.read_csv(edit_surfaces_REPORT / 'stress_manifest.csv', keep_default_na=False)
        edits.record_id = edits.record_id.astype(str)
        eligible = edits.loc[edits.level == 'light', 'record_id'].tolist()
        assert len(eligible) == 5709 and len(set(eligible)) == 5709
        chosen = assignment(eligible)
        assert Counter(chosen.values()) == {'light': 2901, 'original': 2808}
        for surface in SURFACES:
            output = RUN / surface
            output.mkdir(parents=True, exist_ok=True)
            current = frame.copy()
            current['row_index'] = np.arange(50000)
            current['version'] = 'original'
            current['cache_row'] = np.arange(50000)
            if surface != 'original':
                level = 'light' if surface == 'mixed' else surface
                rows = edits[edits.level == level].set_index('record_id')
                use = current.record_id.map(chosen).eq('light') if surface == 'mixed' else current.record_id.isin(rows.index)
                current.loc[use, 'image_path'] = current.loc[use, 'record_id'].map(rows.new_path)
                current.loc[use, 'version'] = level
            current['image_cache_version'] = current.version
            current['text_cache_version'] = 'original'
            current.to_parquet(output / 'input_mapping.parquet', index=False)
            groups = exact_groups(current)
            dump(output / 'exact_groups.json', dict(key=['image_path_string', 'text'], groups=groups, groups_count=len(groups), exact_removed=50000 - len(groups)))

def namespace(name, path):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module

def official_dj():
    import PIL.Image
    from datasets import Image as DatasetImage
    site = Path(md.distribution('py-data-juicer').locate_file(''))
    dj = site / 'data_juicer'
    im = site / 'imagededup'
    for (name, path) in [('imagededup.methods', im / 'methods'), ('imagededup.handlers', im / 'handlers'), ('imagededup.utils', im / 'utils')]:
        namespace(name, path)
    hashing = importlib.import_module('imagededup.methods.hashing')
    method = sys.modules['imagededup.methods']
    for name in ['PHash', 'DHash', 'WHash', 'AHash']:
        setattr(method, name, getattr(hashing, name))
    for (name, path) in [('data_juicer', dj), ('data_juicer.utils', dj / 'utils'), ('data_juicer.ops', dj / 'ops'), ('data_juicer.ops.deduplicator', dj / 'ops/deduplicator')]:
        namespace(name, path)

    class Registry:

        def register_module(self, *args, **kwargs):
            return lambda cls: cls

    class Base:

        def __init__(self, *args, **kwargs):
            self.image_key = kwargs.get('image_key', 'images')
            self.image_bytes_key = None
    base = types.ModuleType('data_juicer.ops.base_op')
    base.OPERATORS = Registry()
    base.Deduplicator = Base
    sys.modules[base.__name__] = base
    fusion = types.ModuleType('data_juicer.ops.op_fusion')
    fusion.LOADED_IMAGES = Registry()
    sys.modules[fusion.__name__] = fusion
    const = types.ModuleType('data_juicer.utils.constant')
    const.HashKeys = type('HashKeys', (), {'imagehash': '__dj__imagehash', 'hash': '__dj__hash'})
    sys.modules[const.__name__] = const
    lazy = types.ModuleType('data_juicer.utils.lazy_loader')
    lazy.LazyLoader = lambda name: importlib.import_module(name)
    sys.modules[lazy.__name__] = lazy
    mm = types.ModuleType('data_juicer.utils.mm_utils')
    mm.PIL = PIL
    mm.io = io
    mm.Image = DatasetImage
    tree = ast.parse((dj / 'utils/mm_utils.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['load_mm_bytes_from_sample', 'load_data_with_context', 'load_image']]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(dj / 'utils/mm_utils.py'), 'exec'), mm.__dict__)
    sys.modules[mm.__name__] = mm
    doc = types.ModuleType('data_juicer.ops.deduplicator.document_deduplicator')
    doc.DocumentDeduplicator = Base
    sys.modules[doc.__name__] = doc
    module = importlib.import_module('data_juicer.ops.deduplicator.image_deduplicator')
    return (module.ImageDeduplicator(method='phash', consider_text=False), module.__file__)

class NativeDataset(list):

    def __getitem__(self, key):
        return [row[key] for row in self] if isinstance(key, str) else super().__getitem__(key)

    def filter(self, function, fn_kwargs=None, **kwargs):
        return NativeDataset((row for row in self if function(row, **fn_kwargs or {})))

def data_juicer(surface):
    output = RUN / surface
    (operator, source) = official_dj()
    frame = pd.read_parquet(output / 'input_mapping.parquet')
    rows = NativeDataset()
    for row in frame.itertuples():
        sample = {'record_id': row.record_id, 'text': row.text, 'images': [row.image_path]}
        result = operator.compute_hash(sample)
        if not result.get('__dj__imagehash'):
            raise ValueError('empty native hash')
        rows.append(result)
    (selected, _) = operator.process(rows)
    selected_ids = {r['record_id'] for r in selected}
    native = pd.DataFrame({'record_id': frame.record_id, 'native_hash': [r['__dj__imagehash'] for r in rows]})
    native['selected'] = native.record_id.isin(selected_ids)
    native.to_parquet(output / 'data_juicer_native.parquet', index=False)
    roots = pd.factorize(native.native_hash, sort=False)[0]
    np.save(output / 'data_juicer_components.npy', roots)

class NoEncoding:

    def encode(self, *args, **kwargs):
        raise RuntimeError('New encoding forbidden')

def fused_embeddings(surface):
    image = np.load(SOURCE / 'features/clip/image_emb.npy').copy()
    if surface != 'original':
        level = 'light' if surface == 'mixed' else surface
        edited = np.load(edit_surfaces / level / 'clip.npy', mmap_mode='r')
        versions = pd.read_parquet(RUN / surface / 'input_mapping.parquet').version
        mask = versions.ne('original').to_numpy()
        image[mask] = edited[mask]
    text = np.load(SOURCE / 'features/clip/text_emb.npy').copy()
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    fused = np.concatenate([image / np.sqrt(2), text / np.sqrt(2)], axis=1).astype(np.float32)
    fused /= np.linalg.norm(fused, axis=1, keepdims=True)
    return fused

def semhash(surface, thresholds):
    from semhash import SemHash
    output = RUN / surface
    with threadpool_limits(limits=8):
        RUN.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(output / 'input_mapping.parquet')
        records = [{'record_id': r.record_id, 'row_index': i, 'image': str(r.image_path), 'text': r.text} for (i, r) in enumerate(frame.itertuples())]
        index = SemHash.from_embeddings(embeddings=fused_embeddings(surface), records=records, model=NoEncoding(), columns=['image', 'text'], ann_backend='basic')
        original_query = index.index.query_threshold
        neighbors = []

        def query(*args, **kwargs):
            result = original_query(*args, **kwargs)
            neighbors[:] = [len(x) for x in result]
            return result
        index.index.query_threshold = query
    for threshold in thresholds:
        with threadpool_limits(limits=8):
            RUN.mkdir(parents=True, exist_ok=True)
            result = index.self_deduplicate(threshold=threshold)
            selected = [r['row_index'] for r in result.selected]
            links = []
            exact = []
            filtered = []
            with (output / f'semhash_{threshold:g}_filtered.jsonl').open('w') as f:
                for duplicate in result.filtered:
                    a = duplicate.record['row_index']
                    filtered.append(a)
                    entries = []
                    for (other, score) in duplicate.duplicates:
                        b = other['row_index']
                        item = (a, b, float(score))
                        (exact if duplicate.exact else links).append(item)
                        entries.append(dict(row_index=b, score=float(score)))
                    f.write(json.dumps(dict(row_index=a, exact=bool(duplicate.exact), duplicates=entries)) + '\n')
            links_array = np.asarray(links, dtype=np.float64).reshape(-1, 3)
            exact_array = np.asarray(exact, dtype=np.float64).reshape(-1, 3)
            np.savez_compressed(output / f'semhash_{threshold:g}_native.npz', selected=np.asarray(selected), filtered=np.asarray(filtered), native_similarity_link=links_array, exact_relation=exact_array)
            all_links = np.concatenate([links_array[:, :2], exact_array[:, :2]]).astype(int)
            roots = components(len(frame), all_links[:, 0], all_links[:, 1])
            np.save(output / f'semhash_{threshold:g}_components.npy', roots)

def available_methods(surface, roles):
    return ['data_juicer'] + [f'semhash_{t:g}' for t in roles]

def load_reference():
    with SEALED.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    reference = pd.DataFrame(rows)
    eligible = set(pd.read_parquet(RUN / 'mixed/input_mapping.parquet').query("version == 'light'").record_id)
    edits = pd.read_csv(edit_surfaces_REPORT / 'stress_manifest.csv', keep_default_na=False)
    eligible = set(edits.loc[edits.level == 'light', 'record_id'].astype(str))
    validation = reference[reference.partition.eq('validation')]
    endpoints = set(validation.record_id_a.astype(str)) | set(validation.record_id_b.astype(str))
    intersection = eligible & endpoints
    assert len(endpoints) == 5886 and (not intersection)
    m1 = pd.read_parquet(RUN / 'mixed/input_mapping.parquet')
    test = reference[reference.partition.eq('test')]
    layers = version_layers(m1, test)
    known = test.visual_relation_mask.to_numpy(dtype=bool)
    truth = test.visual_relation.eq('same').to_numpy()
    expected = {'original_original': (2121, 200, 132), 'light_light': (2235, 209, 145), 'original_light': (4345, 384, 266)}
    actual = {k: (int(((layers == k) & known & truth).sum()), int(((layers == k) & known & ~truth).sum()), int(((layers == k) & ~known).sum())) for k in expected}
    assert actual == expected
    return reference

def predictions(surface, method, pairs):
    frame = pd.read_parquet(RUN / surface / 'input_mapping.parquet')
    ids = pd.Index(frame.record_id)
    (a, b) = [ids.get_indexer(pairs[c].astype(str)) for c in ['record_id_a', 'record_id_b']]
    assert min(a.min(), b.min()) >= 0
    if method in MODELS:
        if surface == 'mixed':
            edges = np.load(RUN / surface / f'{method}_selected_edges.npz')
            codes = np.unique(pair_codes(edges['a'], edges['b'], len(frame)))
            roots = np.load(RUN / surface / f'{method}_components.npy')
            return (np.isin(pair_codes(a, b, len(frame)), codes), roots[a] == roots[b])
        if surface == 'original':
            reference = pd.read_parquet(SOURCE / 'full_test_predictions.parquet')
            order = pd.MultiIndex.from_frame(reference[['record_id_a', 'record_id_b']]).get_indexer(pd.MultiIndex.from_frame(pairs[['record_id_a', 'record_id_b']]))
            assert min(order) >= 0
            return (reference[method + '_direct_selected'].to_numpy()[order], reference[method + '_co_membership'].to_numpy()[order])
        reference = pd.read_parquet(edit_surfaces / 'test_identity.parquet')
        order = pd.MultiIndex.from_frame(reference[['record_id_a', 'record_id_b']]).get_indexer(pd.MultiIndex.from_frame(pairs[['record_id_a', 'record_id_b']]))
        assert min(order) >= 0
        native = np.load(edit_surfaces / surface / f'{method}_test.npz')
        return (native['direct'][order], native['component'][order])
    if method == 'data_juicer':
        roots = np.load(RUN / surface / 'data_juicer_components.npy')
        return (roots[a] == roots[b], roots[a] == roots[b])
    t = float(method.split('_', 1)[1])
    native = np.load(RUN / surface / f'semhash_{t:g}_native.npz')
    links = native['native_similarity_link'][:, :2].astype(int)
    codes = np.unique(pair_codes(links[:, 0], links[:, 1], len(frame)))
    direct = np.isin(pair_codes(a, b, len(frame)), codes)
    roots = np.load(RUN / surface / f'semhash_{t:g}_components.npy')
    return (direct, roots[a] == roots[b])

def evaluate(surface, methods, split='test', suffix=''):
    reference = load_reference()
    pairs = reference[reference.partition.eq(split)].copy().reset_index(drop=True)
    known = pairs.visual_relation_mask.to_numpy(dtype=bool)
    truth = pairs.visual_relation.eq('same').to_numpy()
    assert (len(pairs), int(known.sum()), int((known & truth).sum())) == ((10037, 9494, 8701) if split == 'test' else (10326, 9766, int((known & truth).sum())))
    frame = pd.read_parquet(RUN / surface / 'input_mapping.parquet')
    layers = version_layers(frame, pairs) if surface == 'mixed' else np.full(len(pairs), 'full')
    if surface == 'mixed':
        expected = {'original_original': (2121, 200, 132), 'light_light': (2235, 209, 145), 'original_light': (4345, 384, 266)}
        for (layer, counts) in expected.items():
            mask = layers == layer
            assert (int((mask & known & truth).sum()), int((mask & known & ~truth).sum()), int((mask & ~known).sum())) == counts
    result = {}
    for method in methods:
        (direct, component) = predictions(surface, method, pairs)
        cell = {}
        for layer in ['full'] + (['original_original', 'light_light', 'original_light'] if surface == 'mixed' else []):
            mask = np.ones(len(pairs), dtype=bool) if layer == 'full' else layers == layer
            cell[layer] = {name: confusion(truth[mask], pred[mask], known[mask]) for (name, pred) in [('direct', direct), ('component', component)]}
            risks = {}
            for category in ['L3', 'M', 'HN']:
                col = 'grade'
                if col not in pairs:
                    raise ValueError('No typed risk column: ' + str(pairs.columns.tolist()))
                applicable = pairs[col].eq(category).to_numpy() & mask
                n = int(applicable.sum())
                risks[category] = dict(n=n, direct=int(direct[applicable].sum()), component=int(component[applicable].sum()), direct_rate=float(direct[applicable].mean()) if n else None, component_rate=float(component[applicable].mean()) if n else None)
            cell[layer]['risks'] = risks
        result[method] = cell
    dump(REPORT / f'{surface}_{split}{suffix}.json', dict(surface=surface, split=split, results=result))
    return result

def select():
    path = RUN / 'selection.json'
    with threadpool_limits(limits=8):
        RUN.mkdir(parents=True, exist_ok=True)
        results = evaluate('original', [f'semhash_{t:g}' for t in GRID], split='validation')
        threshold = min(GRID, key=lambda t: (-results[f'semhash_{t:g}']['full']['component']['f1'], results[f'semhash_{t:g}']['full']['component']['fp'], -t))
        dump(path, dict(threshold=threshold, rule='validation component F1 descending, FP ascending, threshold descending', results=results, test_used=False))
    return threshold

def baseline_m1():
    output = RUN / 'mixed'
    from scripts import build_candidates as b
    from scripts.pair_features import ObservableFeatures
    import faiss
    import joblib
    with threadpool_limits(limits=8):
        RUN.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(output / 'input_mapping.parquet')
        frame['partition'] = 'public_global'
        use = frame.version.eq('light').to_numpy()
        image = np.load(SOURCE / 'features/clip/image_emb.npy').copy()
        image[use] = np.load(edit_surfaces / 'light/clip.npy', mmap_mode='r')[use]
        text = np.load(SOURCE / 'features/clip/text_emb.npy').copy()
        hashes = pd.read_parquet(edit_surfaces / 'original/features/phash/phash.parquet').phash_int.to_numpy(dtype=np.uint64)
        hashes[use] = pd.read_parquet(edit_surfaces / 'light/features/phash/phash.parquet').phash_int.to_numpy(dtype=np.uint64)[use]
        for channel in ['sscd', 'dinov2']:
            matrix = np.load(edit_surfaces / 'original' / f'{channel}.npy').copy()
            matrix[use] = np.load(edit_surfaces / 'light' / f'{channel}.npy', mmap_mode='r')[use]
            np.save(output / f'{channel}.npy', matrix)
        np.save(output / 'clip.npy', image)
        p = output / 'features/phash'
        p.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'record_id': frame.record_id, 'phash_int': hashes}).to_parquet(p / 'phash.parquet', index=False)
        candidate_path = output / 'candidates.parquet'
    with threadpool_limits(limits=8):
        RUN.mkdir(parents=True, exist_ok=True)
        faiss.omp_set_num_threads(8)
        edges = {}
        image_norm = image.copy()
        text_norm = text.copy()
        faiss.normalize_L2(image_norm)
        faiss.normalize_L2(text_norm)
        b.build_phash_edges(frame, output / 'features', 2, edges)
        b.build_embedding_edges(image_norm, 'clip_image', 'clip_image_score', 30, None, edges, frame.partition.to_numpy())
        b.build_embedding_edges(text_norm, 'clip_text', 'clip_text_score', 10, None, edges, frame.partition.to_numpy())
        b.build_text_exact_edges(frame, 20, edges)
        candidates = b.edges_to_frame(edges, frame)
        candidates.to_parquet(candidate_path, index=False)
    with threadpool_limits(limits=8):
        RUN.mkdir(parents=True, exist_ok=True)
        candidates = pd.read_parquet(candidate_path)
        feature = ObservableFeatures(frame, image, text, hashes)
        ids = pd.Index(frame.record_id)
        (a, b) = [ids.get_indexer(candidates[c]) for c in ['record_id_a', 'record_id_b']]
        for name in MODELS:
            model_path = SOURCE / (name + '.joblib')
            bundle = joblib.load(model_path)
            scores = np.empty(len(candidates))
            for start in range(0, len(candidates), 32768):
                features = feature.transform(candidates.iloc[start:start + 32768])
                probabilities = bundle['model'].predict_proba(features)
                names = list(bundle['model'].classes_)
                raw = probabilities[:, [names.index('duplicate'), names.index('keep_augment_candidate')]].sum(axis=1)
                scores[start:start + len(features)] = bundle['calibrator'].predict(raw)
            assert np.isfinite(scores).all()
            selected = scores >= 1.0
            roots = components(len(frame), a[selected], b[selected])
            np.save(output / f'{name}_candidate_probability.npy', scores)
            np.save(output / f'{name}_components.npy', roots)
            np.savez_compressed(output / f'{name}_selected_edges.npz', a=a[selected], b=b[selected])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'original', 'stress', 'baseline', 'all'])
    args = parser.parse_args()
    prepare()
    if args.command == 'prepare':
        return
    if args.command in ['original', 'all']:
        data_juicer('original')
        semhash('original', GRID)
        threshold = select()
        roles = sorted(set([threshold, 0.9, 0.99]))
        with threadpool_limits(limits=8):
            RUN.mkdir(parents=True, exist_ok=True)
            evaluate('original', available_methods('original', roles))
    if args.command in ['stress', 'all']:
        threshold = select()
        roles = sorted(set([threshold, 0.9, 0.99]))
        for surface in ['mixed', 'mild', 'strong']:
            data_juicer(surface)
            semhash(surface, roles)
            with threadpool_limits(limits=8):
                RUN.mkdir(parents=True, exist_ok=True)
                evaluate(surface, available_methods(surface, roles))
    if args.command in ['baseline', 'all']:
        baseline_m1()
if __name__ == '__main__':
    main()
