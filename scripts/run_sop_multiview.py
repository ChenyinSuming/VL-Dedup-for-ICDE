import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import os
for _thread_name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_name] = '4'
os.environ['GRAPH_CPU_THREADS'] = '4'
import argparse
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import re
import sys
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import sklearn
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from scripts import build_candidates as retrieval
from scripts.baseline_matrix_text import normalize_baseline_text
from scripts.pair_features import ObservableFeatures, choose_threshold
from scripts.train_hgb_classifier import FEATURES, make_models, raw_probability, pairwise_counts, confusion
from scripts.cluster_with_size_cap import GRID, build_graph, edge_order
RES = RESOURCES
RUN_ID = 'sop_multiview'
OUT = RES / 'runs' / RUN_ID
REPORT = ROOT / str(REPORTS / 'sop_multiview')
DATA = RES / 'datasets/benchmark'
SOURCE = RES / 'runs/pair_features'
baseline_surfaces = RES / 'runs/baselines_surfaces'
hgb_classifier = RES / 'runs/hgb_sscd'
edit_surfaces = RES / 'runs/edit_stress'
INPUT = OUT / 'input'
ENDS = ['record_id_a', 'record_id_b']
SCORERS = ['B2', 'B5a', 'B5c', 'B6', 'LR_frozen', 'HGB_frozen', 'LR_retrained', 'HGB_retrained']
THRESHOLDS = dict(B2=0.828769862651825, B5a=0.9999997615814209, B5c=0.9791054227081195, B6=0.9259631557561843)
EXTENDED_CAPTION_IDS = frozenset({'sop_multiview_record_040090_view_1', 'sop_multiview_record_013867_view_0', 'sop_multiview_record_009802_view_1', 'sop_multiview_record_036713_view_1', 'sop_multiview_record_016691_view_1', 'sop_multiview_record_016932_view_0', 'sop_multiview_record_048497_view_0', 'sop_multiview_record_045672_view_0', 'sop_multiview_record_032733_view_1', 'sop_multiview_record_035300_view_0', 'sop_multiview_record_025098_view_1'})

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def read_partition(path, partition):
    pattern = re.compile('"partition"\\s*:\\s*"' + partition + '"')
    with Path(path).open() as stream:
        return pd.DataFrame([json.loads(line) for line in stream if pattern.search(line)])

def require_selection():
    return read(OUT / 'selection_frozen.json')

def references(partition, lineage=False):
    filename = 'private/lineage.jsonl' if lineage else 'sealed/relation_applicability.jsonl'
    frame = read_partition(DATA / filename, partition)
    assert len(frame) and frame.partition.eq(partition).all()
    return frame

def base_metadata():
    with (DATA / 'private/lineage.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    frame = pd.DataFrame(rows)
    selected = pd.read_csv(RES / 'datasets/VL-Dedup-sources/metadata/stanford_online_products/selected_manifest.csv')
    categories = dict(zip(selected.source_asset_id, selected.caption.str.extract('super_class_id=(\\d+)', expand=False).astype(int)))
    source_paths = dict(zip(selected.source_asset_id, selected.image_path.str.removeprefix('raw/stanford_online_products/')))
    sop = frame.source_dataset.eq('stanford_online_products')
    frame['super_class_id'] = [categories[s.removeprefix('sop_')] if is_sop else -1 for (s, is_sop) in zip(frame.source_image_id, sop)]
    frame['is_sop'] = sop
    frame['sop_path'] = [source_paths[s.removeprefix('sop_')] if is_sop else '' for (s, is_sop) in zip(frame.source_image_id, sop)]
    return frame

def validate_manifest(manifest, bases):
    columns = ['base_record_id', 'partition', 'class_id', 'super_class_id', 'copy_index', 'sop_path', 'file']
    if not set(columns) <= set(manifest):
        raise ValueError('Missing subset manifest columns')
    if manifest[columns].isna().any().any() or not len(manifest):
        raise ValueError('Empty or incomplete subset manifest')
    frame = manifest.copy()
    base = bases.set_index('record_id')
    if not frame.base_record_id.isin(base.index).all():
        raise ValueError('Unknown base record')
    matched = base.loc[frame.base_record_id]
    if not matched.is_sop.all() or not np.array_equal(frame.partition, matched.partition):
        raise ValueError('Manifest base/partition mismatch')
    if not np.array_equal(frame.class_id.astype(int), matched.sop_global_class_id.astype(int)):
        raise ValueError('Manifest product mismatch')
    if not np.array_equal(frame.super_class_id.astype(int), matched.super_class_id):
        raise ValueError('Manifest super-class mismatch')
    if not frame.file.eq(frame.sop_path.str.replace('/', '__', regex=False)).all():
        raise ValueError('Manifest filename is not flattened SOP path')
    if (frame.sop_path.to_numpy() == matched.sop_path.to_numpy()).any():
        raise ValueError('Manifest repeats the base photograph')
    if frame.file.str.contains('[/\\\\]').any() or (frame.copy_index.astype(int) < 0).any():
        raise ValueError('Invalid subset filename/copy index')
    if frame.duplicated(['base_record_id', 'copy_index']).any() or frame.file.duplicated().any():
        raise ValueError('Duplicate manifest identity')
    if frame.groupby('base_record_id').size().max() > 2:
        raise ValueError('More than two new images per product')
    if frame[frame.partition.eq('train')].base_record_id.nunique() > 1500:
        raise ValueError('Training product budget exceeded')
    frame['record_id'] = ['sop_multiview_' + str(b) + '_view_' + str(int(i)) for (b, i) in zip(frame.base_record_id, frame.copy_index)]
    frame['visual_group_id'] = matched.visual_group_id.to_numpy()
    if not frame.record_id.is_unique or frame.record_id.isin(base.index).any():
        raise ValueError('New record identity collision')
    return frame.sort_values('record_id').reset_index(drop=True)

def canonical_pairs(frame):
    result = frame.copy()
    (a, b) = (result[ENDS[0]].astype(str), result[ENDS[1]].astype(str))
    if a.eq(b).any():
        raise ValueError('Self pair')
    (result[ENDS[0]], result[ENDS[1]]) = (np.minimum(a, b), np.maximum(a, b))
    return result.sort_values(ENDS).reset_index(drop=True)

def pair_keys(frame):
    return set(frame[ENDS].itertuples(index=False, name=None))

def positive_pairs(records):
    rows = []
    for (_, local) in records.sort_values('record_id').groupby('visual_group_id', sort=True):
        rows.extend(combinations(local.record_id.tolist(), 2))
    return pd.DataFrame(rows, columns=ENDS).sort_values(ENDS).reset_index(drop=True)

def sample_negative_pairs(records, count, seed, excluded=()):
    frame = records.sort_values('record_id').reset_index(drop=True)
    ids = frame.record_id.to_numpy()
    (groups, categories) = (frame.visual_group_id.to_numpy(), frame.super_class_id.to_numpy())
    excluded = set(excluded)
    available = sum((len(g) * (len(g) - 1) // 2 for (_, g) in frame.groupby('super_class_id')))
    available -= sum((len(g) * (len(g) - 1) // 2 for (_, g) in frame.groupby(['super_class_id', 'visual_group_id'])))
    index = {r: i for (i, r) in enumerate(ids)}
    available -= sum((a in index and b in index and (categories[index[a]] == categories[index[b]]) and (groups[index[a]] != groups[index[b]]) for (a, b) in excluded))
    if count > available or count < 0:
        raise ValueError('Insufficient eligible negative pairs')
    rng = np.random.Generator(np.random.PCG64(seed))
    chosen = set()
    while len(chosen) < count:
        (a, b) = sorted(rng.integers(0, len(frame), size=2).tolist())
        if a == b or categories[a] != categories[b] or groups[a] == groups[b]:
            continue
        key = (ids[a], ids[b])
        if key not in excluded:
            chosen.add(key)
    return pd.DataFrame(sorted(chosen), columns=ENDS)

def labeled_pairs(records, original, partition):
    sop = records[records.is_sop].copy()
    positive = positive_pairs(sop)
    original = canonical_pairs(original)
    if partition == 'train':
        reference = original[original.graded_relation_mask].copy()
        exclude = pair_keys(reference)
        seed = 20_261_031
    else:
        reference = original[original.relation.eq('hard_negative')].copy()
        exclude = pair_keys(reference)
        seed = 20_261_032
    negative = sample_negative_pairs(sop, len(positive), seed, excluded=exclude)
    positive = positive.assign(relation='duplicate', visual_relation='same')
    negative = negative.assign(relation='hard_negative', visual_relation='different')
    added = pd.concat([positive, negative], ignore_index=True).assign(partition=partition, graded_relation_mask=True, visual_relation_mask=True, origin='multiview_SOP')
    reference['origin'] = 'original'
    combined = canonical_pairs(pd.concat([reference, added], ignore_index=True))
    for (_, duplicate) in combined[combined.duplicated(ENDS, keep=False)].groupby(ENDS):
        if duplicate.relation.nunique() != 1:
            raise ValueError('Conflicting labels on a repeated pair')
    combined = combined.drop_duplicates(ENDS).reset_index(drop=True)
    assert combined[ENDS].duplicated().sum() == 0
    assert len(combined) == len(reference) + 2 * len(positive)
    return (combined, dict(total=len(combined), added_positive=len(positive), sampled_negative=len(negative), initial_pairs=len(reference), positive=int(combined.visual_relation.eq('same').sum()), negative=int(combined.visual_relation.eq('different').sum()), seed=seed))

def select_k(rows):
    if len(rows) != len(GRID) or {r['k'] for r in rows} != set(GRID):
        raise ValueError('Complete K grid required')
    return min(rows, key=lambda r: (-Fraction(r['pairwise']['exact_f1']), GRID.index(r['k'])))['k']

def partition_records(partition):
    base = references(partition, lineage=True)
    selected = pd.read_csv(RES / 'datasets/VL-Dedup-sources/metadata/stanford_online_products/selected_manifest.csv')
    categories = dict(zip(selected.source_asset_id, selected.caption.str.extract('super_class_id=(\\d+)', expand=False).astype(int)))
    base['is_sop'] = base.source_dataset.eq('stanford_online_products')
    base['super_class_id'] = [categories[s.removeprefix('sop_')] if q else -1 for (s, q) in zip(base.source_image_id, base.is_sop)]
    new = pd.read_parquet(OUT / 'new_manifest.parquet')
    new = new[new.partition.eq(partition)].copy().assign(is_sop=True)
    columns = ['record_id', 'visual_group_id', 'partition', 'super_class_id', 'is_sop']
    result = pd.concat([base[columns], new[columns]], ignore_index=True).sort_values('record_id')
    assert result.record_id.is_unique
    return result

def inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    official = pd.read_csv(INPUT / 'Ebay_info.txt', sep='\\s+').set_index('path')
    bases = base_metadata()
    sop_mask = bases.is_sop
    bases['sop_global_class_id'] = -1
    bases.loc[sop_mask, 'sop_global_class_id'] = official.loc[bases.loc[sop_mask, 'sop_path'], 'class_id'].to_numpy()
    manifest = validate_manifest(pd.read_csv(INPUT / 'subset_manifest.csv'), bases)
    manifest['image_path'] = manifest.file.map(lambda f: str(INPUT / 'subset' / f))
    manifest.to_parquet(OUT / 'new_manifest.parquet', index=False)
    contract = read(DATA / 'contract.json')
    requests = [dict(id=r.record_id, messages=[dict(role='user', content=[dict(type='image', url=r.image_path), dict(type='text', text=contract['generation_prompt'])])]) for r in manifest.itertuples()]
    with (OUT / 'caption_requests.jsonl').open('w') as stream:
        for row in requests:
            stream.write(json.dumps(row) + '\n')
    construction = {}
    for (partition, local) in manifest.groupby('partition'):
        sizes = local.groupby('base_record_id').size()
        construction[partition] = dict(new_records=len(local), products=len(sizes), one_added=int((sizes == 1).sum()), two_added=int((sizes == 2).sum()), new_positive_pairs=sum((int(k) * (int(k) + 1) // 2 for k in sizes)))
    dump(OUT / 'construction_budget.json', construction)

def captions():
    from types import SimpleNamespace
    import torch
    from scripts.run_image_caption_batch import run
    from scripts.assemble_captions import qc
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'caption_requests.jsonl').open() as stream:
        requests = [json.loads(line) for line in stream]
    extended = [r for r in requests if r['id'] in EXTENDED_CAPTION_IDS]
    extended_path = OUT / 'caption_extended_requests.jsonl'
    extended_path.write_text(''.join((json.dumps(r) + '\n' for r in extended)))
    accepted = {}
    for (request_path, output, token_limit) in [(OUT / 'caption_requests.jsonl', OUT / 'captions', 80), (extended_path, OUT / 'captions_extended', 160)]:
        torch.set_num_threads(4)
        torch.manual_seed(23)
        run(SimpleNamespace(requests=request_path, model=RES / 'models/InternVL3-2B-hf', output=output, family='internvl', max_new_tokens=token_limit, batch_size=48, max_patches=1))
        with (output / 'responses.jsonl').open() as stream:
            responses = [json.loads(line) for line in stream]
        quality = [dict(record_id=r['id'], **qc(r)) for r in responses]
        (output / 'quality.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in quality)))
        for row in quality:
            if token_limit == 80 and row['record_id'] in EXTENDED_CAPTION_IDS:
                continue
            if row['text'] is None:
                raise ValueError('Caption did not pass quality checks: ' + row['record_id'])
            accepted[row['record_id']] = row['text']
    frame = pd.read_parquet(OUT / 'new_manifest.parquet')
    if set(accepted) != set(frame.record_id):
        raise ValueError('Caption record IDs do not match the multiview manifest')
    frame['text'] = frame.record_id.map(accepted)
    frame.to_parquet(OUT / 'new_records.parquet', index=False)

def encode():
    import torch
    from PIL import Image
    import imagehash
    from torchvision import transforms
    from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor
    from scripts.extract_features import clip_feature_tensor
    with threadpool_limits(limits=4):
        OUT.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(4)
        torch.manual_seed(23)
        new = pd.read_parquet(OUT / 'new_records.parquet')
        texts = new.text.tolist()
        n = len(new)
        image_vectors = np.empty((n, 768), dtype=np.float32)
        text_vectors = np.empty_like(image_vectors)
        root = RES / 'models/clip-vit-large-patch14-336'
        processor = CLIPProcessor.from_pretrained(root, local_files_only=True)
        model = CLIPModel.from_pretrained(root, local_files_only=True).to('cuda').eval()
        with torch.inference_mode():
            for start in range(0, n, 48):
                paths = new.image_path.iloc[start:start + 48]
                images = [Image.open(p).convert('RGB') for p in paths]
                inputs = processor(images=images, return_tensors='pt').to('cuda')
                output = clip_feature_tensor(model.get_image_features(**inputs))
                image_vectors[start:start + len(images)] = torch.nn.functional.normalize(output.float(), dim=1).cpu().numpy()
                for image in images:
                    image.close()
                inputs = processor.tokenizer(texts[start:start + 48], padding=True, truncation=True, max_length=77, return_tensors='pt').to('cuda')
                output = clip_feature_tensor(model.get_text_features(**inputs))
                text_vectors[start:start + len(paths)] = torch.nn.functional.normalize(output.float(), dim=1).cpu().numpy()
                if start % 480 == 0:
                    print('CLIP', start, '/', n, flush=True)
        del model, processor
        torch.cuda.empty_cache()
        model = torch.jit.load(str(RES / 'runs/VL-Dedup-runs/current/models/sscd/sscd_disc_mixup.torchscript.pt'), map_location='cuda').eval()
        preprocess = transforms.Compose([transforms.Resize(288, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(288), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        sscd = np.empty((n, 512), dtype=np.float32)
        hashes = []
        with torch.inference_mode():
            for start in range(0, n, 48):
                images = [Image.open(p).convert('RGB') for p in new.image_path.iloc[start:start + 48]]
                hashes.extend((int(str(imagehash.phash(im)), 16) for im in images))
                batch = torch.stack([preprocess(im) for im in images]).to('cuda')
                output = model(batch)
                sscd[start:start + len(images)] = torch.nn.functional.normalize(output.float(), dim=1).cpu().numpy()
                for image in images:
                    image.close()
                if start % 480 == 0:
                    print('SSCD', start, '/', n, flush=True)
        del model
        torch.cuda.empty_cache()
        root = baseline_surfaces / 'models/simcse'
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        model = AutoModel.from_pretrained(root, local_files_only=True).to('cuda').eval()
        simcse = np.empty((n, model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, n, 128):
                inputs = tokenizer(texts[start:start + 128], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
                output = model(**inputs).pooler_output
                simcse[start:start + len(output)] = torch.nn.functional.normalize(output.float(), dim=1).cpu().numpy()
        torch.cuda.synchronize()
        for (name, vectors) in dict(image=image_vectors, text=text_vectors, sscd=sscd, simcse=simcse).items():
            assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=3e-06)
            np.save(OUT / ('new_' + name + '.npy'), vectors)
        pd.DataFrame(dict(record_id=new.record_id, phash_int=np.array(hashes, dtype=np.uint64))).to_parquet(OUT / 'new_phash.parquet', index=False)

def assemble():
    base = pd.read_parquet(SOURCE / 'public_manifest.parquet')
    meta = pd.read_parquet(SOURCE / 'features/clip/metadata.parquet')
    hashes = pd.read_parquet(SOURCE / 'features/phash/phash.parquet')
    assert len(base) == 50000 and np.array_equal(base.record_id, meta.record_id)
    assert np.array_equal(meta.row_index, np.arange(50000))
    assert np.array_equal(hashes.record_id, base.record_id)
    assert np.array_equal(pd.read_parquet(baseline_surfaces / 'simcse_metadata.parquet').record_id, base.record_id)
    new = pd.read_parquet(OUT / 'new_records.parquet')
    new_hash = pd.read_parquet(OUT / 'new_phash.parquet')
    assert np.array_equal(new.record_id, new_hash.record_id)
    frame = pd.concat([base, new[base.columns]], ignore_index=True)
    order = np.argsort(frame.record_id.to_numpy(), kind='stable')
    frame = frame.iloc[order].reset_index(drop=True)
    assert frame.record_id.is_unique and frame.record_id.is_monotonic_increasing
    frame.to_parquet(OUT / 'public_manifest.parquet', index=False)
    paths = dict(image=SOURCE / 'features/clip/image_emb.npy', text=SOURCE / 'features/clip/text_emb.npy', sscd=edit_surfaces / 'original/sscd.npy', simcse=baseline_surfaces / 'simcse.npy')
    for (name, path) in paths.items():
        reference = np.load(path, mmap_mode='r')
        added = np.load(OUT / ('new_' + name + '.npy'), mmap_mode='r')
        assert len(reference) == len(base) and len(added) == len(new)
        np.save(OUT / (name + '.npy'), np.concatenate([reference, added])[order])
    merged_hash = pd.concat([hashes[['record_id', 'phash_int']], new_hash], ignore_index=True).iloc[order]
    target = OUT / 'features/phash/phash.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    merged_hash.to_parquet(target, index=False)
    pd.DataFrame(dict(record_id=frame.record_id, row_index=np.arange(len(frame)))).to_parquet(OUT / 'embedding_metadata.parquet', index=False)
    return frame

def candidates():
    with threadpool_limits(limits=4):
        OUT.mkdir(parents=True, exist_ok=True)
        frame = assemble()
        global_frame = frame.assign(partition='sop_multiview_global')
        (edges, counts) = ({}, {})
        retrieval.faiss.omp_set_num_threads(4)
        counts['phash'] = retrieval.build_phash_edges(global_frame, OUT / 'features', 2, edges)
        for (channel, name, k) in [('clip_image', 'image', 30), ('clip_text', 'text', 10)]:
            vectors = np.array(np.load(OUT / (name + '.npy'), mmap_mode='r'), dtype=np.float32, order='C', copy=True)
            retrieval.faiss.normalize_L2(vectors)
            counts[channel] = retrieval.build_embedding_edges(vectors, channel, channel + '_score', k, None, edges, global_frame.partition.to_numpy())
        counts['text_exact'] = retrieval.build_text_exact_edges(global_frame, 20, edges)
        result = retrieval.edges_to_frame(edges, global_frame)
        assert len(result) == result.edge_id.nunique()
        result.to_parquet(OUT / 'candidates.parquet', index=False)

class Features:

    def __init__(self):
        self.frame = pd.read_parquet(OUT / 'public_manifest.parquet')
        self.ids = pd.Index(self.frame.record_id)
        meta = pd.read_parquet(OUT / 'embedding_metadata.parquet')
        assert np.array_equal(meta.record_id, self.ids) and np.array_equal(meta.row_index, np.arange(len(meta)))
        (self.image, self.text, self.sscd, self.simcse) = [np.load(OUT / (name + '.npy'), mmap_mode='r') for name in ['image', 'text', 'sscd', 'simcse']]
        hashes = pd.read_parquet(OUT / 'features/phash/phash.parquet')
        assert np.array_equal(hashes.record_id, self.ids)
        self.observable = ObservableFeatures(self.frame, self.image, self.text, hashes.phash_int)
        self.normalized = self.frame.text.map(normalize_baseline_text).to_numpy()

    def indices(self, pairs):
        (a, b) = [self.ids.get_indexer(pairs[c]) for c in ENDS]
        assert (a >= 0).all() and (b >= 0).all() and (a != b).all()
        return (a, b)

    def transform(self, pairs):
        (a, b) = self.indices(pairs)
        x = self.observable.transform(pairs)
        result = np.column_stack([x, np.sum(self.sscd[a] * self.sscd[b], axis=1, dtype=np.float32)])
        assert result.shape == (len(pairs), 12) and np.isfinite(result).all()
        return result

    def baseline(self, name, pairs, model=None):
        (a, b) = self.indices(pairs)
        if name == 'B2':
            return np.sum(self.sscd[a] * self.sscd[b], axis=1, dtype=np.float32)
        if name == 'B5a':
            return np.sum(self.image[a] * self.image[b], axis=1, dtype=np.float32)
        if name == 'B5c':
            x = np.column_stack([self.baseline('B5a', pairs), np.sum(self.text[a] * self.text[b], axis=1, dtype=np.float32)])
        elif name == 'B6':
            x = np.column_stack([self.baseline('B2', pairs), np.sum(self.simcse[a] * self.simcse[b], axis=1, dtype=np.float32), (self.normalized[a] == self.normalized[b]).astype(np.float32)])
        else:
            raise ValueError(name)
        return model.predict_proba(x)[:, 1]

def codes_for(pairs, ids):
    (a, b) = [ids.get_indexer(pairs[c]) for c in ENDS]
    assert (a >= 0).all() and (b >= 0).all()
    return np.minimum(a, b).astype(np.int64) * len(ids) + np.maximum(a, b)

def score_candidates(feature, pairs, bundles):
    arrays = {m: {kind: np.lib.format.open_memmap(OUT / f'{m}_{kind}.npy', mode='w+', dtype=np.float64, shape=(len(pairs),)) for kind in ['raw', 'score']} for m in SCORERS}
    models = {m: joblib.load(baseline_surfaces / (m + '.joblib')) for m in ['B5c', 'B6']}
    for start in range(0, len(pairs), 8192):
        batch = pairs.iloc[start:start + 8192]
        x = feature.transform(batch)
        for name in SCORERS:
            if name in THRESHOLDS:
                raw = feature.baseline(name, batch, models.get(name))
                score = raw
            else:
                bundle = bundles[name]
                raw = raw_probability(bundle['model'], x)
                score = bundle['calibrator'].predict(raw)
            arrays[name]['raw'][start:start + len(batch)] = raw
            arrays[name]['score'][start:start + len(batch)] = score
        if start % 81920 == 0:
            print('score', start, '/', len(pairs), flush=True)
    for values in arrays.values():
        for array in values.values():
            array.flush()
    dump(OUT / 'scoring_frozen.json', dict(run_id=RUN_ID, at_utc=now(), scorers=SCORERS, candidates=len(pairs), score_dtype='float64', features=FEATURES))

def build_cell(name, cap, a, b, ordered, n):
    (labels, retained, rejected, counts) = build_graph(n, a[ordered], b[ordered], cap)
    path = OUT / 'graphs' / name / str(cap)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'components.npy', labels)
    np.save(path / 'retained_candidate_indices.npy', ordered[retained])
    np.save(path / 'rejections.npy', rejected)
    sizes = np.unique(labels, return_counts=True)[1]
    structure = dict(global_components=len(sizes), singletons=int((sizes == 1).sum()), largest_component=int(sizes.max()), U=sum((int(s) * (int(s) - 1) // 2 for s in sizes)), **counts)
    assert n - counts['union_merges'] == len(sizes)
    result = dict(scorer=name, k=cap, positive_candidate_edges=len(ordered), structure=structure, artifacts=str(path), frozen_at_utc=now())
    dump(path / 'cell.json', result)
    return result

def cell_metrics(cell, pairs, records, ids, candidate_codes, sop_only=False):
    labels = np.load(Path(cell['artifacts']) / 'components.npy')
    retained = np.load(Path(cell['artifacts']) / 'retained_candidate_indices.npy')
    (a, b) = [ids.get_indexer(pairs[c]) for c in ENDS]
    assert (a >= 0).all() and (b >= 0).all()
    direct = np.isin(codes_for(pairs, ids), candidate_codes[retained])
    co = labels[a] == labels[b]
    known = pairs.visual_relation_mask.fillna(False).to_numpy(dtype=bool)
    y = pairs.visual_relation.eq('same').to_numpy()
    scope = records[records.is_sop] if sop_only else records
    positions = ids.get_indexer(scope.record_id)
    assert (positions >= 0).all()
    return dict(direct=confusion(y[known], direct[known]), component=confusion(y[known], co[known]), pairwise=pairwise_counts(labels[positions], scope.visual_group_id), unknown=dict(n=int((~known).sum()), selected=int(direct[~known].sum()), connected=int(co[~known].sum())))

def train_select():
    with threadpool_limits(limits=4):
        OUT.mkdir(parents=True, exist_ok=True)
        feature = Features()
        candidates_frame = pd.read_parquet(OUT / 'candidates.parquet', columns=ENDS)
        (a, b) = feature.indices(candidates_frame)
        codes = codes_for(candidates_frame, feature.ids)
        assert len(codes) == len(np.unique(codes))
        (train_records, val_records) = (partition_records('train'), partition_records('validation'))
        (train, train_counts) = labeled_pairs(train_records, references('train'), 'train')
        (val, val_counts) = labeled_pairs(val_records, references('validation'), 'validation')
        construction = read(OUT / 'construction_budget.json')
        assert train_counts['added_positive'] == construction['train']['new_positive_pairs']
        assert val_counts['added_positive'] == construction['validation']['new_positive_pairs']
        assert train_counts['reference_pairs'] == 46425 and val_counts['reference_pairs'] == 164
        train.to_parquet(OUT / 'train_pairs.parquet', index=False)
        val.to_parquet(OUT / 'validation_pairs.parquet', index=False)
        (xtrain, xval) = (feature.transform(train), feature.transform(val))
        recalled = np.isin(codes_for(val, feature.ids), codes)
        y = val.visual_relation.eq('same').to_numpy()
        (bundles, selections) = ({}, {})
        for name in ['LR', 'HGB']:
            bundle = joblib.load(hgb_classifier / (name + '.joblib'))
            bundles[name + '_frozen'] = bundle
            selections[name + '_frozen'] = dict(threshold=float(bundle['threshold']), source=str(hgb_classifier / (name + '.joblib')))
            model = make_models()[name]
            print('fit', name, len(train), flush=True)
            model.fit(xtrain, train.relation.to_numpy())
            raw = raw_probability(model, xval)
            calibrator = IsotonicRegression(out_of_bounds='clip').fit(raw, y)
            score = calibrator.predict(raw)
            threshold = choose_threshold(y, score, recalled)
            bundle = dict(model=model, calibrator=calibrator, threshold=threshold, features=FEATURES, run_id=RUN_ID)
            joblib.dump(bundle, OUT / (name + '_retrained.joblib'))
            bundles[name + '_retrained'] = bundle
            selections[name + '_retrained'] = dict(threshold=threshold, classes=model.classes_.tolist(), effective_parameters={k: v for (k, v) in model.get_params().items() if isinstance(v, (str, int, float, bool, tuple, type(None)))}, validation_at_threshold=confusion(y, recalled & (score >= threshold)))
        for (name, threshold) in THRESHOLDS.items():
            selections[name] = dict(threshold=threshold, source='baseline_surfaces frozen original validation selection')
        score_candidates(feature, candidates_frame, bundles)
        (cells, validation) = ([], [])
        for name in SCORERS:
            raw = np.load(OUT / f'{name}_raw.npy', mmap_mode='r')
            score = np.load(OUT / f'{name}_score.npy', mmap_mode='r')
            assert np.isfinite(raw).all() and np.isfinite(score).all()
            positive = np.flatnonzero(score >= selections[name]['threshold'])
            ordered = edge_order(positive, score, a, b, raw[positive])
            np.save(OUT / f'{name}_ordered.npy', ordered)
            rows = []
            for cap in GRID:
                cell = build_cell(name, cap, a, b, ordered, len(feature.ids))
                cells.append(cell)
                row = dict(scorer=name, k=cap, **cell_metrics(cell, val, val_records, feature.ids, codes))
                rows.append(row)
                print('validation', name, cap, row['pairwise']['f1'], flush=True)
            selections[name]['K_double_star'] = select_k(rows)
            validation.extend(rows)
        frozen = dict(run_id=RUN_ID, all_scorers_frozen=True, frozen_at_utc=now(), scorers=selections, validation=validation, train_denominators=train_counts, validation_denominators=val_counts, validation_pairwise_scope='all validation records including COCO', k_grid=GRID, sklearn_version=sklearn.__version__)
        dump(OUT / 'selection_frozen.json', frozen)
        dump(REPORT / 'selection_frozen.json', frozen)
        requested = [c for c in cells if c['k'] in [8, 'inf', selections[c['scorer']]['K_double_star']]]
        dump(OUT / 'graphs_frozen.json', dict(run_id=RUN_ID, all_requested_frozen=True, frozen_at_utc=now(), cells=requested))

def evaluate():
    with threadpool_limits(limits=4):
        OUT.mkdir(parents=True, exist_ok=True)
        frozen = require_selection()
        cells = read(OUT / 'graphs_frozen.json')['cells']
        feature = Features()
        candidates_frame = pd.read_parquet(OUT / 'candidates.parquet', columns=ENDS)
        codes = codes_for(candidates_frame, feature.ids)
        records = partition_records('test')
        original = references('test')
        known = original[original.visual_relation_mask].copy()
        assert len(known) == 9494 and known.visual_relation.eq('same').sum() == 8701
        (test, counts) = labeled_pairs(records, original, 'test')
        assert counts['added_positive'] == read(OUT / 'construction_budget.json')['test']['new_positive_pairs']
        assert counts['reference_pairs'] == 159
        test.to_parquet(OUT / 'test_pairs.parquet', index=False)
        metrics = {}
        for cell in cells:
            key = (cell['scorer'], cell['k'])
            metrics[key] = dict(**cell_metrics(cell, test, records, feature.ids, codes, sop_only=True), original_known=cell_metrics(cell, original, records, feature.ids, codes, sop_only=True), structure=cell['structure'])
        rows = []
        for name in SCORERS:
            cap = frozen['scorers'][name]['K_double_star']
            for (role, k) in [('inf', 'inf'), ('K8', 8), ('K_double_star', cap)]:
                rows.append(dict(scorer=name, role=role, k=k, **metrics[name, k]))
        result = dict(run_id=RUN_ID, sklearn_version=sklearn.__version__, features=FEATURES, train_denominators=frozen['train_denominators'], validation_denominators=frozen['validation_denominators'], test_denominators=counts, test_pairwise_scope='SOP records only; full unordered pair universe', selections=frozen['scorers'], rows=rows, evaluated_at_utc=now())
        dump(OUT / 'results.json', result)

def main():
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    parser = argparse.ArgumentParser(description=__doc__)
    phases = {'inputs': inputs, 'captions': captions, 'encode': encode, 'retrieval': candidates, 'train-select': train_select, 'evaluate': evaluate}
    parser.add_argument('--phase', required=True, choices=list(phases))
    args = parser.parse_args()
    phases[args.phase]()
if __name__ == '__main__':
    main()
