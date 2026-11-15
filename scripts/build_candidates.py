from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import itertools
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable
PHASH_BITS = 64
DUPLICATE_RELATIONS = {'duplicate'}
CANDIDATE_COLUMNS = ['edge_id', 'record_id_a', 'record_id_b', 'idx_a', 'idx_b', 'channels', 'channel_count', 'clip_image_score', 'clip_text_score', 'dinov2_score', 'phash_distance', 'sscd_score', 'text_exact_match']

def canonical_edge_id(record_id_a: object, record_id_b: object) -> str:
    (a, b) = (str(record_id_a), str(record_id_b))
    (lo, hi) = (a, b) if a <= b else (b, a)
    return f'{lo}__{hi}'

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=repo_root() / 'data' / 'manifests' / 'vl_dedupbench_50k_manifest.parquet')
    parser.add_argument('--pairs', type=Path, default=repo_root() / 'data' / 'manifests' / 'vl_dedupbench_50k_pairs_eval.parquet')
    parser.add_argument('--features-root', type=Path, default=repo_root() / 'features')
    parser.add_argument('--out-dir', type=Path, default=repo_root() / 'outputs' / 'candidates')
    parser.add_argument('--tables-dir', type=Path, default=repo_root() / 'outputs' / 'tables')
    parser.add_argument('--limit', type=int, default=None, help='Optional manifest row limit for smoke tests.')
    parser.add_argument('--phash-threshold', type=int, default=2)
    parser.add_argument('--sscd-topk', type=int, default=30)
    parser.add_argument('--dinov2-topk', type=int, default=30)
    parser.add_argument('--clip-image-topk', type=int, default=30)
    parser.add_argument('--clip-text-topk', type=int, default=10)
    parser.add_argument('--text-exact-neighbors', type=int, default=20, help='Maximum same-normalized-text neighbors per record. Use 0 for full cliques.')
    parser.add_argument('--channels', type=str, default='phash,sscd,dinov2,clip_image,clip_text,text_exact', help='Comma-separated channels to run.')
    parser.add_argument('--cpu-faiss', action='store_true', help='Force CPU FAISS even if GPU FAISS is available.')
    return parser.parse_args()

def selected_channels(raw: str) -> list[str]:
    valid = {'phash', 'sscd', 'dinov2', 'clip_image', 'clip_text', 'text_exact'}
    channels = [item.strip().lower() for item in raw.split(',') if item.strip()]
    unknown = sorted(set(channels) - valid)
    if unknown:
        raise ValueError(f'Unknown channels: {unknown}. Valid channels: {sorted(valid)}')
    return channels

def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', str(text)).lower()
    normalized = re.sub('[^\\w\\s]+', ' ', normalized)
    normalized = re.sub('\\s+', ' ', normalized).strip()
    return normalized

def edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)

def pair_id_from_indices(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)

def hamming_masks(radius: int) -> list[int]:
    masks = [0]
    bit_positions = range(PHASH_BITS)
    for r in range(1, radius + 1):
        for combo in itertools.combinations(bit_positions, r):
            mask = 0
            for bit in combo:
                mask |= 1 << bit
            masks.append(mask)
    return masks

def require_partition_ids(manifest: pd.DataFrame) -> np.ndarray:
    if 'partition' not in manifest:
        raise ValueError("manifest is missing the 'partition' column; candidate generation cannot enforce split isolation without it")
    return manifest['partition'].astype(str).to_numpy()

def update_edge(edges: dict[tuple[int, int], dict[str, object]], i: int, j: int, channel: str, values: dict[str, object], partition_ids: np.ndarray) -> None:
    if i == j:
        return
    if partition_ids[i] != partition_ids[j]:
        return
    key = pair_id_from_indices(i, j)
    row = edges.setdefault(key, {'idx_a': key[0], 'idx_b': key[1], 'channels': set()})
    row['channels'].add(channel)
    for (name, value) in values.items():
        reference = row.get(name)
        if reference is None:
            row[name] = value
        elif name.endswith('_distance'):
            row[name] = min(reference, value)
        else:
            row[name] = max(reference, value)

def build_phash_edges(manifest: pd.DataFrame, features_root: Path, threshold: int, edges: dict[tuple[int, int], dict[str, object]]) -> int:
    phash_path = features_root / 'phash' / 'phash.parquet'
    if not phash_path.exists():
        raise FileNotFoundError(f'Missing pHash features: {phash_path}')
    phashes = pd.read_parquet(phash_path)
    id_to_idx = {record_id: idx for (idx, record_id) in enumerate(manifest['record_id'])}
    partition_ids = require_partition_ids(manifest)
    valid = phashes.dropna(subset=['phash_int']).copy()
    buckets: dict[int, list[int]] = defaultdict(list)
    for (record_id, phash_int) in valid[['record_id', 'phash_int']].itertuples(index=False):
        idx = id_to_idx.get(record_id)
        if idx is not None:
            buckets[int(phash_int)].append(idx)
    before = len(edges)
    masks = hamming_masks(threshold)
    for (phash_int, indices) in tqdm(buckets.items(), desc='pHash candidates'):
        for (i, j) in itertools.combinations(sorted(indices), 2):
            update_edge(edges, i, j, 'phash', {'phash_distance': 0}, partition_ids)
        for mask in masks[1:]:
            other_hash = phash_int ^ mask
            if phash_int >= other_hash or other_hash not in buckets:
                continue
            distance = mask.bit_count()
            for i in indices:
                for j in buckets[other_hash]:
                    update_edge(edges, i, j, 'phash', {'phash_distance': distance}, partition_ids)
    return len(edges) - before

def faiss_resources(cpu_faiss: bool) -> object | None:
    if cpu_faiss:
        return None
    if not hasattr(faiss, 'get_num_gpus') or faiss.get_num_gpus() <= 0:
        return None
    return faiss.StandardGpuResources()

def make_index(dim: int, resources: object | None) -> faiss.Index:
    cpu_index = faiss.IndexFlatIP(dim)
    if resources is None:
        return cpu_index
    return faiss.index_cpu_to_gpu(resources, 0, cpu_index)

def load_embeddings(path: Path, expected_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f'Missing embedding file: {path}')
    embeddings = np.load(path).astype('float32', copy=False)
    if embeddings.shape[0] != expected_rows:
        raise ValueError(f'Embedding row count mismatch for {path}: {embeddings.shape[0]} embeddings vs {expected_rows} manifest rows')
    faiss.normalize_L2(embeddings)
    return embeddings

def build_embedding_edges(embeddings: np.ndarray, channel: str, score_column: str, topk: int, resources: object | None, edges: dict[tuple[int, int], dict[str, object]], partition_ids: np.ndarray) -> int:
    before = len(edges)
    index = make_index(embeddings.shape[1], resources)
    index.add(embeddings)
    (scores, indices) = index.search(embeddings, topk + 1)
    for i in tqdm(range(embeddings.shape[0]), desc=f'{channel} candidates'):
        for (score, j) in zip(scores[i], indices[i]):
            j = int(j)
            if j < 0 or j == i:
                continue
            update_edge(edges, i, j, channel, {score_column: float(score)}, partition_ids)
    return len(edges) - before

def build_text_exact_edges(manifest: pd.DataFrame, max_neighbors: int, edges: dict[tuple[int, int], dict[str, object]]) -> int:
    before = len(edges)
    partition_ids = require_partition_ids(manifest)
    groups: dict[str, list[int]] = defaultdict(list)
    for (idx, text) in enumerate(manifest['text'].astype(str)):
        key = normalize_text(text)
        if key:
            groups[key].append(idx)
    for (key, indices) in tqdm(groups.items(), desc='text_exact candidates'):
        if len(indices) < 2:
            continue
        sorted_indices = sorted(indices)
        if max_neighbors <= 0:
            pairs = itertools.combinations(sorted_indices, 2)
        else:
            pairs = ((i, sorted_indices[j_pos]) for (i_pos, i) in enumerate(sorted_indices) for j_pos in range(i_pos + 1, min(len(sorted_indices), i_pos + max_neighbors + 1)))
        for (i, j) in pairs:
            update_edge(edges, i, j, 'text_exact', {'text_exact_match': True}, partition_ids)
    return len(edges) - before

def edges_to_frame(edges: dict[tuple[int, int], dict[str, object]], manifest: pd.DataFrame) -> pd.DataFrame:
    if not edges:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    rows = []
    record_ids = manifest['record_id'].to_numpy()
    for row in edges.values():
        channels = row['channels']
        out = {key: value for (key, value) in row.items() if key != 'channels'}
        out['record_id_a'] = record_ids[int(row['idx_a'])]
        out['record_id_b'] = record_ids[int(row['idx_b'])]
        out['edge_id'] = canonical_edge_id(out['record_id_a'], out['record_id_b'])
        out['channels'] = ';'.join(sorted(channels))
        out['channel_count'] = len(channels)
        rows.append(out)
    frame = pd.DataFrame(rows)
    for column in CANDIDATE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    first_cols = ['edge_id', 'record_id_a', 'record_id_b', 'idx_a', 'idx_b', 'channels', 'channel_count']
    other_cols = [col for col in frame.columns if col not in first_cols]
    return frame[first_cols + sorted(other_cols)].sort_values(['record_id_a', 'record_id_b'])

def channel_stats(candidates: pd.DataFrame, channel_new_edges: dict[str, int]) -> pd.DataFrame:
    channel_sets = candidates['channels'].str.split(';').apply(set) if len(candidates) else pd.Series(dtype=object)
    rows = []
    for (channel, added_edges) in channel_new_edges.items():
        count = int(channel_sets.apply(lambda values: channel in values).sum()) if len(candidates) else 0
        rows.append({'channel': channel, 'candidate_edges': count, 'new_edges_after_merge': int(added_edges)})
    return pd.DataFrame(rows)

def candidate_recall(pairs: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    candidate_keys = {edge_key(a, b) for (a, b) in candidates[['record_id_a', 'record_id_b']].itertuples(index=False)}
    eval_pairs = pairs.copy()
    eval_pairs['edge_key'] = [edge_key(a, b) for (a, b) in eval_pairs[['record_id_a', 'record_id_b']].itertuples(index=False)]
    eval_pairs['covered'] = eval_pairs['edge_key'].isin(candidate_keys)
    rows = []
    groupings = [('overall', None), ('relation', 'relation'), ('grade', 'grade'), ('partition', 'partition'), ('split', 'pair_type')]
    for (group_name, column) in groupings:
        if column is None:
            slices = [('all', eval_pairs)]
        else:
            slices = list(eval_pairs.groupby(column, dropna=False))
        for (value, group) in slices:
            duplicate_group = group[group['relation'].isin(DUPLICATE_RELATIONS)]
            rows.append({'grouping': group_name, 'value': str(value), 'pairs': int(len(group)), 'covered_pairs': int(group['covered'].sum()), 'pair_recall': float(group['covered'].mean()) if len(group) else 0.0, 'duplicate_pairs': int(len(duplicate_group)), 'covered_duplicate_pairs': int(duplicate_group['covered'].sum()), 'duplicate_recall': float(duplicate_group['covered'].mean()) if len(duplicate_group) else 0.0})
    return pd.DataFrame(rows)

def main() -> None:
    args = parse_args()
    channels = selected_channels(args.channels)
    manifest = pd.read_parquet(args.manifest)
    pairs = pd.read_parquet(args.pairs)
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()
        kept_ids = set(manifest['record_id'])
        pairs = pairs[pairs['record_id_a'].isin(kept_ids) & pairs['record_id_b'].isin(kept_ids)].copy()
    features_root = args.features_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    edges: dict[tuple[int, int], dict[str, object]] = {}
    channel_new_edges: dict[str, int] = {}
    resources = faiss_resources(args.cpu_faiss)
    partition_ids = require_partition_ids(manifest)
    print(f'records: {len(manifest):,}')
    print(f"channels: {', '.join(channels)}")
    print(f"faiss: {('gpu' if resources is not None else 'cpu')}")
    if 'phash' in channels:
        channel_new_edges['phash'] = build_phash_edges(manifest, features_root, args.phash_threshold, edges)
    if 'sscd' in channels:
        embeddings = load_embeddings(features_root / 'sscd' / 'image_emb.npy', len(manifest))
        channel_new_edges['sscd'] = build_embedding_edges(embeddings, 'sscd', 'sscd_score', args.sscd_topk, resources, edges, partition_ids)
    if 'dinov2' in channels:
        embeddings = load_embeddings(features_root / 'dinov2' / 'image_emb.npy', len(manifest))
        channel_new_edges['dinov2'] = build_embedding_edges(embeddings, 'dinov2', 'dinov2_score', args.dinov2_topk, resources, edges, partition_ids)
    if 'clip_image' in channels:
        embeddings = load_embeddings(features_root / 'clip' / 'image_emb.npy', len(manifest))
        channel_new_edges['clip_image'] = build_embedding_edges(embeddings, 'clip_image', 'clip_image_score', args.clip_image_topk, resources, edges, partition_ids)
    if 'clip_text' in channels:
        embeddings = load_embeddings(features_root / 'clip' / 'text_emb.npy', len(manifest))
        channel_new_edges['clip_text'] = build_embedding_edges(embeddings, 'clip_text', 'clip_text_score', args.clip_text_topk, resources, edges, partition_ids)
    if 'text_exact' in channels:
        channel_new_edges['text_exact'] = build_text_exact_edges(manifest, args.text_exact_neighbors, edges)
    candidates = edges_to_frame(edges, manifest)
    candidates.to_parquet(args.out_dir / 'candidate_edges.parquet', index=False)
    stats = channel_stats(candidates, channel_new_edges)
    stats.loc[len(stats)] = {'channel': 'all', 'candidate_edges': int(len(candidates)), 'new_edges_after_merge': int(len(candidates))}
    stats.to_csv(args.tables_dir / 'candidate_channel_stats.csv', index=False)
    recall = candidate_recall(pairs, candidates)
    recall.to_csv(args.tables_dir / 'candidate_recall.csv', index=False)
    duplicate_recall = recall[recall['grouping'].eq('overall') & recall['value'].eq('all')]['duplicate_recall'].iloc[0]
    print(f'candidate_edges: {len(candidates):,}')
    print(f'overall duplicate recall: {duplicate_recall:.4f}')
    print(f"wrote: {args.out_dir / 'candidate_edges.parquet'}")
if __name__ == '__main__':
    main()
