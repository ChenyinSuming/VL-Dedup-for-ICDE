import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    from scripts import build_candidates as candidate
except ModuleNotFoundError:
    import build_candidates as candidate
FEATURES = ['clip_image', 'clip_text', 'phash_similarity', 'raw_text_equal', 'normalized_text_equal', 'clipscore_min', 'clipscore_mean', 'clipscore_difference', 'cross_clip_min', 'cross_clip_mean', 'same_image_asset']
RELATIONS = ['duplicate', 'hard_negative', 'keep_augment_candidate', 'mismatch', 'text_duplicate_only']

def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + chr(10))

def write_rows(path, rows):
    with Path(path).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + chr(10))

def public_frame(root):
    rows = read_rows(root / 'public_records.jsonl')
    if any((len(r['images']) != 1 or not r.get('text') for r in rows)):
        raise ValueError('This controlled revision requires complete single-image records')
    return pd.DataFrame([{'record_id': r['record_id'], 'text': r['text'], 'image_path': r['images'][0]['image_path']} for r in rows]).sort_values('record_id').reset_index(drop=True)

def prepare_features(args):
    import torch
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    start = time.perf_counter()
    frame = public_frame(args.root)
    lineage = pd.DataFrame(read_rows(args.root / 'private/lineage.jsonl')).set_index('record_id')
    initial_ids = lineage.loc[frame.record_id, 'original_record_id'].tolist()
    meta = pd.read_parquet(args.initial_features / 'clip/metadata.parquet').set_index('record_id')
    indices = meta.loc[initial_ids, 'row_index'].to_numpy(dtype=int)
    if not meta.loc[initial_ids, 'read_ok'].all():
        raise ValueError('Image cache contains failed images')
    info = json.loads((args.initial_features / 'clip/feature_info.json').read_text())
    if info['model_name'] != 'openai/clip-vit-large-patch14-336':
        raise ValueError('Unexpected image encoder cache')
    source = Path(json.loads((args.root / 'contract.json').read_text())['source_root'])
    for row in frame.itertuples():
        if Path(row.image_path).resolve() != (source / lineage.loc[row.record_id, 'original_image_path']).resolve():
            raise ValueError('Image asset changed: ' + row.record_id)
    out = args.output / 'features'
    (out / 'clip').mkdir(parents=True, exist_ok=True)
    (out / 'phash').mkdir(exist_ok=True)
    image = np.load(args.initial_features / 'clip/image_emb.npy', mmap_mode='r')[indices].copy()
    assert image.shape == (len(frame), 768) and np.isfinite(image).all()
    tokenizer = AutoTokenizer.from_pretrained(str(args.clip_model), local_files_only=True)
    model = CLIPTextModelWithProjection.from_pretrained(str(args.clip_model), local_files_only=True).to('cuda').eval()
    batches = []
    truncated = 0
    with torch.inference_mode():
        for i in range(0, len(frame), args.batch_size):
            texts = frame.text.iloc[i:i + args.batch_size].tolist()
            lengths = tokenizer(texts, truncation=False, padding=False)['input_ids']
            truncated += sum((len(x) > 77 for x in lengths))
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors='pt').to('cuda')
            value = model(**encoded).text_embeds.float()
            value = torch.nn.functional.normalize(value, dim=1)
            batches.append(value.cpu().numpy())
            if i % (args.batch_size * 25) == 0:
                print('text_features', i, '/', len(frame), flush=True)
    text = np.concatenate(batches).astype('float32')
    np.save(out / 'clip/image_emb.npy', image)
    np.save(out / 'clip/text_emb.npy', text)
    np.save(out / 'clip/clipscore.npy', np.sum(image * text, axis=1))
    pd.DataFrame({'record_id': frame.record_id, 'row_index': np.arange(len(frame))}).to_parquet(out / 'clip/metadata.parquet', index=False)
    phash = pd.read_parquet(args.initial_features / 'phash/phash.parquet').set_index('record_id').loc[initial_ids].reset_index(drop=True)
    assert phash.read_ok.all() and phash.phash_int.notna().all()
    phash['record_id'] = frame.record_id.to_numpy()
    phash['image_path'] = frame.image_path.to_numpy()
    phash['row_index'] = np.arange(len(frame))
    phash.to_parquet(out / 'phash/phash.parquet', index=False)
    frame.to_parquet(args.output / 'public_manifest.parquet', index=False)
    receipt = {'records': len(frame), 'reused_image_cache': str(args.initial_features), 'model': info['model_name'], 'revision': 'ce19dc912ca5cd21c8a653c79e251e808ccabcd1', 'text_features_recomputed': len(text), 'clipscore_recomputed': len(text), 'token_truncations': truncated, 'lineage_usage': 'offline cache row alignment and unchanged-image verification only', 'model_input': 'revised public text only; max_length=77; float32'}
    print(json.dumps(receipt), flush=True)

def build_candidates(args):
    start = time.perf_counter()
    frame = public_frame(args.root)
    candidate.faiss.omp_set_num_threads(8)
    (edges, counts) = ({}, {})
    counts['phash'] = candidate.build_phash_edges(frame, args.output / 'features', 2, edges)
    for (name, filename, k) in [('clip_image', 'image_emb.npy', 30), ('clip_text', 'text_emb.npy', 10)]:
        embedding = candidate.load_embeddings(args.output / 'features/clip' / filename, len(frame))
        counts[name] = candidate.build_embedding_edges(embedding, name, name + '_score', k, None, edges)
    counts['text_exact'] = candidate.build_text_exact_edges(frame, 20, edges)
    result = candidate.edges_to_frame(edges, frame)
    result.to_parquet(args.output / 'candidates.parquet', index=False)
    dump(args.output / 'stage_b_receipt.json', {'records': len(frame), 'candidates': len(result), 'channel_new_edges': counts, 'partition_filter': False, 'label_access': False, 'cpu_threads': 8, 'config': {'phash_radius': 2, 'clip_image_topk': 30, 'clip_text_topk': 10, 'text_exact_neighbors': 20}})
    print('candidates', len(result), flush=True)

class ObservableFeatures:

    def __init__(self, frame, image, text, phashes):
        self.ids = dict(zip(frame.record_id, range(len(frame))))
        (self.image, self.text) = (image, text)
        self.phashes = np.asarray(phashes, dtype=np.uint64)
        self.raw = frame.text.to_numpy()
        self.normalized = frame.text.map(candidate.normalize_text).to_numpy()
        self.assets = frame.image_path.map(lambda p: str(Path(p).resolve())).to_numpy()
        self.scores = np.sum(image * text, axis=1)
        self.bits = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)

    def transform(self, pairs):
        a = pairs.record_id_a.map(self.ids).to_numpy(dtype=int)
        b = pairs.record_id_b.map(self.ids).to_numpy(dtype=int)
        out = np.empty((len(a), len(FEATURES)), dtype=np.float32)
        for start in range(0, len(a), 8192):
            (x, y) = (a[start:start + 8192], b[start:start + 8192])
            (ca, cb) = (self.scores[x], self.scores[y])
            cross_a = np.sum(self.image[x] * self.text[y], axis=1)
            cross_b = np.sum(self.image[y] * self.text[x], axis=1)
            xor = np.bitwise_xor(self.phashes[x], self.phashes[y])
            distance = self.bits[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1) / 64
            out[start:start + len(x)] = np.column_stack([np.sum(self.image[x] * self.image[y], axis=1), np.sum(self.text[x] * self.text[y], axis=1), 1 - distance, self.raw[x] == self.raw[y], self.normalized[x] == self.normalized[y], np.minimum(ca, cb), (ca + cb) / 2, np.abs(ca - cb), np.minimum(cross_a, cross_b), (cross_a + cross_b) / 2, self.assets[x] == self.assets[y]])
        if not np.isfinite(out).all():
            raise ValueError('Non-finite observable features')
        return out

def exact_actions(frame):
    (representatives, actions) = ({}, [])
    for row in frame.sort_values('record_id').itertuples():
        key = (str(Path(row.image_path).resolve()), row.text)
        target = representatives.get(key) if row.text else None
        actions.append({'record_id': row.record_id, 'action': 'delete' if target else 'keep', 'replacement_record_id': target, 'reason': 'identical_image_asset_and_original_text' if target else 'retained', 'evidence_origin': 'observable_exact_rule', 'human_support_gold': False})
        if target is None:
            representatives[key] = row.record_id
    return pd.DataFrame(actions)

def binary_metrics(y, pred):
    tp = int(np.sum(y & pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    return {'n': len(y), 'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / max(1, tp + fp), 'recall': tp / max(1, tp + fn), 'f1': 2 * tp / max(1, 2 * tp + fp + fn)}

def choose_threshold(y, score, recalled=None):
    if recalled is None:
        recalled = np.ones(len(y), dtype=bool)
    eligible = []
    for threshold in np.unique(np.r_[np.linspace(0, 1, 201), score]):
        metrics = binary_metrics(y, recalled & (score >= threshold))
        if metrics['tp'] + metrics['fp'] >= 20 and metrics['precision'] >= 0.99:
            eligible.append((metrics['recall'], float(threshold), metrics))
    return max(eligible, default=(0.0, 1.000001, {}), key=lambda x: (x[0], x[1]))[1]

def graph_components(frame, pairs):
    parent = {rid: rid for rid in frame.record_id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (a, b) in pairs[['record_id_a', 'record_id_b']].itertuples(index=False, name=None):
        (a, b) = (find(a), find(b))
        if a != b:
            parent[max(a, b)] = min(a, b)
    return {rid: find(rid) for rid in parent}

def fit_and_evaluate(args):
    if getattr(args, 'target', None) != 'construction_analysis':
        raise ValueError('Five-class fitting is construction_analysis only; use train_visual_classifier.py for visual scoring')
    start = time.perf_counter()
    frame = public_frame(args.root)
    candidates = pd.read_parquet(args.output / 'candidates.parquet')
    pairs = pd.DataFrame(read_rows(args.root / 'sealed/relation_applicability.jsonl'))
    features = ObservableFeatures(frame, np.load(args.output / 'features/clip/image_emb.npy'), np.load(args.output / 'features/clip/text_emb.npy'), pd.read_parquet(args.output / 'features/phash/phash.parquet').phash_int)
    x = features.transform(pairs)
    candidate_x = features.transform(candidates)
    np.save(args.output / 'pair_features.npy', x)
    np.save(args.output / 'candidate_features.npy', candidate_x)
    keys = set(candidates.edge_id)
    recalled = np.array([candidate.canonical_edge_id(a, b) in keys for (a, b) in pairs[['record_id_a', 'record_id_b']].itertuples(index=False, name=None)])
    y = pairs.relation.to_numpy()
    train = (pairs.partition == 'train').to_numpy() & pairs.graded_relation_mask.to_numpy()
    val = (pairs.partition == 'validation').to_numpy() & pairs.graded_relation_mask.to_numpy()
    test = (pairs.partition == 'test').to_numpy() & pairs.graded_relation_mask.to_numpy()
    classifiers = {'logistic': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1, random_state=23)), 'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, l2_regularization=1, early_stopping=False, random_state=23), 'mlp': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=160, early_stopping=False, random_state=23))}
    (summary, candidate_probs) = ({}, {})
    test_predictions = pairs.loc[test].copy()
    test_predictions['candidate_recalled'] = recalled[test]
    visual = pairs.visual_relation_mask.to_numpy()
    vy = (pairs.visual_relation == 'same').to_numpy()
    for (name, model) in classifiers.items():
        t = time.perf_counter()
        print('fit', name, int(train.sum()), flush=True)
        model.fit(x[train], y[train])
        prob = model.predict_proba(x)
        pred = model.classes_[prob.argmax(axis=1)]
        vcolumns = [list(model.classes_).index(k) for k in ('duplicate', 'keep_augment_candidate')]
        raw = prob[:, vcolumns].sum(axis=1)
        calibrator = IsotonicRegression(out_of_bounds='clip').fit(raw[val & visual], vy[val & visual])
        calibrated = calibrator.predict(raw)
        threshold = choose_threshold(vy[val & visual], calibrated[val & visual], recalled[val & visual])
        joblib.dump({'model': model, 'calibrator': calibrator, 'threshold': threshold, 'features': FEATURES}, args.output / (name + '.joblib'))
        predict_start = time.perf_counter()
        cprob = np.concatenate([model.predict_proba(candidate_x[i:i + 50000]) for i in range(0, len(candidate_x), 50000)])
        cscores = calibrator.predict(cprob[:, vcolumns].sum(axis=1))
        candidate_probs[name] = cscores
        selected = cscores >= threshold
        edges = candidates.loc[selected, ['edge_id', 'record_id_a', 'record_id_b']].copy()
        edges['visual_probability'] = cscores[selected]
        edges.to_parquet(args.output / (name + '_stage_d_edges.parquet'), index=False)
        components = graph_components(frame, edges)
        pd.DataFrame({'record_id': list(components), 'component': list(components.values())}).to_parquet(args.output / (name + '_components.parquet'), index=False)
        graph_pred = np.array([components[a] == components[b] for (a, b) in pairs[['record_id_a', 'record_id_b']].itertuples(index=False, name=None)])
        tv = test & visual
        full_pred = np.where(recalled, pred, 'unretrieved')
        binary_pred = recalled & (calibrated >= threshold)
        test_predictions[name + '_relation'] = pred[test]
        test_predictions[name + '_visual_probability'] = calibrated[test]
        test_predictions[name + '_direct_selected'] = binary_pred[test]
        test_predictions[name + '_co_membership'] = graph_pred[test]
        summary[name] = {'test_pairs': int(test.sum()), 'train_pairs': int(train.sum()), 'validation_pairs': int(val.sum()), 'relation_accuracy_fixed_pairs': accuracy_score(y[test], pred[test]), 'relation_macro_f1_fixed_pairs': f1_score(y[test], pred[test], labels=RELATIONS, average='macro', zero_division=0), 'relation_macro_f1_retrieval_pipeline': f1_score(y[test], full_pred[test], labels=RELATIONS, average='macro', zero_division=0), 'relation_accuracy_retrieval_pipeline': accuracy_score(y[test], full_pred[test]), 'per_relation': classification_report(y[test], pred[test], labels=RELATIONS, output_dict=True, zero_division=0), 'log_loss': log_loss(y[test], prob[test], labels=model.classes_), 'visual_threshold': threshold, 'visual_full_test_direct': binary_metrics(vy[tv], binary_pred[tv]), 'validation_visual_direct': binary_metrics(vy[val & visual], recalled[val & visual] & (calibrated[val & visual] >= threshold)), 'visual_full_test_co_membership': binary_metrics(vy[tv], graph_pred[tv]), 'selected_global_edges': len(edges), 'global_components': len(set(components.values())), 'test_relation_risk_rows': {k: {'n': int(np.sum(test & (y == k))), 'selected': int(np.sum(test & (y == k) & binary_pred)), 'co_membership': int(np.sum(test & (y == k) & graph_pred))} for k in RELATIONS}}
        print(name, json.dumps({k: v for (k, v) in summary[name].items() if k not in ('per_relation', 'test_relation_risk_rows')}), flush=True)
        dump(args.output / 'baseline_results.json', summary)
    test_predictions.to_parquet(args.output / 'full_test_predictions.parquet', index=False)
    scores = candidates[['edge_id', 'record_id_a', 'record_id_b']].copy()
    for (name, score) in candidate_probs.items():
        scores[name + '_visual_probability'] = score
    scores.to_parquet(args.output / 'candidate_predictions.parquet', index=False)
    actions = exact_actions(frame)
    survivors = set(actions.loc[actions.action == 'keep', 'record_id'])
    assert actions.record_id.nunique() == len(frame) and all((t in survivors for t in actions.replacement_record_id.dropna()))
    actions.to_parquet(args.output / 'record_actions.parquet', index=False)
    public = read_rows(args.root / 'public_records.jsonl')
    write_rows(args.output / 'retained_records.jsonl', (r for r in public if r['record_id'] in survivors))
    asset_components = frame.image_path.map(lambda p: str(Path(p).resolve())).to_numpy()
    asset_by_id = dict(zip(frame.record_id, asset_components))
    exact_visual = np.array([asset_by_id[a] == asset_by_id[b] for (a, b) in pairs[['record_id_a', 'record_id_b']].itertuples(index=False, name=None)])
    receipt = {'records': len(frame), 'retained': len(survivors), 'deleted': len(frame) - len(survivors), 'full_test_pairs': int(test.sum()), 'recalled_test_pairs': int(np.sum(test & recalled)), 'unretrieved_test_pairs': int(np.sum(test & ~recalled)), 'visual_test_denominator': int(np.sum(test & visual)), 'candidate_edges': len(candidates), 'replacement_targets_survive': True, 'exact_asset_visual_baseline': binary_metrics(vy[test & visual], exact_visual[test & visual]), 'record_baselines': {'keep_all': len(frame), 'exact_asset_and_text': len(survivors)}, 'information_preservation_rule': 'each deleted record has identical image asset and exact text in a surviving record'}
    print(json.dumps(receipt), flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-features', type=Path, required=True)
    p.add_argument('--clip-model', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--target', choices=['masked_visual_identity_same_vs_different', 'construction_analysis'], default='masked_visual_identity_same_vs_different')
    p.add_argument('--stage', choices=['features', 'candidates', 'fit', 'all'], default='all')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for (name, fn) in [('features', prepare_features), ('candidates', build_candidates), ('fit', fit_and_evaluate)]:
        if args.stage in (name, 'all'):
            fn(args)
if __name__ == '__main__':
    main()
