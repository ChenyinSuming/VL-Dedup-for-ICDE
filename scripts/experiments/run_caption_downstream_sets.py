#!/usr/bin/env python3
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
from scripts.experiments.support import export_cluster
"""CAPTION_DOWNSTREAM_RUN CPU-only exports and downstream training-set construction.

export-b -> semdedup -> finalize (requires coverage_complete.json); test creates
fixed Karpathy test.parquet. All writes stay under --run; historical inputs read
only. No new hashes. Cleaning functions receive observable IDs/scores/vectors,
not caption provenance; validation truth and final composition are separate.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import itertools
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = RESOURCES
RUN = RES / 'runs/caption_downstream'
NOISY_POOL_RUN = RES / 'runs/caption_pilot'
MISMATCH_SCORES_RUN = RES / 'runs/mismatch_scores'
STATES = ('retained', 'pending', 'deleted')


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def writel(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for value in values:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
    temporary.replace(path)


def require_complete(run, marker):
    path = run / marker
    if not path.exists():
        raise RuntimeError(f'Required completion marker missing: {path}')
    receipt = load(path)
    if receipt.get('status') not in ('complete', 'completed'):
        raise RuntimeError(f'Completion marker is not complete: {path}')


def unique_index(values, key):
    result = {r[key]: r for r in values}
    if len(result) != len(values):
        raise ValueError(f'Duplicate {key}')
    return result


def stage_inputs(run):
    require_complete(run, 'stage_e_complete.json')
    records = rows(run / 'records.jsonl')
    record_map = unique_index(records, 'record_id')
    scores = unique_index(rows(run / 'record_scores.jsonl'), 'record_id')
    if set(scores) != set(record_map):
        raise ValueError('Record scoring coverage differs from frozen records')
    pairs = unique_index(rows(run / 'train_pairs.jsonl'), 'pair_id')
    verdicts = unique_index(rows(run / 'pair_verdicts.jsonl'), 'pair_id')
    if set(pairs) != set(verdicts):
        raise ValueError('Pair verdict coverage differs from frozen pairs')
    groups = defaultdict(list)
    for row in records:
        groups[(row['partition'], row['component'])].append(row['record_id'])
    expected = {tuple(sorted(pair)) for (part, _), ids in groups.items() if part == 'train'
                for pair in itertools.combinations(ids, 2)}
    actual = {tuple(sorted((p['record_id_a'], p['record_id_b']))) for p in pairs.values()}
    if actual != expected or len(actual) != len(pairs):
        raise ValueError('Train pairs differ from complete within-component pair set')
    return records, record_map, scores, pairs, verdicts, groups


def frozen_fusion(records, probabilities):
    from scripts import run_combined_actions as combined_actions
    reference = [r['record_id'] for r in rows(MISMATCH_SCORES_RUN / 'records.jsonl') if r['partition'] == 'validation']
    if len(reference) != 6132 or len(set(reference)) != 6132:
        raise ValueError('Frozen MISMATCH_SCORES_RUN validation reference must have 6132 records')
    cpu = load(MISMATCH_SCORES_RUN / 'cpu_scores.json')
    reference_scores = unique_index(rows(MISMATCH_SCORES_RUN / 'qwen25_vllm.jsonl'), 'record_id')
    # Synthetic namespace keeps CAPTION_DOWNSTREAM_RUN IDs independent of the historical reference.
    clips = {f'ref:{rid}': cpu[rid]['clipscore'] for rid in reference}
    probs = {f'ref:{rid}': reference_scores[rid] for rid in reference}
    for r in records:
        rid = r['record_id']
        clips[f'new:{rid}'] = r['clipscore']
        p = probabilities[rid]
        probs[f'new:{rid}'] = p.get('yes_probability', p)
    scores = combined_actions.combined_scores(clips, probs, [f'ref:{r}' for r in reference])
    return {r['record_id']: scores[f"new:{r['record_id']}"] for r in records}


def export_b(run):
    if (run / 'b_export_complete.json').exists():
        require_complete(run, 'b_export_complete.json')
        return
    import pandas as pd
    from scripts import run_action_checks as checks
    from scripts import run_combined_actions as combined_actions
    records, record_map, probabilities, pairs, verdicts, groups = stage_inputs(run)
    fusion = frozen_fusion(records, probabilities)
    truth = pd.read_parquet(NOISY_POOL_RUN / 'b/validation/corpus.parquet').set_index('record_id')
    validation = sorted(r['record_id'] for r in records if r['partition'] == 'validation')
    if set(validation) != set(truth.index):
        raise ValueError('Validation record identity mismatch')
    selected = checks.select_threshold([fusion[r] for r in validation],
                                      [bool(truth.loc[r, 'caption_swapped']) for r in validation])
    selected.update(reference='MISMATCH_SCORES_RUN original validation6132', validation_records=len(validation), status='frozen')
    threshold_path = run / 'threshold_frozen.json'
    if threshold_path.exists() and load(threshold_path) != selected:
        raise ValueError('Existing frozen threshold differs')
    dump(threshold_path, selected)  # freeze before train exports
    decisions = {tuple(sorted((p['record_id_a'], p['record_id_b']))): verdicts[pid]['parsed']['decision']
                 for pid, p in pairs.items()}
    exports = []
    for (part, component), ids in sorted(groups.items()):
        if part != 'train':
            continue
        ids = sorted(ids)
        representative = min(ids, key=lambda rid: (-record_map[rid]['clipscore'], rid))
        actions = {pair: combined_actions.pair_action(pair, fusion, decisions[pair], selected['threshold'])
                   for pair in itertools.combinations(ids, 2)}
        arm = export_cluster(ids, representative, actions)['b']
        exports.append(dict(component=component, record_ids=ids, representative=representative, b=arm))
    writel(run / 'b_exports.jsonl', exports)
    dump(run / 'fusion.json', fusion)
    dump(run / 'b_export_complete.json', dict(status='complete', components=len(exports), records=sum(len(x['record_ids']) for x in exports)))


def graph_counts(labels, truth):
    cells, predicted, actual = Counter(), Counter(), Counter()
    for label, group in zip(labels, truth, strict=True):
        cells[(int(label), int(group))] += 1
        predicted[int(label)] += 1
        actual[int(group)] += 1
    choose = lambda n: n * (n - 1) // 2
    tp = sum(map(choose, cells.values()))
    pp = sum(map(choose, predicted.values()))
    positives = sum(map(choose, actual.values()))
    score = Fraction(2 * tp, pp + positives) if pp + positives else Fraction(1)
    return dict(TP=tp, predicted_positive=pp, truth_positive=positives, exact_F1=str(score))


def representatives(ids, labels, scores):
    groups = defaultdict(list)
    for rid, label in zip(ids, labels, strict=True):
        groups[int(label)].append(rid)
    return sorted(min(group, key=lambda rid: (-scores[rid], rid)) for group in groups.values())


def semdedup(run):
    if (run / 'semdedup_complete.json').exists():
        require_complete(run, 'semdedup_complete.json')
        return
    require_complete(run, 'stage_e_complete.json')
    import numpy as np
    import pandas as pd
    from scripts import run_graph_baselines as baseline
    frames, features = {}, {}
    for part in ('validation', 'train'):
        frames[part] = pd.read_parquet(NOISY_POOL_RUN / f'b/{part}/corpus.parquet')
        vectors = np.load(NOISY_POOL_RUN / f'b/{part}/CLIP_image.npy')
        if vectors.shape != (len(frames[part]), 768) or not np.isfinite(vectors).all():
            raise ValueError('SemDeDup feature shape/finite mismatch')
        features[part] = np.ascontiguousarray(vectors / np.linalg.norm(vectors, axis=1, keepdims=True), dtype=np.float32)
    grid = []
    for k in (50, 100, 200, 500):
        assignment, _ = baseline.kmeans(features['validation'], k)
        for epsilon in (.005, .01, .02, .05, .1, .15, .2, .3):
            labels, edges = baseline.semdedup(features['validation'], assignment, epsilon)
            grid.append(dict(k=k, epsilon=epsilon, edges=edges,
                **graph_counts(labels, frames['validation'].original_image_id)))
        dump(run / 'semdedup_validation_progress.json', grid)
    index = min(range(len(grid)), key=lambda i: (-Fraction(grid[i]['exact_F1']), grid[i]['predicted_positive'], i))
    selected = grid[index]
    dump(run / 'semdedup_frozen.json', dict(status='frozen', selection=selected, validation_grid=grid))
    assignment, _ = baseline.kmeans(features['train'], selected['k'])
    labels, edges = baseline.semdedup(features['train'], assignment, selected['epsilon'])
    record_map = unique_index(rows(run / 'records.jsonl'), 'record_id')
    ids = frames['train'].record_id.tolist()
    kept = representatives(ids, labels, {rid: record_map[rid]['clipscore'] for rid in ids})
    dump(run / 'semdedup_complete.json', dict(status='complete', selection=selected, retained=kept, train_edges=edges))


def arm_members(exports, arm, universe):
    all_ids, state_ids = [], {state: [] for state in STATES}
    for export in exports:
        cluster = export['record_ids']
        all_ids.extend(cluster)
        parts = export[arm]
        seen = []
        for state in STATES:
            state_ids[state].extend(parts[state])
            seen.extend(parts[state])
        if len(seen) != len(set(seen)) or set(seen) != set(cluster):
            raise ValueError(f'{arm} cluster state conservation failed')
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(universe):
        raise ValueError(f'{arm} global identity coverage failed')
    return state_ids


def composition(frame, retained, image_ids):
    frame = frame.set_index('record_id', drop=False)
    selected = frame.loc[sorted(retained)]
    counts = selected.groupby('original_image_id').size().to_dict()
    histogram = Counter(counts.get(image, 0) for image in image_ids)
    original = (~frame.is_copy.astype(bool)) & (~frame.caption_swapped.astype(bool))
    return dict(records=len(selected), distinct_original_images=selected.original_image_id.nunique(),
        distinct_image_assets=selected.image_path.nunique(), captions_per_original_image_histogram={str(k): v for k, v in sorted(histogram.items())},
        retained_copies=int(selected.is_copy.sum()), retained_swapped=int(selected.caption_swapped.sum()),
        removed_original_noncopy_nonswapped=int((original & ~frame.index.isin(retained)).sum()))


def finalize(run):
    import pandas as pd
    require_complete(run, 'coverage_complete.json')
    require_complete(run, 'b_export_complete.json')
    require_complete(run, 'semdedup_complete.json')
    records, record_map, _, _, _, groups = stage_inputs(run)
    train = {r['record_id']: r for r in records if r['partition'] == 'train'}
    ids = sorted(train)
    exports = rows(run / 'coverage_exports.jsonl')
    b = arm_members(exports, 'b', ids)
    b_cov = arm_members(exports, 'b_cov', ids)
    frozen_exports = rows(run / 'b_exports.jsonl')
    frozen_components = {e['component']: set(e['record_ids']) for e in frozen_exports}
    coverage_components = {e['component']: set(e['record_ids']) for e in exports}
    if len(coverage_components) != len(exports) or coverage_components != frozen_components:
        raise ValueError('Coverage component identities differ from frozen b exports')
    original_b = arm_members(frozen_exports, 'b', ids)
    if b != original_b:
        if any(set(b[s]) != set(original_b[s]) for s in STATES):
            raise ValueError('Coverage changed frozen b states')
    removed = len(b_cov['pending']) + len(b_cov['deleted'])
    clip_removed = set(sorted(ids, key=lambda r: (train[r]['clipscore'], r))[:removed])
    random_removed = set(random.Random(20261104).sample(ids, removed))
    keep_one = [min(members, key=lambda r: (-train[r]['clipscore'], r))
                for (part, _), members in groups.items() if part == 'train']
    methods = dict(no_dedup=ids, keep_one=keep_one, semdedup=load(run / 'semdedup_complete.json')['retained'],
        clipscore=sorted(set(ids)-clip_removed), random=sorted(set(ids)-random_removed), b=b['retained'], b_cov=b_cov['retained'])
    provenance = pd.read_parquet(NOISY_POOL_RUN / 'b/train/corpus.parquet')
    if set(provenance.record_id) != set(ids):
        raise ValueError('Train record identity differs from frozen corpus')
    lineage = provenance.set_index('record_id').original_image_id.to_dict()
    image_ids = sorted(set(lineage.values()))
    if len(image_ids) != 5000:
        raise ValueError('Expected original 5000 training images')
    target = run / 'training_sets'
    target.mkdir(parents=True, exist_ok=True)
    report = {}
    for method, retained in methods.items():
        if len(retained) != len(set(retained)) or not set(retained) <= set(ids):
            raise ValueError(f'{method}: invalid retained identities')
        output = [dict(record_id=rid, original_image_id=int(lineage[rid]), image_path=train[rid]['image_path'], text=train[rid]['caption']) for rid in sorted(retained)]
        pd.DataFrame(output).to_parquet(target / f'{method}.parquet', index=False)
        report[method] = composition(provenance, set(retained), image_ids)
    clean = pd.read_parquet(NOISY_POOL_RUN / 'b/clean.parquet').sort_values(['image_id', 'caption_id'])
    output = clean.rename(columns={'image_id': 'original_image_id'}).copy()
    output['record_id'] = [f"clean:{r.image_id}:{r.caption_id}" for r in clean.itertuples()]
    if len(output) != 25009 or set(output.original_image_id) != set(image_ids) or output.record_id.duplicated().any():
        raise ValueError('Clean identity/count contract failed')
    output[['record_id', 'original_image_id', 'image_path', 'text']].to_parquet(target / 'clean.parquet', index=False)
    # Compare original caption identity, not caption wording, for Clean reporting.
    original_ids = {f"noisy_pool:train:{int(r.image_id):012d}:0:{r.caption_id}" for r in clean.itertuples()}
    clean_report = composition(provenance, original_ids & set(ids), image_ids)
    clean_report.update(records=len(output), distinct_original_images=output.original_image_id.nunique(),
        distinct_image_assets=output.image_path.nunique(), retained_copies=0, retained_swapped=0,
        removed_original_noncopy_nonswapped=0,
        captions_per_original_image_histogram={str(k): int(v) for k,v in sorted(Counter(output.groupby('original_image_id').size()).items())})
    report['clean'] = clean_report
    dump(run / 'training_sets_complete.json', dict(status='complete', removed_budget=removed, composition=report))


def resolve_test_images(run, images, image_root=None):
    """Reuse local COCO_SCALE_RUN assets and extract only unavailable IDs from local ZIPs."""
    import pandas as pd
    import shutil
    import zipfile
    resolved = {}
    if image_root is not None:
        root = Path(image_root)
        for image in images:
            image_id = int(image['cocoid'])
            candidates = [root / image['filepath'] / image['filename'],
                          root / image['filename'],
                          root / 'train2017' / f'{image_id:012d}.jpg',
                          root / 'val2017' / f'{image_id:012d}.jpg',
                          root / f'{image_id:012d}.jpg']
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(f'No local image for COCO image_id={image_id} in {root}')
            resolved[image_id] = path
        return resolved
    requested = {int(image['cocoid']) for image in images}
    cached = run / 'test_images'
    for image_id in requested:
        path = cached / f'{image_id:012d}.jpg'
        if path.is_file():
            resolved[image_id] = path
    source = RES / 'runs/coco_scale/corpus.parquet'
    frame = pd.read_parquet(source, columns=['image_id', 'image_path'])
    for image in frame.itertuples(index=False):
        image_id = int(image.image_id)
        if image_id in requested and image_id not in resolved and Path(image.image_path).is_file():
            resolved[image_id] = Path(image.image_path)
    for partition in ('train2017', 'val2017'):
        if len(resolved) == len(requested):
            break
        archive_path = COCO_ARCHIVES / f'{partition}.zip'
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            for image_id in sorted(requested - set(resolved)):
                member = f'{partition}/{image_id:012d}.jpg'
                if member not in names:
                    continue
                cached.mkdir(parents=True, exist_ok=True)
                path = cached / f'{image_id:012d}.jpg'
                temporary = path.with_suffix('.tmp')
                with archive.open(member) as source_stream, temporary.open('wb') as target:
                    shutil.copyfileobj(source_stream, target)
                temporary.replace(path)
                resolved[image_id] = path
    if set(resolved) != requested:
        raise FileNotFoundError(f'Missing frozen test images: {sorted(requested-set(resolved))[:20]}')
    return resolved


def produce_test(run, karpathy_json, image_root=None):
    import pandas as pd
    marker = run / 'test_membership_complete.json'
    if marker.exists():
        require_complete(run, marker.name)
        if not (run / 'test.parquet').is_file():
            raise FileNotFoundError(run / 'test.parquet')
        return
    ids = sorted(load(NOISY_POOL_RUN / 'b_membership.json')['image_ids']['test'])
    images = [r for r in load(karpathy_json)['images'] if r['split'] == 'test']
    if len(images) != 5000 or sorted(int(r['cocoid']) for r in images) != ids or len(set(ids)) != 5000:
        raise ValueError('Karpathy test and frozen NOISY_POOL_RUN membership differ')
    paths = resolve_test_images(run, images, image_root)
    output = [dict(image_id=image_id, original_image_id=image_id, image_path=str(paths[image_id]))
              for image_id in ids]
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_parquet(run / 'test.parquet', index=False)
    dump(marker, dict(status='complete', images=len(output), source=str(karpathy_json)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['export-b', 'semdedup', 'finalize', 'test'])
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--karpathy-json', type=Path, default=NOISY_POOL_RUN / 'dataset_coco.json')
    parser.add_argument('--image-root', type=Path)
    args = parser.parse_args()
    if args.command == 'test':
        produce_test(args.run, args.karpathy_json, args.image_root)
    else:
        {'export-b': export_b, 'semdedup': semdedup, 'finalize': finalize}[args.command](args.run)


if __name__ == '__main__':
    main()
