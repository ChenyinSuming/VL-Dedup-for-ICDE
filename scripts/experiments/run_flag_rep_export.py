#!/usr/bin/env python3
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
"""FLAG_REP_RUN frozen-score exports; writes only FLAG_REP_RUN resources and reports."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiments.support import export_cluster
from scripts.run_combined_actions import pair_action
from scripts.experiments.run_coverage_export import coverage_export
from scripts.experiments.run_caption_downstream_sets import load, rows, dump, writel

RES = RESOURCES
RUN = RES / 'runs/flag_rep_export'
REPORT = ROOT / 'outputs/reports/flag_rep_export'
COMBINED_ACTIONS_RUN = RES / 'runs/fused_main'
COVERAGE_EXPORT_RUN = RES / 'runs/coverage_export'
CAPTION_DOWNSTREAM_RUN = RES / 'runs/caption_downstream'
STATES = ('retained', 'pending', 'deleted')


def flag_rep(ids, clip, fusion, decisions, threshold):
    """Keep complete ids/actions, including incident mismatch effects on peers."""
    ids = sorted(ids)
    flagged = {r for r in ids if fusion[r] < threshold}
    if flagged == set(ids):
        return None, dict(retained=[], pending=ids, deleted=[])
    representative = min(set(ids) - flagged, key=lambda r: (-clip[r], r))
    actions = {pair: pair_action(pair, fusion, decisions[pair], threshold)
               for pair in itertools.combinations(ids, 2)}
    arm = export_cluster(ids, representative, actions)['b']
    if not flagged <= set(arm['pending']):
        raise ValueError('Flagged records escaped held state')
    validate_arm(ids, arm)
    return representative, arm


def validate_arm(ids, arm):
    parts = [set(arm[s]) for s in STATES]
    if set.union(*parts) != set(ids) or sum(map(len, parts)) != len(ids):
        raise ValueError('Export state conservation failed')


def matched_controls(ids, clip, removed):
    ids = sorted(ids)
    clip_removed = set(sorted(ids, key=lambda r: (clip[r], r))[:removed])
    random_removed = set(random.Random(20261106).sample(ids, removed))
    return dict(clipscore_fr=sorted(set(ids) - clip_removed),
                random_fr=sorted(set(ids) - random_removed))


def counts(arms):
    return {s: sum(len(a[s]) for a in arms) for s in STATES}


def gate(name, actual, expected, receipt):
    receipt['checks'][name] = dict(actual=actual, expected=expected, passed=actual == expected)
    dump(RUN / 'gates.json', receipt)
    if actual != expected:
        raise RuntimeError(f'Gate {name}: independently computed {actual!r} != {expected!r}')


def benchmark():
    from scripts import run_combined_actions as combined_actions
    begin = time.monotonic()
    protocol = load(CONFIG / 'flag_rep_export.json')
    receipt = dict(status='running', checks={})
    parts, groups, clip = combined_actions.inputs()
    scores = load(COMBINED_ACTIONS_RUN / 'record_scores.json')
    fusion = {r: v['fusion'] for r, v in scores.items()}
    gate('benchmark_scored_records', len(scores), 38102, receipt)
    frozen_cov = {(r['partition'], r['component']): r for r in rows(COVERAGE_EXPORT_RUN / 'exports.jsonl')
                  if r['domain'] == 'benchmark'}
    cache = {}
    for r in rows(COVERAGE_EXPORT_RUN / 'checks.jsonl'):
        if not r['job'].startswith('benchmark:'):
            continue
        key = (r['source_id'], tuple(sorted(r['retained_ids'])))
        value = r['covered'] if r['valid'] else None
        if key in cache and cache[key] != value:
            raise ValueError('Conflicting frozen COVERAGE_EXPORT_RUN coverage checks')
        cache[key] = value

    def check(source, retained):
        return cache[source, tuple(sorted(retained))]

    exports, summary = [], {}
    for part in ('train', 'validation', 'test'):
        decisions = defaultdict(dict)
        for row in rows(combined_actions.REFERENCE / f'actions_{part}.jsonl'):
            decisions[row['component']][combined_actions.key(row)] = row['parsed']['decision']
        old = {r['component']: r for r in rows(COMBINED_ACTIONS_RUN / f'exports_{part}.jsonl')}
        flagged_reps = all_flagged = b_diff = cov_diff = 0
        selected = []
        for component, ids in sorted(groups.items()):
            if parts[ids[0]] != part or len(ids) == 1:
                continue
            representative = min(ids, key=lambda r: (-clip[r], r))
            actions = {pair: pair_action(pair, fusion, decisions[component][pair], combined_actions.TF)
                       for pair in itertools.combinations(sorted(ids), 2)}
            b = export_cluster(ids, representative, actions)['b']
            if any(b[s] != old[component][s] for s in STATES):
                raise RuntimeError(f'COMBINED_ACTIONS_RUN per-cluster replay mismatch: {part}/{component}')
            b_cov = coverage_export(b, clip, check)
            archived = frozen_cov[part, component]
            if b != archived['b'] or b_cov != archived['b_cov']:
                raise RuntimeError(f'COVERAGE_EXPORT_RUN per-cluster replay mismatch: {part}/{component}')
            new_rep, fr = flag_rep(ids, clip, fusion, decisions[component], combined_actions.TF)
            fr_cov = coverage_export(fr, clip, check)
            was_flagged = fusion[representative] < combined_actions.TF
            if not was_flagged and b != fr:
                raise RuntimeError('Unflagged representative changed b')
            flagged_reps += was_flagged
            all_flagged += new_rep is None
            b_diff += b != fr
            cov_diff += b_cov != fr_cov
            item = dict(partition=part, component=component, record_ids=sorted(ids),
                        representative=representative, flag_representative=new_rep,
                        b=b, b_cov=b_cov, flag_rep=fr, flag_rep_cov=fr_cov)
            selected.append(item)
        # Historical count gates include singleton components, which A1 does not modify.
        singleton_count = sum(len(v) == 1 and parts[v[0]] == part for v in groups.values())
        for arm in ('b', 'b_cov'):
            totals = counts([r[arm] for r in selected])
            totals['retained'] += singleton_count
            gate(f'benchmark_{part}_{arm}', list(totals.values()),
                 protocol['gates']['benchmark_' + arm][part], receipt)
        gate(f'benchmark_{part}_clusters', len(selected),
             protocol['benchmark']['multi_clusters_expected'][part], receipt)
        summary[part] = dict(multi_clusters=len(selected), flagged_representatives=flagged_reps,
                             all_flagged_clusters=all_flagged, b_different_clusters=b_diff,
                             b_cov_different_clusters=cov_diff, singletons_excluded=singleton_count)
        exports.extend(selected)
    sampled = rows(COVERAGE_EXPORT_RUN / 'sample_clusters.jsonl')
    index = {r['component']: r for r in exports}
    audit = {}
    for r in sampled:
        if set(r['record_ids']) != set(index[r['component']]['record_ids']):
            raise RuntimeError('Frozen 200-cluster audit identity differs')
    for arm in ('b', 'b_cov', 'flag_rep', 'flag_rep_cov'):
        audit[arm] = counts([index[r['component']][arm] for r in sampled])
    gate('audit_b', list(audit['b'].values()), [687, 148, 118], receipt)
    gate('audit_b_cov', list(audit['b_cov'].values()), [750, 148, 55], receipt)
    receipt.update(status='passed', A1_seconds=time.monotonic() - begin)
    dump(RUN / 'gates.json', receipt)
    writel(RUN / 'benchmark_exports.jsonl', exports)
    result = dict(task='flag_rep', stage='A1', partitions=summary,
                  audit_clusters=len(sampled), audit_records=sum(len(r['record_ids']) for r in sampled),
                  audit=audit, elapsed_seconds=receipt['A1_seconds'],
                  paper_fates='Unchanged iff per-cluster retained/pending/deleted sets are identical; no new audit labels.',
                  singletons='A2 separate report only; no A1 changes')
    dump(REPORT / 'benchmark/results_a1.json', result)
    return result


def training_sets():
    import pandas as pd
    from scripts.experiments import run_caption_downstream_sets as caption_downstream
    from scripts.experiments.run_coverage_export import text_key
    begin = time.monotonic()
    receipt = load(RUN / 'gates.json')
    records, record_map, _, pairs, verdicts, groups = caption_downstream.stage_inputs(CAPTION_DOWNSTREAM_RUN)
    threshold = load(CAPTION_DOWNSTREAM_RUN / 'threshold_frozen.json')['threshold']
    fusion = load(CAPTION_DOWNSTREAM_RUN / 'fusion.json')
    gate('caption_downstream_threshold', threshold, .06547619047619047, receipt)
    train = {r['record_id']: r for r in records if r['partition'] == 'train'}
    gate('caption_downstream_train_records', len(train), 35208, receipt)
    decisions = {tuple(sorted((p['record_id_a'], p['record_id_b']))): verdicts[pid]['parsed']['decision']
                 for pid, p in pairs.items()}
    clip = {r: v['clipscore'] for r, v in train.items()}
    cache = {}
    coverage_export_protocol = load(ROOT / 'outputs/reports/coverage_export/protocol.json')
    historical = Path(coverage_export_protocol['inputs']['coverage_check'])
    inputs = {(r['group'], r['unit_id']): r for r in rows(historical / 'inputs.jsonl')}
    for prediction in rows(historical / 'predictions.jsonl'):
        row = inputs[prediction['group'], prediction['unit_id']]
        key = json.dumps(dict(deleted_caption=row['deleted_caption'],
                             retained_captions=[r['caption'] for r in row['retained']]), ensure_ascii=False)
        cache[key] = prediction
    historical_records = load(COVERAGE_EXPORT_RUN / 'records.json')
    for prediction in rows(COVERAGE_EXPORT_RUN / 'predictions.jsonl'):
        key = text_key(prediction['source_id'], prediction['retained_ids'], historical_records)
        cache[key] = prediction
    for path in [CAPTION_DOWNSTREAM_RUN / 'coverage_predictions.jsonl', RUN / 'coverage_predictions.jsonl']:
        if path.exists():
            for prediction in rows(path):
                cache[prediction['text_key']] = prediction
    logical = []

    def check(source, retained):
        key = text_key(source, retained, record_map)
        if key not in cache:
            dump(RUN / 'new_coverage_required.json', dict(text_key=key, source_id=source, retained_ids=retained))
            raise RuntimeError('New coverage text_key requires GPU coverage phase')
        prediction = cache[key]
        value = prediction['covered'] if prediction['valid'] else None
        logical.append(dict(source_id=source, retained_ids=retained, text_key=key, covered=value, origin=prediction.get('origin')))
        return value

    frozen = {e['component']: e for e in rows(CAPTION_DOWNSTREAM_RUN / 'coverage_exports.jsonl')}
    exports = []
    flagged_representatives = 0
    for (part, component), ids in sorted(groups.items()):
        if part != 'train':
            continue
        ids = sorted(ids)
        rep = min(ids, key=lambda r: (-clip[r], r))
        actions = {p: pair_action(p, fusion, decisions[p], threshold) for p in itertools.combinations(ids, 2)}
        b = export_cluster(ids, rep, actions)['b']
        b_cov = coverage_export(b, clip, check)
        if b != frozen[component]['b'] or b_cov != frozen[component]['b_cov']:
            raise RuntimeError(f'CAPTION_DOWNSTREAM_RUN old per-cluster replay differs: {component}')
        new_rep, fr = flag_rep(ids, clip, fusion, decisions, threshold)
        if fusion[rep] >= threshold and fr != b:
            raise RuntimeError('CAPTION_DOWNSTREAM_RUN unflagged representative changed b')
        fr_cov = coverage_export(fr, clip, check)
        flagged_representatives += fusion[rep] < threshold
        exports.append(dict(component=component, record_ids=ids, representative=rep,
                            flag_representative=new_rep, b=b, b_cov=b_cov, b_fr=fr, b_fr_cov=fr_cov))
    for arm, expected in [('b', 25250), ('b_cov', 27494)]:
        members = caption_downstream.arm_members(exports, arm, train)
        archived_ids = set(pd.read_parquet(CAPTION_DOWNSTREAM_RUN / f'training_sets/{arm}.parquet').record_id)
        gate(f'caption_downstream_{arm}_exact_retained_IDs', set(members['retained']) == archived_ids, True, receipt)
        gate(f'caption_downstream_{arm}_records', len(members['retained']), expected, receipt)
    # Reporting truth is opened only after all method decisions are fixed.
    provenance = pd.read_parquet(caption_downstream.NOISY_POOL_RUN / 'b/train/corpus.parquet')
    if set(provenance.record_id) != set(train):
        raise RuntimeError('NOISY_POOL_RUN train provenance identity differs')
    swapped = set(provenance[provenance.caption_swapped.astype(bool)].record_id)
    flagged = {r for r in train if fusion[r] < threshold}
    all_swapped = [e for e in exports if set(e['record_ids']) <= swapped]
    for name, actual, expected in [
        ('flagged_swapped', len(flagged & swapped), 6767),
        ('flagged_other', len(flagged - swapped), 51),
        ('all_swapped_clusters', len(all_swapped), 1162),
        ('all_swapped_cluster_flagged_representatives', sum(e['representative'] in flagged for e in all_swapped), 1096)]:
        gate('caption_downstream_' + name, actual, expected, receipt)
    members = {arm: caption_downstream.arm_members(exports, arm, train) for arm in ('b_fr', 'b_fr_cov')}
    removed = len(members['b_fr_cov']['pending']) + len(members['b_fr_cov']['deleted'])
    methods = {arm: members[arm]['retained'] for arm in members}
    methods.update(matched_controls(train, clip, removed))
    lineage = provenance.set_index('record_id').original_image_id.to_dict()
    image_ids = sorted(set(lineage.values()))
    composition = {}
    target = RUN / 'training_sets'
    target.mkdir(exist_ok=True)
    for method, kept in methods.items():
        frame = pd.DataFrame([dict(record_id=r, original_image_id=int(lineage[r]),
                     image_path=train[r]['image_path'], text=train[r]['caption']) for r in sorted(kept)])
        frame.to_parquet(target / f'{method}.parquet', index=False)
        composition[method] = caption_downstream.composition(provenance, set(kept), image_ids)
    writel(RUN / 'training_exports.jsonl', exports)
    writel(RUN / 'coverage_checks.jsonl', logical)
    dump(RUN / 'subsets.json', load(CAPTION_DOWNSTREAM_RUN / 'subsets.json'))
    old_retained = set(caption_downstream.arm_members(exports, 'b_cov', train)['retained'])
    new_held = set(members['b_fr_cov']['pending'])
    result = dict(status='complete', task='flag_rep', composition=composition,
                  states={arm: counts([e[arm] for e in exports]) for arm in members},
                  matched_removed_records=removed, flagged_records=len(flagged),
                  flagged_representatives=flagged_representatives,
                  caption_downstream_b_cov_flagged_representatives_now_held=len(old_retained & flagged & new_held),
                  coverage_logical_checks=len(logical), new_coverage_requests=0,
                  elapsed_seconds=time.monotonic() - begin)
    dump(RUN / 'training_sets_complete.json', result)
    dump(REPORT / 'downstream/training_sets.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['benchmark', 'training-sets'])
    args = parser.parse_args()
    try:
        (benchmark if args.stage == 'benchmark' else training_sets)()
    except Exception as exc:
        dump(RUN / 'stop.json', dict(status='stopped', stage=args.stage,
             error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
