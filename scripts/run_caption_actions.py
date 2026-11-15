import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
RES = RESOURCES
RUN = RES / 'runs/caption_actions'
OUT = RUN / 'actions'
REPORT = ROOT / str(REPORTS / 'caption_actions')
DATA = RES / 'datasets/benchmark'
GROUPS = ('L0', 'L2', 'L3', 'M', 'unlabeled')
DECISIONS = ('duplicate', 'non_duplicate', 'keep_augment', 'mismatch', 'uncertain')

def stamp():
    return datetime.now(timezone.utc).isoformat()

def cfg():
    return json.loads((REPORT / 'settings.json').read_text())

def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)

def load(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def dump_rows(path, values):
    with Path(path).open('w') as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

def key(row):
    return tuple(sorted((row['record_id_a'], row['record_id_b'])))

def action_checks():
    from scripts import run_action_checks
    return run_action_checks

def record_inputs():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    parts = {r['record_id']: r['partition'] for r in rows(DATA / 'private/lineage.jsonl')}
    clip = RES / 'runs/pair_features/features/clip'
    meta = pq.read_table(clip / 'metadata.parquet', columns=['record_id', 'row_index']).to_pydict()
    values = np.load(clip / 'clipscore.npy')
    assert set(meta['record_id']) == set(parts) and len(parts) == len(values) == 50000
    assert sorted(meta['row_index']) == list(range(50000)) and np.isfinite(values).all()
    scores = {rid: float(values[i]) for (rid, i) in zip(meta['record_id'], meta['row_index'], strict=True)}
    return (parts, scores)

def pair_score(identity, scores):
    return min((scores[rid] for rid in identity))

def reference(part):
    return {tuple(sorted((r['record_id_a'], r['record_id_b']))): r for r in action_checks().reference_rows(DATA, part)}

def labeled(population, refs, scores):
    (selected, excluded) = ([], Counter())
    for row in population:
        identity = key(row)
        if identity not in refs:
            continue
        grade = refs[identity]['grade']
        if grade not in ('L0', 'L2', 'L3', 'M'):
            excluded[grade] += 1
            continue
        selected.append(dict(pair_id=row['pair_id'], endpoints=identity, group=grade, positive=grade == 'M', score=pair_score(identity, scores)))
    return (selected, dict(excluded))

def confusion(selected, predict):
    counts = Counter(((r['positive'], bool(predict(r))) for r in selected))
    result = action_checks().metrics(counts[True, True], counts[False, True], counts[True, False], counts[False, False])
    assert sum((result[k] for k in ('TP', 'FP', 'FN', 'TN'))) == len(selected)
    return result

def gpu_worker(part):
    from scripts import run_stage_e_clusters as stage
    stage.OUT = RUN / f'supplement_{part}'
    stage.infer('1b')

def rule_counts(selected, judgments, threshold):
    predictions = {'R_s': lambda r: r['score'] < threshold, 'VLM': lambda r: judgments[r['pair_id']] == 'mismatch', 'R_sv': lambda r: r['score'] < threshold or judgments[r['pair_id']] == 'mismatch'}
    result = {name: confusion(selected, predict) for (name, predict) in predictions.items()}
    extra = [r for r in selected if not predictions['R_s'](r) and predictions['VLM'](r)]
    extra_fp = {g: sum((not r['positive'] and r['group'] == g for r in extra)) for g in GROUPS[:3]}
    result['extra_over_R_s'] = dict(TP=sum((r['positive'] for r in extra)), FP=sum(extra_fp.values()), FP_by_group=extra_fp)
    assert result['R_sv']['TP'] == result['R_s']['TP'] + result['extra_over_R_s']['TP']
    assert result['R_sv']['FP'] == result['R_s']['FP'] + result['extra_over_R_s']['FP']
    curve = action_checks().pr_curve([r['score'] for r in selected], [r['positive'] for r in selected])
    (target_tp, total_p) = (result['VLM']['TP'], result['VLM']['TP'] + result['VLM']['FN'])
    eligible = [p for p in curve if p['TP'] >= target_tp]
    point = max(eligible, key=lambda p: (p['precision'], -p['recall']))
    result['R_s_PR'] = dict(AUPRC_average_precision=sum(((b['recall'] - a['recall']) * b['precision'] for (a, b) in zip(curve, curve[1:]))), AUPRC_trapezoidal=sum(((b['recall'] - a['recall']) * (b['precision'] + a['precision']) / 2 for (a, b) in zip(curve, curve[1:]))), at_exact_VLM_recall=dict(target_numerator=target_tp, target_denominator=total_p, target_recall=target_tp / total_p, precision=point['precision'], point=point))
    return (result, curve)

def action_summary(population, refs, judgments, public):
    records = []
    for row in population:
        (a, b) = key(row)
        grade = refs.get((a, b), {}).get('grade', 'unlabeled')
        group = grade
        assert group in GROUPS, grade
        (ra, rb) = (public[a], public[b])
        (na, nb) = (action_checks().normalize_text(ra['text']), action_checks().normalize_text(rb['text']))
        records.append(dict(pair_id=row['pair_id'], endpoints=(a, b), group=group, decision=judgments[row['pair_id']], normalized_equal=na == nb, raw_equal=ra['text'] == rb['text'], empty_normalized_equal=na == nb == '', same_image_file=Path(ra['images'][0]['image_path']).resolve() == Path(rb['images'][0]['image_path']).resolve()))
    distribution = {g: {d: sum((r['group'] == g and r['decision'] == d for r in records)) for d in DECISIONS} for g in GROUPS}
    assert sum((sum(c.values()) for c in distribution.values())) == len(population)
    return dict(distribution=distribution, keep_augment={g: dict(count=c['keep_augment'], denominator=sum(c.values()), rate=c['keep_augment'] / sum(c.values()) if sum(c.values()) else None) for (g, c) in distribution.items()}, captions=action_checks().caption_summary(records))

def policy_export(ids, scores, decisions, threshold):
    from scripts.retained_set_helpers import export_cluster
    representative = min(ids, key=lambda rid: (-scores[rid], rid))
    effective = {identity: 'mismatch' if pair_score(identity, scores) < threshold else decision for (identity, decision) in decisions.items()}
    exported = export_cluster(ids, representative, effective)['b']
    assert representative in exported['retained'] and representative not in exported['deleted']
    return (representative, exported)

def full_export():
    train = project_actions('train')
    import numpy as np
    import pyarrow.parquet as pq
    (parts, scores) = record_inputs()
    ids = pq.read_table(RES / 'runs/pair_features/public_manifest.parquet', columns=['record_id']).column(0).to_pylist()
    components = np.load(RES / cfg()['inputs']['components'])
    assert ids == sorted(ids) and len(ids) == len(components) == 50000 and (set(ids) == set(parts))
    groups = defaultdict(list)
    for (rid, component) in zip(ids, components, strict=True):
        groups[int(component)].append(rid)
    assert len(groups) == 20092
    actions = [r for p in ('test', 'validation', 'train') for r in rows(OUT / f'actions_{p}.jsonl')]
    population = list(rows(RUN / 'U_new.jsonl'))
    assert len(actions) == len(population) == len({r['pair_id'] for r in actions}) == 71930
    assert {r['pair_id'] for r in actions} == {r['pair_id'] for r in population}
    threshold = load(RUN / 'threshold_frozen.json')['t_new']
    decisions = defaultdict(dict)
    for row in actions:
        decisions[row['component']][key(row)] = row['parsed']['decision']
    (exports, totals) = ([], {p: Counter() for p in ('test', 'validation', 'train')})
    for (component, members) in groups.items():
        assert len({parts[rid] for rid in members}) == 1
        (representative, exported) = policy_export(members, scores, decisions[component], threshold)
        part = parts[members[0]]
        for name in ('retained', 'held', 'deleted'):
            totals[part][name] += len(exported[name])
        exports.append(dict(component=component, partition=part, record_ids=sorted(members), representative_id=representative, **exported))
    pair_counts = {}
    for part in totals:
        subset = [r for r in actions if r['partition'] == part]
        pair_counts[part] = dict(U=len(subset), below_t=sum((pair_score(key(r), scores) < threshold for r in subset)), terminal=len(subset), valid=sum((r['status'] == 'completed' and r['parsed']['contract_valid'] for r in subset)), invalid=sum((r['status'] != 'completed' or not r['parsed']['contract_valid'] for r in subset)), valid_uncertain=sum((r['status'] == 'completed' and r['parsed']['contract_valid'] and (r['parsed']['decision'] == 'uncertain') for r in subset)))
        assert sum(totals[part].values()) == sum((p == part for p in parts.values()))
    total = {name: sum((v[name] for v in totals.values())) for name in ('retained', 'held', 'deleted')}
    assert sum(total.values()) == 50000
    dump_rows(OUT / 'corpus_exports.jsonl', exports)
    result = dict(run_id=cfg()['run_id'], at_utc=stamp(), t_new=threshold, record_counts={p: dict(v) for (p, v) in totals.items()}, total_records=total, pair_coverage=pair_counts, terminal_total=len(actions), valid_total=sum((r['valid'] for r in pair_counts.values())), invalid_total=sum((r['invalid'] for r in pair_counts.values())), train_projection=train, exports_path=str(OUT / 'corpus_exports.jsonl'))
    dump(REPORT / 'exports_summary.json', result)

def prepare():
    population = list(rows(RUN / 'U_new.jsonl'))
    (parts, _) = record_inputs()
    assert len({r['pair_id'] for r in population}) == len(population)
    assert all((parts[r['record_id_a']] == parts[r['record_id_b']] == r['partition'] for r in population))
    OUT.mkdir(parents=True, exist_ok=True)
    for part in ('validation', 'test', 'train'):
        directory = RUN / f'supplement_{part}'
        directory.mkdir(parents=True, exist_ok=True)
        dump_rows(directory / 'phase1b.jsonl', [r for r in population if r['partition'] == part])

def select():
    (_, scores) = record_inputs()
    refs = reference('validation')
    population = [r for r in rows(RUN / 'U_new.jsonl') if r['partition'] == 'validation']
    (selected, excluded) = labeled(population, refs, scores)
    assert not excluded
    selection = action_checks().select_threshold([r['score'] for r in selected], [r['positive'] for r in selected])
    result = dict(selection, t_new=selection['threshold'], partition='validation', M=selection['positive'], same=selection['negative'], excluded_grades=excluded)
    dump(REPORT / 'threshold_frozen.json', result)
    dump(RUN / 'threshold_frozen.json', result)
    return result

def project_actions(part):
    source = RUN / f'supplement_{part}/phase1b_verdicts.jsonl'
    predictions = {r['pair_id']: r for r in rows(source)}
    population = [r for r in rows(RUN / 'U_new.jsonl') if r['partition'] == part]
    assert set(predictions) == {r['pair_id'] for r in population}
    projected = []
    for pair in population:
        original = predictions[pair['pair_id']]
        parsed = original['parsed']
        assert parsed['decision'] in DECISIONS
        if original['status'] != 'completed' or not parsed['contract_valid']:
            assert parsed['decision'] == 'uncertain'
        projected.append(dict(pair, status=original['status'], parsed=dict(decision=parsed['decision'], contract_valid=parsed['contract_valid']), source_file=str(source)))
    dump_rows(OUT / f'actions_{part}.jsonl', projected)
    return dict(rows=len(projected), valid=sum((r['status'] == 'completed' and r['parsed']['contract_valid'] for r in projected)))

def evaluate(part):
    assert part in ('validation', 'test')
    refs = reference(part)
    (_, scores) = record_inputs()
    population = [r for r in rows(RUN / 'U_new.jsonl') if r['partition'] == part]
    projected = list(rows(OUT / f'actions_{part}.jsonl'))
    judgments = {r['pair_id']: r['parsed']['decision'] for r in projected}
    assert len(judgments) == len(population) and set(judgments) == {r['pair_id'] for r in population}
    (selected, excluded) = labeled(population, refs, scores)
    observed = {g: sum((r['group'] == g for r in selected)) for g in GROUPS[:-1]}
    observed.update({g: excluded.get(g, 0) for g in ('L1', 'HN')})
    threshold = load(RUN / 'threshold_frozen.json')['t_new']
    (rules, curve) = rule_counts(selected, judgments, threshold)
    public = {r['record_id']: r for r in rows(DATA / 'public_records.jsonl')}
    summary = action_summary(population, refs, judgments, public)
    result = dict(partition=part, t_new=threshold, U=len(population), below_t=sum((pair_score(key(r), scores) < threshold for r in population)), labels=observed, rules=rules, **summary, invalid_terminal=sum((r['status'] != 'completed' or not r['parsed']['contract_valid'] for r in projected)))
    dump(OUT / f'{part}_PR_curves.json', dict(main=curve))
    dump(REPORT / f'{part}_actions.json', result)
    return result
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'select', 'infer', 'project', 'evaluate', 'export'])
    p.add_argument('--partition', choices=['validation', 'test', 'train'], default='validation')
    a = p.parse_args()
    if a.action == 'infer':
        gpu_worker(a.partition)
    elif a.action == 'project':
        project_actions(a.partition)
    elif a.action == 'evaluate':
        evaluate(a.partition)
    elif a.action == 'export':
        full_export()
    else:
        globals()[a.action]()
