import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import sys
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.build_candidates import normalize_text
pa.set_cpu_count(4)
pa.set_io_thread_count(4)
RES = RESOURCES
OUT = RES / 'runs/action_checks'
REPORT = ROOT / str(REPORTS / 'action_checks')
GROUPS = ('L0', 'L2', 'L3', 'M', 'unlabeled')
DECISIONS = ('duplicate', 'non_duplicate', 'keep_augment', 'mismatch', 'uncertain')

def stamp():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)

def key(a, b):
    return tuple(sorted((a, b)))

def pair_id(a, b):
    return 'stage_e_clusters__' + '__'.join(key(a, b))

def metrics(tp, fp, fn, tn):
    return dict(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn), precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn) if tp + fn else 0.0, F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)

def pr_curve(scores, positive):
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    assert len(scores) == len(positive) and len(scores) and np.isfinite(scores).all()
    order = np.argsort(scores, kind='stable')
    (values, counts) = np.unique(scores[order], return_counts=True)
    ends = np.cumsum(counts)
    cumulative = np.cumsum(positive[order])
    (total_p, total_n) = (int(positive.sum()), int((~positive).sum()))
    curve = [dict(cutoff_score=None, threshold=float(values[0]), **metrics(0, 0, total_p, total_n))]
    curve[0]['precision'] = 1.0
    for (value, end) in zip(values, ends, strict=True):
        tp = int(cumulative[end - 1])
        fp = int(end) - tp
        curve.append(dict(cutoff_score=float(value), threshold=float(np.nextafter(value, np.inf)), **metrics(tp, fp, total_p - tp, total_n - fp)))
    return curve

def select_threshold(scores, positive):
    values = np.unique(np.asarray(scores, dtype=float))
    curve = pr_curve(scores, positive)
    assert curve[0]['FN'] > 0 and curve[0]['TN'] > 0
    objective = [Fraction(2 * p['TP'], 2 * p['TP'] + p['FP'] + p['FN']) for p in curve[:-1]]
    best = max(objective)
    indices = [i for (i, f1) in enumerate(objective) if f1 == best]
    intervals = []
    for i in indices:
        if intervals and i == intervals[-1]['last_candidate_index'] + 1:
            intervals[-1].update(upper=float(values[i]), last_candidate_index=i)
        else:
            intervals.append(dict(lower=float(values[i - 1]) if i else None, upper=float(values[i]), first_candidate_index=i, last_candidate_index=i))
    interval = intervals[0]
    threshold = interval['upper'] if interval['lower'] is None else (interval['lower'] + interval['upper']) / 2
    (s, y) = (np.asarray(scores, dtype=float), np.asarray(positive, dtype=bool))
    pred = s < threshold
    observed = metrics(int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()), int((~pred & ~y).sum()))
    assert Fraction(2 * observed['TP'], 2 * observed['TP'] + observed['FP'] + observed['FN']) == best
    return dict(threshold=threshold, optimal_intervals=intervals, exact_F1=str(best), distinct_score_candidates=len(values), positive=int(y.sum()), negative=int((~y).sum()), metrics=observed)

def inputs(cfg):
    src = cfg['inputs']
    graph = Path(src['graph_root'])
    ids = pq.read_table(graph / 'component_row_ids.parquet', columns=['record_id']).column(0).to_pylist()
    components = np.load(graph / 'HGB_K8/components.npy')
    parts = {r['record_id']: r['partition'] for r in rows(Path(src['dataset_root']) / 'private/lineage.jsonl')}
    metadata = pq.read_table(src['clipscore_mapping'].split(':')[0], columns=['record_id', 'row_index']).to_pydict()
    cache = np.load(src['clipscore'])
    assert len(ids) == len(set(ids)) == len(components) == len(parts) == len(cache) == 50000
    assert set(ids) == set(parts) == set(metadata['record_id'])
    assert sorted(metadata['row_index']) == list(range(len(cache)))
    scores = {rid: float(cache[i]) for (rid, i) in zip(metadata['record_id'], metadata['row_index'], strict=True)}
    assert all((np.isfinite(v) for v in scores.values()))
    return (dict(zip(ids, map(int, components), strict=True)), parts, scores)

def reference_rows(dataset, partition):
    pattern = re.compile('"partition"\\s*:\\s*"' + partition + '"')
    with (Path(dataset) / 'sealed/relation_applicability.jsonl').open() as stream:
        for line in stream:
            if pattern.search(line):
                row = json.loads(line)
                assert row['partition'] == partition
                yield row

def labeled_population(cfg, partition, components, parts, scores):
    (selected, excluded, references) = ([], Counter(), {})
    for row in reference_rows(cfg['inputs']['dataset_root'], partition):
        (a, b) = (row['record_id_a'], row['record_id_b'])
        assert parts[a] == parts[b] == partition
        if components[a] != components[b]:
            continue
        identity = key(a, b)
        assert identity not in references
        grade = row['grade']
        references[identity] = grade
        if grade in ('L0', 'L2', 'L3', 'M'):
            selected.append(dict(endpoints=identity, positive=grade == 'M', score=min(scores[a], scores[b])))
        else:
            excluded[grade] += 1
    return (selected, references, dict(excluded))

def select(cfg):
    target = OUT / 'selection_frozen.json'
    (components, parts, scores) = inputs(cfg)
    (population, _, excluded) = labeled_population(cfg, 'validation', components, parts, scores)
    selection = select_threshold([r['score'] for r in population], [r['positive'] for r in population])
    selection.update(frozen_at_utc=stamp(), partition='validation', excluded_grades=excluded)
    write_json(target, selection)

def caption_summary(pairs):

    def summarize(selected):
        result = {}
        for group in (*GROUPS, 'ALL'):
            subset = [r for r in selected if group == 'ALL' or r['group'] == group]
            counts = Counter((r['decision'] for r in subset))
            result[group] = dict(total=len(subset), same_image_file=sum((r['same_image_file'] for r in subset)), decisions={d: counts[d] for d in DECISIONS}, keep_augment=counts['keep_augment'], keep_augment_rate=counts['keep_augment'] / len(subset) if subset else None)
        assert sum((result[g]['total'] for g in GROUPS)) == result['ALL']['total']
        assert all((sum(s['decisions'].values()) == s['total'] for s in result.values()))
        return result
    result = {}
    for mode in ('normalized_equal', 'raw_equal'):
        result[mode] = {kind: summarize([r for r in pairs if r[mode] == equal]) for (kind, equal) in (('equal', True), ('different', False))}
        assert sum((result[mode][kind]['ALL']['total'] for kind in ('equal', 'different'))) == len(pairs)
    result['normalized_only'] = summarize([r for r in pairs if r['normalized_equal'] and (not r['raw_equal'])])
    result['empty_normalized_equal'] = summarize([r for r in pairs if r['empty_normalized_equal']])
    assert all((not r['raw_equal'] or r['normalized_equal'] for r in pairs))
    return result

def evaluate(cfg):
    selection = read_json(OUT / 'selection_frozen.json')
    (components, parts, scores) = inputs(cfg)
    (population, references, excluded) = labeled_population(cfg, 'test', components, parts, scores)
    assert not excluded, excluded
    stage_e_clusters = Path(cfg['inputs']['stage_e_clusters_root'])
    utest = list(rows(stage_e_clusters / 'U_test.jsonl'))
    members = {}
    for (rid, component) in components.items():
        if parts[rid] == 'test':
            members.setdefault(component, []).append(rid)
    import itertools
    actual = {key(a, b) for group in members.values() for (a, b) in itertools.combinations(group, 2)}
    assert actual == {key(r['record_id_a'], r['record_id_b']) for r in utest}
    assert len(utest) == len(actual)
    verdicts = {}
    for phase in ('1a', '1b'):
        for row in rows(stage_e_clusters / f'phase{phase}_verdicts.jsonl'):
            assert row['pair_id'] not in verdicts
            verdicts[row['pair_id']] = row['parsed']['decision']
    assert set(verdicts) == {r['pair_id'] for r in utest}
    records = {r['record_id']: r for r in rows(Path(cfg['inputs']['dataset_root']) / 'public_records.jsonl')}
    pair_results = []
    for row in utest:
        (a, b) = (row['record_id_a'], row['record_id_b'])
        assert row['pair_id'] == pair_id(a, b) and parts[a] == parts[b] == 'test'
        (ra, rb) = (records[a], records[b])
        assert len(ra['images']) == len(rb['images']) == 1
        assert isinstance(ra['text'], str) and isinstance(rb['text'], str)
        (na, nb) = (normalize_text(ra['text']), normalize_text(rb['text']))
        grade = references.get(key(a, b), 'unlabeled')
        group = grade
        assert group in GROUPS and verdicts[row['pair_id']] in DECISIONS
        pair_results.append(dict(pair_id=row['pair_id'], endpoints=key(a, b), group=group, decision=verdicts[row['pair_id']], normalized_equal=na == nb, raw_equal=ra['text'] == rb['text'], empty_normalized_equal=na == nb == '', same_image_file=Path(ra['images'][0]['image_path']).resolve() == Path(rb['images'][0]['image_path']).resolve()))
    aggregate = read_json(ROOT / cfg['inputs']['stage_e_clusters_aggregate'])
    s = np.array([r['score'] for r in population])
    y = np.array([r['positive'] for r in population])
    pred = s < selection['threshold']
    frozen_metrics = metrics(int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()), int((~pred & ~y).sum()))
    curve = pr_curve(s, y)
    ap = sum(((b['recall'] - a['recall']) * b['precision'] for (a, b) in zip(curve, curve[1:])))
    trap = sum(((b['recall'] - a['recall']) * (a['precision'] + b['precision']) / 2 for (a, b) in zip(curve, curve[1:])))

    def at_recall(target):
        eligible = [p for p in curve if p['recall'] >= target]
        point = max(eligible, key=lambda p: (p['precision'], -p['recall']))
        return dict(target_recall=target, precision=point['precision'], point=point)
    mismatch = dict(validation=selection, test=dict(positive=int(y.sum()), negative=int((~y).sum()), frozen_threshold=selection['threshold'], metrics=frozen_metrics, AUPRC_average_precision=ap, AUPRC_trapezoidal=trap, PR_curve='test_pr_curve.json', PR_points=len(curve), at_requested_recall=at_recall(0.892256), at_exact_vlm_recall=at_recall(aggregate['recall'])), vlm_point=dict(source=cfg['inputs']['stage_e_clusters_aggregate'], confusion=aggregate['confusion'], precision=aggregate['precision'], recall=aggregate['recall'], F1=aggregate['F1']))
    result = dict(run_id=cfg['run_id'], source_contract=cfg['source_contract'], mismatch=mismatch, captions=caption_summary(pair_results), provenance=cfg['inputs'], new_vlm_api_gpu_encoding_fitting=0)
    write_json(OUT / 'pair_analysiss.json', pair_results)
    write_json(REPORT / 'selection_frozen.json', selection)
    write_json(REPORT / 'test_pr_curve.json', dict(prediction='score < threshold', positive=int(y.sum()), negative=int((~y).sum()), points=curve))
    write_json(REPORT / 'results.json', result)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['select', 'evaluate'])
    a = p.parse_args()
    config = read_json(REPORT / 'settings.json')
    globals()[a.action](config)
