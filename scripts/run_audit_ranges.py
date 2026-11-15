import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
ROOT = REPO_ROOT
RUNS = RESOURCES / 'runs'
RUN = RUNS / 'audit_ranges'
REPORT = ROOT / str(REPORTS / 'audit_ranges')
NATURAL = RUNS / 'caption_errors/natural'
RETAINED = RUNS / 'retained_set/retained_set'
mismatch_scores = RUNS / 'mismatch_scores'
THRESHOLDS = ROOT / str(REPORTS / 'mismatch_scores' / 'thresholds.json')
ERROR = {'local_error', 'whole_mismatch'}
STRATA = ('L0', 'L2', 'L3')
RULES = ('strict', 'protocol', 'loose')

def load(path):
    return json.loads(path.read_text())

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def unique(items, key):
    result = {r[key]: r for r in items}
    assert len(result) == len(items), f'duplicate {key}'
    return result

def caption_errors_error(first, second, rule, protocol_error=None):
    if rule == 'strict':
        return any((first[c]['label'] in ERROR and second[c]['label'] in ERROR for c in ('a', 'b')))
    if rule == 'loose':
        return any((a[c]['label'] in ERROR for a in (first, second) for c in ('a', 'b')))
    assert rule == 'protocol' and isinstance(protocol_error, bool)
    return protocol_error

def human_error(answer):
    return any((answer[c]['label'] != 'correct' for c in ('a', 'b')))

def retained_set_lost(first, second, rule):
    assert {first, second} <= {'lost', 'uncertain', 'covered_single', 'covered_joint'}
    assert rule in ('strict', 'loose')
    return first == second == 'lost' if rule == 'strict' else 'lost' in (first, second)

def midrank(reference, values):
    reference = np.sort(np.asarray(reference, dtype=float))
    assert len(reference)
    return (np.searchsorted(reference, values, side='left') + np.searchsorted(reference, values, side='right')) / (2 * len(reference))

def combined_scores(records, clip, probabilities):
    rec = unique(records, 'record_id')
    prob = unique(probabilities, 'record_id')
    assert set(rec) == set(clip) == set(prob)
    ids = list(rec)
    validation = [rid for rid in ids if rec[rid]['partition'] == 'validation']
    cp = midrank([clip[r]['clipscore'] for r in validation], [clip[r]['clipscore'] for r in ids])
    qp = midrank([prob[r]['probability'] for r in validation if prob[r]['valid']], [prob[r]['probability'] if prob[r]['valid'] else 1.0 for r in ids])
    return ({r: float((c + (q if prob[r]['valid'] else 1.0)) / 2) for (r, c, q) in zip(ids, cp, qp, strict=True)}, len(validation))

def in_combined(pair, scores, threshold):
    return min(scores[pair['record_id_a']], scores[pair['record_id_b']]) >= threshold

def count_rate(values):
    return {'count': sum(values), 'n': len(values), 'rate': sum(values) / len(values)}

def compare(audit, control):
    (a, c) = (count_rate(audit), count_rate(control))
    return {'audit': a, 'control': c, 'rate_difference': a['rate'] - c['rate'], 'rate_ratio': a['rate'] / c['rate'] if c['rate'] else None, 'fisher_one_sided_greater_p': float(fisher_exact([[a['count'], a['n'] - a['count']], [c['count'], c['n'] - c['count']]], alternative='greater').pvalue)}

def summarize_caption_errors(pairs):
    result = {}
    for rule in RULES:
        result[rule] = compare(*[[p[rule] for p in pairs if p['arm'] == arm] for arm in ('audit', 'control')])
        result[rule]['strata'] = {g: compare(*[[p[rule] for p in pairs if p['arm'] == arm and p['group'] == g] for arm in ('audit', 'control')]) for g in STRATA}
    types = {}
    for arm in ('audit', 'control'):
        (captions, labels) = ([], Counter())
        for p in pairs:
            if p['arm'] != arm:
                continue
            for c in ('a', 'b'):
                if all((a[c]['label'] in ERROR for a in p['answers'])):
                    union = sorted(set().union(*(a[c]['error_types'] for a in p['answers'])))
                    labels.update(union)
                    captions.append({'blind_id': p['blind_id'], 'caption': c, 'error_types': union})
        types[arm] = {'strict_error_captions': len(captions), 'type_caption_counts': dict(sorted(labels.items())), 'captions': captions}
    result['strict_error_types'] = types
    return result

def main():
    protocol = load(REPORT / 'settings.json')
    source = Path(protocol['human_inputs']['source'])
    human = read_human_labels(load(source))
    accepted = unique(rows(NATURAL / 'ai_accepted.jsonl'), 'blind_id')
    key = rows(NATURAL / 'private_key.jsonl')
    assert len(unique(key, 'blind_id')) == 120
    assert set(accepted).isdisjoint(human) and set(accepted) | set(human) == {p['blind_id'] for p in key}
    (inputs, pairs) = ([], [])
    review = {'AI_same': 0, 'human_changed': 0, 'changed_ids': [], 'AI_different': 0, 'human_error': 0}
    for p in key:
        bid = p['blind_id']
        answers = []
        for auditor in [1, 2]:
            terminal = load(NATURAL / f'requests/auditor_{auditor}/{bid}/terminal.json')
            assert terminal['blind_id'] == terminal['answer']['blind_id'] == bid
            assert terminal['auditor'] == auditor
            answer = terminal['answer']
            assert all((answer[c]['label'] in ERROR | {'correct', 'unsure'} for c in ('a', 'b')))
            answers.append(answer)
        if bid in accepted:
            assert all((answers[0][c]['label'] == answers[1][c]['label'] != 'unsure' for c in ('a', 'b')))
            pe = accepted[bid]['pair_error']
            assert pe == caption_errors_error(*answers, 'strict')
        else:
            pe = human_error(human[bid])
            inputs.append({'blind_id': bid, 'pair_id': p['pair_id'], 'arm': p['arm'], 'group': p['group'], 'auditor_1': answers[0], 'auditor_2': answers[1], 'human': human[bid], 'pair_error': pe})
            for c in ('a', 'b'):
                same = answers[0][c]['label'] == answers[1][c]['label']
                review['AI_same' if same else 'AI_different'] += 1
                if same and human[bid][c]['label'] != answers[0][c]['label']:
                    review['human_changed'] += 1
                    review['changed_ids'].append(f'{bid} {c}')
                if not same and human[bid][c]['label'] in ERROR:
                    review['human_error'] += 1
        pairs.append(dict(p, answers=answers, **{rule: caption_errors_error(*answers, rule, pe) for rule in RULES}))
    dump(RUN / 'caption_errors_inputs.json', {'run_id': RUN.name, 'count': len(inputs), 'source': str(source), 'pairs': inputs})
    (scores, validation_n) = combined_scores(rows(mismatch_scores / 'records.jsonl'), load(mismatch_scores / 'cpu_scores.json'), rows(mismatch_scores / 'qwen25_vllm.jsonl'))
    threshold = load(THRESHOLDS)['thresholds']['fusion']['threshold']
    assert threshold == protocol['caption_errors']['fusion']['t_f']
    combined = [p for p in pairs if in_combined(p, scores, threshold)]
    excluded = {a: [p['blind_id'] for p in pairs if p['arm'] == a and (not in_combined(p, scores, threshold))] for a in ('audit', 'control')}
    agreement = rows(RETAINED / 'agreement.jsonl')
    cells = unique(rows(RETAINED / 'audit_cells.jsonl'), 'cell_id')
    assert set(unique(agreement, 'cell_id')) == set(cells)
    clusters = unique(rows(RETAINED / 'sample_clusters.jsonl'), 'cluster_id')
    retained_set = {}
    for strategy in ('a', 'b'):
        retained_set[strategy] = {}
        for size in ('all', '2-4', '5'):
            selected = [r for r in agreement if r['strategy'] == strategy and (size == 'all' or (2 <= clusters[r['cluster_id']]['size'] <= 4 if size == '2-4' else clusters[r['cluster_id']]['size'] == 5))]
            labels = []
            for r in selected:
                assert all((r[f'audit_{i}']['status'] == 'completed' for i in (1, 2)))
                assert all((r[k] == cells[r['cell_id']][k] for k in ('source_id', 'strategy', 'retained_ids', 'unit_id')))
                labels.append(tuple((r[f'audit_{i}']['audit']['label'] for i in (1, 2))))
            retained_set[strategy][size] = {rule: count_rate([retained_set_lost(*ls, rule) for ls in labels]) for rule in ('strict', 'loose')}
            retained_set[strategy][size]['auditor_lost'] = {str(i + 1): sum((ls[i] == 'lost' for ls in labels)) for i in (0, 1)}
    result = {'run_id': RUN.name, 'provenance': {'caption_errors': str(NATURAL), 'retained_set': str(RETAINED), 'mismatch_scores': str(mismatch_scores), 'thresholds': str(THRESHOLDS), 'additional_CLIP_input': str(mismatch_scores / 'cpu_scores.json')}, 'human_caption_review': review, 'fusion': {'threshold': threshold, 'validation_records': validation_n, 'excluded': excluded, 'excluded_details': [dict(blind_id=p['blind_id'], arm=p['arm'], group=p['group'], scores={p[k]: scores[p[k]] for k in ('record_id_a', 'record_id_b')}) for p in pairs if not in_combined(p, scores, threshold)]}, 'caption_errors': {'original': summarize_caption_errors(pairs), 'fusion': summarize_caption_errors(combined)}, 'RETAINED': retained_set}
    dump(RUN / 'caption_errors_pair_rules.json', [{k: v for (k, v) in p.items() if k != 'answers'} for p in pairs])
    dump(REPORT / 'results.json', result)

def read_human_labels(returned, expected=None):
    answers = unique(returned['answers'], 'blind_id')
    if expected is not None:
        assert set(answers) == set(expected)
    for row in answers.values():
        assert all((row[side]['label'] in ERROR | {'correct'} for side in ('a', 'b')))
    return answers
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.parse_args()
    main()
