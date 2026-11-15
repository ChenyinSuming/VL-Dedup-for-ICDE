#!/usr/bin/env python3
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
"""Read CAPTION_DOWNSTREAM_RUN receipts and write scoped reports; never run an experiment or hash files.

Performance receipts are opened only for --status completed, after the campaign,
Stage E, coverage and Clean gates independently authorize a complete campaign.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = (RESOURCES / 'runs/caption_downstream')
REPORT = ROOT / 'outputs/reports/caption_downstream'
METHODS = ('no_dedup', 'keep_one', 'semdedup', 'clipscore', 'random', 'b', 'b_cov', 'clean')
METRICS = ('CIDEr', 'BLEU-4', 'METEOR', 'SPICE', 'SPICE recall', 'CHAIR_i', 'CHAIR_s')
TAILS = ('two-sided', 'greater', 'less')
SEEDS = tuple(range(20261104, 20261114))


def load(path):
    return json.loads(Path(path).read_text())


def optional(run, name):
    path = run / name
    return load(path) if path.exists() else None


def contrasts():
    result = [(m, 'no_dedup') for m in METHODS[1:-1]]
    result += [('b_cov', m) for m in METHODS[1:-2]]
    assert len(result) == len(set(result)) == 11
    return result


def paired_pvalues(differences):
    if len(differences) < 2:
        raise ValueError('Paired inference requires at least two subsets')
    mean = statistics.mean(differences)
    deviation = statistics.stdev(differences)
    if deviation == 0:
        if mean == 0:
            return {tail: 1.0 for tail in TAILS}
        return {'two-sided': 0.0, 'greater': 0.0 if mean > 0 else 1.0,
                'less': 0.0 if mean < 0 else 1.0}
    from scipy.stats import t
    statistic = mean / (deviation / math.sqrt(len(differences)))
    return {'two-sided': float(2 * t.sf(abs(statistic), len(differences) - 1)),
            'greater': float(t.sf(statistic, len(differences) - 1)),
            'less': float(t.cdf(statistic, len(differences) - 1))}


def by_adjust(pvalues):
    """One Benjamini-Yekutieli family, in caller order."""
    size = len(pvalues)
    if not size or any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError('BY requires a nonempty family of finite probabilities')
    harmonic = sum(1 / rank for rank in range(1, size + 1))
    order = sorted(range(size), key=lambda i: pvalues[i])
    adjusted = [None] * size
    minimum = 1.0
    for index in range(size - 1, -1, -1):
        original = order[index]
        minimum = min(minimum, size * harmonic * pvalues[original] / (index + 1))
        adjusted[original] = minimum
    return adjusted


def compute_statistics(values):
    """values[method][metric] contains paired values in identical seed order."""
    if set(values) != set(METHODS):
        raise ValueError('Exactly the eight prescribed methods are required')
    n = len(values['no_dedup']['CIDEr'])
    if n not in (5, 10):
        raise ValueError('Only the prescribed first five or ten paired subsets are valid')
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in METRICS:
            vector = values[method][metric]
            if len(vector) != n or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                raise ValueError('Invalid or unpaired metric values')
            summary[method][metric] = {'n': n, 'mean': statistics.mean(vector), 'sample_sd': statistics.stdev(vector)}
    rows = []
    for first, second in contrasts():
        for metric in METRICS:
            direction = -1 if metric in ('CHAIR_i', 'CHAIR_s') else 1
            differences = [direction * (a - b) for a, b in zip(values[first][metric], values[second][metric])]
            rows.append({'first': first, 'second': second, 'metric': metric,
                         'oriented_differences': differences, 'mean_oriented_difference': statistics.mean(differences),
                         'raw_p': paired_pvalues(differences)})
    family = [row['raw_p'][tail] for row in rows for tail in TAILS]
    if len(family) != 231:
        raise ValueError('The single BY family must contain exactly 231 p-values')
    adjusted = iter(by_adjust(family))
    counts = {'P': 0, 'N': 0, 'S': 0}
    for row in rows:
        p = row['adjusted_p'] = {tail: next(adjusted) for tail in TAILS}
        row['flag'] = ('P' if p['two-sided'] < .05 and p['greater'] < .05 else
                       'N' if p['two-sided'] < .05 and p['less'] < .05 else 'S')
        counts[row['flag']] += 1
    return {'n': n, 'method_metrics': summary, 'contrasts': rows, 'flag_counts': counts,
            'by_family_size': len(family), 'alpha': .05,
            'scope': 'Paired overlapping training-image subsets; Clean is reference only and excluded from all contrasts'}


def finite_seconds(receipt, field):
    value = receipt.get(field) if receipt else None
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise ValueError('Invalid timing field: ' + field)
    return value


def collect_times(run, complete=False, seeds=()):
    stage = optional(run, 'stage_e_complete.json')
    if stage is None:
        stage = optional(run, 'progress.json')
    gate = optional(run, 'stage_e_timing_gate.json')
    coverage = optional(run, 'coverage_complete.json')
    train_gate = optional(run, 'caption_training_gate.json')
    times = {'stage_e': {'source': 'stage_e_complete.json' if (run / 'stage_e_complete.json').exists() else 'progress.json',
             'status': stage.get('status') if stage else 'not_started',
             'actual_load_seconds': finite_seconds(stage, 'load_seconds'),
             'actual_record_seconds': finite_seconds(stage, 'record_seconds'),
             'actual_pair_seconds': finite_seconds(stage, 'pair_seconds'),
             'actual_total_seconds': finite_seconds(stage, 'actual_stage_e_seconds'),
             'record_terminal': stage.get('record_terminal') if stage else None,
             'pair_terminal': stage.get('pair_terminal') if stage else None,
             'projected_seconds': finite_seconds(gate, 'projected_stage_e_seconds')},
             'coverage': {'status': coverage.get('status') if coverage else 'not_started',
                          'actual_seconds': finite_seconds(coverage, 'elapsed_seconds')},
             'training_evaluation': {'projected_seconds': finite_seconds(train_gate, 'projected_train_eval_seconds'),
                                     'runs': [], 'actual_train_seconds': None, 'actual_eval_seconds': None},
             'lock_wait_seconds': None}
    if complete:
        for seed in seeds:
            for method in METHODS:
                directory = run / 'training' / str(seed) / method
                train = load(directory / 'training_complete.json')
                evaluation = load(directory / 'evaluation_complete.json')
                row = {'seed': seed, 'method': method,
                       'train_seconds': finite_seconds(train, 'train_seconds'),
                       'eval_seconds_including_generation': finite_seconds(evaluation, 'eval_seconds')}
                scorer_path = directory / 'metrics.json'
                if scorer_path.exists():
                    scorer = load(scorer_path)
                    row['official_cpu_scorer_seconds'] = scorer.get('timings_seconds')
                times['training_evaluation']['runs'].append(row)
        runs = times['training_evaluation']['runs']
        for field, key in [('actual_train_seconds', 'train_seconds'), ('actual_eval_seconds', 'eval_seconds_including_generation')]:
            if all(row[key] is not None for row in runs):
                times['training_evaluation'][field] = sum(row[key] for row in runs)
    else:
        # Never open evaluation_complete.json or metrics.json before completion.
        for path in sorted((run / 'training').glob('*/*/training_complete.json')):
            receipt = load(path)
            times['training_evaluation']['runs'].append({'seed': receipt.get('seed'), 'method': receipt.get('method'),
                                                        'train_seconds': finite_seconds(receipt, 'train_seconds')})
    return times


def complete_values(run, gates):
    marker = optional(run, 'training_campaign_complete.json')
    if not marker or marker.get('status') != 'complete':
        raise ValueError('Authoritative training_campaign_complete.json status=complete is required')
    snapshot = optional(run, 'master_snapshot_check.json')
    if (not snapshot or snapshot.get('status') != 'passed'
            or snapshot.get('byte_identical') is not True
            or type(snapshot.get('before_lines')) is not int or snapshot['before_lines'] != 23
            or type(snapshot.get('final_lines')) is not int or snapshot['final_lines'] != 23):
        raise ValueError('Master snapshot protection requires passed, byte_identical=true and 23 before/final lines')
    count = marker.get('subset_count')
    if type(count) is not int or count not in (5, 10):
        raise ValueError('Campaign subset_count must be 5 or 10')
    for name in ['gate1', 'gate2', 'stage_e_timing_gate']:
        if not gates[name] or gates[name].get('status') != 'passed':
            raise ValueError(name + ' has not passed')
    for name in ['stage_e_complete', 'coverage_complete', 'training_sets_complete']:
        receipt = optional(run, name + '.json')
        if not receipt or receipt.get('status') != 'complete':
            raise ValueError(name + ' has not completed')
    gate = optional(run, 'caption_training_gate.json')
    if not gate or gate.get('passed') is not True or gate.get('correctness_passed') is not True or gate.get('subset_count') != count:
        raise ValueError('Clean correctness and training-time gate must pass with matching subset_count')
    seeds = SEEDS[:count]
    if marker.get('seeds', list(seeds)) != list(seeds):
        raise ValueError('Campaign seeds must be the prescribed first subsets')
    paths = [run / 'training' / str(seed) / method / filename
             for seed in seeds for method in METHODS
             for filename in ('training_complete.json', 'evaluation_complete.json')]
    if any(not path.is_file() for path in paths):
        raise ValueError('Campaign has missing per-method training/evaluation completion receipts')
    values = {method: {metric: [] for metric in METRICS} for method in METHODS}
    for seed in seeds:
        clean_count = None
        for method in METHODS:
            directory = run / 'training' / str(seed) / method
            train = load(directory / 'training_complete.json')
            receipt = load(directory / 'evaluation_complete.json')
            for item in [train, receipt]:
                if item.get('seed') != seed or item.get('method') != method:
                    raise ValueError('Receipt method or seed identity mismatch')
            if receipt.get('test_images') != 5000:
                raise ValueError('Every evaluation requires the fixed test5000 denominator')
            if clean_count is None:
                clean_count = train['clean_records']
            if train['clean_records'] != clean_count:
                raise ValueError('Methods on the same subset must share the Clean step budget')
            if train.get('steps') != 10 * (train['clean_records'] // 40):
                raise ValueError('Training receipt violates the prescribed Clean step budget')
            for metric in METRICS:
                value = receipt['metrics'].get(metric)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('Missing or nonfinite required metric: ' + metric)
                values[method][metric].append(value)
    if values['clean']['CIDEr'][0] < .4:
        raise ValueError('Clean subset1 CIDEr correctness gate failed')
    return values, seeds


def build_results(run, protocol, status, reason=None):
    gates = {name: optional(run, name + '.json') for name in ['gate1', 'gate2', 'stage_e_timing_gate']}
    clean_gate = optional(run, 'caption_training_gate.json')
    # Gate state is public before performance reporting; do not expose Clean test metrics here.
    gates['caption_training_gate'] = ({k: clean_gate.get(k) for k in ['passed', 'correctness_passed', 'subset_count', 'projected_train_eval_seconds']}
                                      if clean_gate else None)
    composition = optional(run, 'training_sets_complete.json')
    snapshot = optional(run, 'master_snapshot_check.json')
    result = {'run_id': protocol['run_id'], 'task': protocol['task'], 'status': status,
              'requested_status': status, 
              'source_contract': {'corpus': 'Immutable NOISY_POOL_RUN B corpus, caches and original Karpathy membership',
                                  'noisy_pool_run': protocol['sources']['noisy_pool_run'], 'test_images': 5000,
                                  'test_use': 'Reporting only, except prescribed Clean CIDEr correctness gate'},
              'hash_fields': 'N/A: user prohibits new file hashes',
              'manifest_pairs_hashes': 'N/A', 'model_verdict_hashes': 'N/A',
              'gates': gates, 'master_snapshot_check': snapshot,
              'metrics': None, 'statistics': None, 'reason': reason,
              'composition': composition.get('composition') if composition and composition.get('status') == 'complete' else None,
              'evidence_status': 'executing' if status == 'running' else 'scope_stopped',
              'generated_utc': datetime.now(timezone.utc).isoformat()}
    seeds = ()
    if status == 'completed':
        try:
            values, seeds = complete_values(run, gates)
            stats = compute_statistics(values)
            result.update(metrics=stats.pop('method_metrics'), statistics=stats, subset_count=len(seeds),
                          subset_seeds=list(seeds), paired_values=values, evidence_status='archived-complete')
            if not result['composition'] or set(result['composition']) != set(METHODS):
                raise ValueError('All eight method composition entries are required')
        except (ValueError, KeyError, TypeError) as exc:
            result.update(status='artifact_incomplete', reason=str(exc), metrics=None, statistics=None,
                          paired_values=None, evidence_status='artifact-incomplete')
            seeds = ()
    elif not reason:
        result['reason'] = ('Campaign still running; downstream metrics and significance have not been reported' if status == 'running'
                            else 'Dependent work stopped at a correctness or time gate; inspect named gate receipts')
    result['timing'] = collect_times(run, result['status'] == 'completed', seeds)
    return result


def write_results(report, result):
    report.mkdir(parents=True, exist_ok=True)
    (report / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--protocol', type=Path, default=CONFIG / 'caption_downstream.json')
    parser.add_argument('--status', choices=['running', 'completed', 'gate_stopped'], required=True)
    parser.add_argument('--reason')
    args = parser.parse_args()
    if args.report.resolve() == args.run.resolve() or args.run.resolve() in args.report.resolve().parents:
        raise ValueError('Report directory must be separate from immutable run receipts')
    protocol = load(args.protocol)
    result = build_results(args.run, protocol, args.status, args.reason)
    result['implementation_commit'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
    result['protocol_path'] = str(args.protocol.resolve())
    result['run_path'] = str(args.run.resolve())
    write_results(args.report, result)
    print(json.dumps({'status': result['status'], 'run_id': result['run_id'], 'report': str(args.report), 'reason': result.get('reason')}))
    return 1 if result['status'] == 'artifact_incomplete' else 0


if __name__ == '__main__':
    raise SystemExit(main())
