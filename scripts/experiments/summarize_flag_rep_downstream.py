#!/usr/bin/env python3
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
"""FLAG_REP_RUN paired reporting: one 231-p main family and separate 63-p family."""
from pathlib import Path
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_flag_rep_export import RUN, REPORT, CAPTION_DOWNSTREAM_RUN, load, dump
from scripts.experiments.summarize_caption_downstream import METRICS, TAILS, paired_pvalues, by_adjust


def family(values, comparisons, expected):
    records = []
    for first, second in comparisons:
        for metric in METRICS:
            raw = [a - b for a, b in zip(values[first][metric], values[second][metric], strict=True)]
            differences = [(-v if metric in ('CHAIR_i', 'CHAIR_s') else v) for v in raw]
            records.append(dict(first=first, second=second, metric=metric,
                mean_difference=statistics.mean(raw), mean_oriented_difference=statistics.mean(differences),
                better_subsets=sum(v > 0 for v in differences), tied_subsets=sum(v == 0 for v in differences),
                paired_subsets=10, oriented_differences=differences, raw_p=paired_pvalues(differences)))
    pvalues = [r['raw_p'][tail] for r in records for tail in TAILS]
    if len(pvalues) != expected:
        raise ValueError('Frozen statistical family size differs')
    adjusted = iter(by_adjust(pvalues))
    for row in records:
        p = row['adjusted_p'] = {tail: next(adjusted) for tail in TAILS}
        row['flag'] = ('P' if p['two-sided'] < .05 and p['greater'] < .05 else
                       'N' if p['two-sided'] < .05 and p['less'] < .05 else 'S')
    counts = {v: sum(r['flag'] == v for r in records) for v in ('P', 'N', 'S')}
    by_comparison = {}
    for a, b in comparisons:
        by_comparison[f'{a} vs {b}'] = {v: sum(r['first'] == a and r['second'] == b and r['flag'] == v for r in records)
                                       for v in ('P', 'N', 'S')}
    return dict(by_family_size=expected, alpha=.05, contrasts=records, flag_counts=counts,
                flag_counts_by_comparison=by_comparison)


def main():
    if load(RUN / 'training_campaign_complete.json')['status'] != 'complete':
        raise RuntimeError('Complete all 40 new evaluations before reporting B3')
    protocol = load(CONFIG / 'flag_rep_export.json')
    new_methods = ('b_fr', 'b_fr_cov', 'clipscore_fr', 'random_fr')
    reused_methods = ('no_dedup', 'keep_one', 'semdedup', 'clean', 'b', 'b_cov', 'clipscore', 'random')
    values, receipts = {}, []
    for method in new_methods + reused_methods:
        origin = RUN if method in new_methods else CAPTION_DOWNSTREAM_RUN
        key = method if method in new_methods or method in ('no_dedup', 'keep_one', 'semdedup', 'clean') else 'caption_downstream_' + method
        values[key] = {metric: [] for metric in METRICS}
        for seed in range(20261104, 20261114):
            directory = origin / 'training' / str(seed) / method
            training = load(directory / 'training_complete.json')
            evaluation = load(directory / 'evaluation_complete.json')
            if training['steps'] != 4370 or evaluation['test_images'] != 5000 or evaluation['seed'] != seed:
                raise RuntimeError('Training/evaluation receipt contract differs')
            for metric in METRICS:
                values[key][metric].append(evaluation['metrics'][metric])
            receipts.append(dict(method=key, seed=seed, source=str(directory),
                                 train_seconds=training['train_seconds'], eval_seconds=evaluation['eval_seconds']))
    summary = {method: {metric: dict(n=10, mean=statistics.mean(vector), sample_sd=statistics.stdev(vector))
                       for metric, vector in metrics.items()} for method, metrics in values.items()}
    stat = protocol['downstream']['B3']
    result = dict(task='flag_rep', status='completed', n=10,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        method_metrics=summary, main_family=family(values, stat['main_contrasts'], 231),
        additional_family=family(values, stat['additional_contrasts'], 63),
        per_seed=values, receipts=receipts, training_sets=load(RUN / 'training_sets_complete.json'),
        retraining_gate=load(RUN / 'retraining_gate.json'), runtime_gate=load(RUN / 'runtime_gate.json'),
        scope='Ten paired overlapping subsets; clean test reporting only; Clean reference excluded from comparisons.')
    target = REPORT / 'downstream'
    dump(target / 'results.json', result)


if __name__ == '__main__':
    main()
