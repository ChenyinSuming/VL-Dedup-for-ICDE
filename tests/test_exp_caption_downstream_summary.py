"""Small CPU-only tests use synthetic metrics; never open CAPTION_DOWNSTREAM_RUN performance."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('caption_summary', Path(__file__).parents[1] / 'scripts/experiments/summarize_caption_downstream.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def values(equal=False):
    output = {}
    for index, method in enumerate(summary.METHODS):
        output[method] = {}
        for metric in summary.METRICS:
            sign = -1 if metric.startswith('CHAIR') else 1
            output[method][metric] = [float(sign * ((0 if equal else index) + item)) for item in range(5)]
    return output


def test_single_by_family_and_clean_exclusion():
    data = values()
    result = summary.compute_statistics(data)
    assert result['by_family_size'] == 231
    assert len(result['contrasts']) == 11 * 7
    assert len(summary.contrasts()) == len(set(summary.contrasts())) == 11
    assert all(row['first'] != 'clean' and row['second'] != 'clean' for row in result['contrasts'])
    # Changing reference-only Clean values cannot change a single p-value.
    for metric in summary.METRICS:
        data['clean'][metric] = [99999.0] * 5
    assert summary.compute_statistics(data)['contrasts'] == result['contrasts']
    assert result['n'] == 5
    assert result['method_metrics']['no_dedup']['CIDEr']['sample_sd'] == pytest.approx(1.5811388300841898)


def test_chair_orientation_and_directional_flags():
    rows = summary.compute_statistics(values())['contrasts']
    for metric in summary.METRICS:
        row = next(r for r in rows if (r['first'], r['second'], r['metric']) == ('b_cov', 'no_dedup', metric))
        assert row['mean_oriented_difference'] == 6
        assert row['flag'] == 'P'
        assert row['raw_p'] == {'two-sided': 0, 'greater': 0, 'less': 1}
    data = values()
    data['b_cov']['CHAIR_i'] = [v + 20 for v in data['b_cov']['CHAIR_i']]
    row = next(r for r in summary.compute_statistics(data)['contrasts'] if
               (r['first'], r['second'], r['metric']) == ('b_cov', 'no_dedup', 'CHAIR_i'))
    assert row['flag'] == 'N'
    assert row['raw_p'] == {'two-sided': 0, 'greater': 1, 'less': 0}


def test_by_exact_small_family_and_t_test():
    assert summary.by_adjust([.001, .01, .04]) == pytest.approx([.0055, .0275, .07333333333333333])
    assert summary.by_adjust([.04, .001, .01]) == pytest.approx([.07333333333333333, .0055, .0275])
    p = summary.paired_pvalues([1, 2, 3, 4, 5])
    assert p['two-sided'] == pytest.approx(.013235599563682695)
    assert p['greater'] == pytest.approx(p['two-sided'] / 2)
    assert p['less'] == pytest.approx(1 - p['greater'])
    assert summary.compute_statistics(values(equal=True))['flag_counts'] == {'P': 0, 'N': 0, 'S': 77}
    assert summary.paired_pvalues([0] * 5) == {tail: 1 for tail in summary.TAILS}


def test_incomplete_does_not_read_performance_or_assign_flags(tmp_path):
    directory = tmp_path / 'training' / '20261104' / 'clean'
    directory.mkdir(parents=True)
    (directory / 'evaluation_complete.json').write_text('NOT JSON: must not read before campaign completion')
    protocol = {'run_id': 'synthetic-contract-only', 'task': 'CAPTION_DOWNSTREAM_RUN', 'sources': {'noisy_pool_run': '/synthetic/noisy_pool'}}
    for status in ['running', 'gate_stopped', 'completed']:
        result = summary.build_results(tmp_path, protocol, status)
        assert result['metrics'] is None and result['statistics'] is None
        assert result['status'] == ('artifact_incomplete' if status == 'completed' else status)


def test_completed_contract_writes_results_from_synthetic_receipts(tmp_path):
    import json
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    run = tmp_path / 'run'
    for name in ['gate1', 'gate2', 'stage_e_timing_gate']:
        put(run / (name + '.json'), {'status': 'passed'})
    for name in ['stage_e_complete', 'coverage_complete']:
        put(run / (name + '.json'), {'status': 'complete'})
    put(run / 'training_sets_complete.json', {'status': 'complete', 'composition': {m: {} for m in summary.METHODS}})
    put(run / 'training_campaign_complete.json', {'status': 'complete', 'subset_count': 5})
    put(run / 'master_snapshot_check.json', {'status': 'passed', 'byte_identical': True, 'before_lines': 23, 'final_lines': 23})
    put(run / 'caption_training_gate.json', {'passed': True, 'correctness_passed': True, 'subset_count': 5})
    data = values()
    for index, seed in enumerate(summary.SEEDS[:5]):
        for method in summary.METHODS:
            directory = run / 'training' / str(seed) / method
            put(directory / 'training_complete.json', {'seed': seed, 'method': method, 'clean_records': 400, 'steps': 100, 'train_seconds': 12})
            put(directory / 'evaluation_complete.json', {'seed': seed, 'method': method, 'test_images': 5000, 'eval_seconds': 3,
                'metrics': {metric: data[method][metric][index] for metric in summary.METRICS}})
    protocol = {'run_id': 'synthetic-only', 'task': 'CAPTION_DOWNSTREAM_RUN', 'sources': {'noisy_pool_run': '/synthetic/noisy_pool'}}
    result = summary.build_results(run, protocol, 'completed')
    assert result['status'] == 'completed'
    assert result['timing']['training_evaluation']['actual_train_seconds'] == 12 * 40
    report = tmp_path / 'report'
    summary.write_results(report, result)
    assert {p.name for p in report.iterdir()} == {'results.json'}
    assert json.loads((report / 'results.json').read_text())['statistics']['by_family_size'] == 231
    # Same-seed method budgets must agree even if each receipt is self-consistent.
    path = run / 'training' / str(summary.SEEDS[0]) / 'b_cov' / 'training_complete.json'
    changed = json.loads(path.read_text());changed.update(clean_records=800, steps=200);put(path, changed)
    stopped = summary.build_results(run, protocol, 'completed')
    assert stopped['status'] == 'artifact_incomplete' and stopped['metrics'] is None


@pytest.mark.parametrize('snapshot', [
    None,
    {'status': 'failed', 'byte_identical': True, 'before_lines': 23, 'final_lines': 23},
    {'status': 'passed', 'byte_identical': False, 'before_lines': 23, 'final_lines': 23},
    {'status': 'passed', 'byte_identical': True, 'before_lines': 22, 'final_lines': 23},
    {'status': 'passed', 'byte_identical': True, 'before_lines': 23, 'final_lines': 24},
])
def test_completed_rejects_invalid_master_snapshot_before_reading_performance(tmp_path, snapshot):
    import json
    (tmp_path / 'training_campaign_complete.json').write_text(json.dumps({'status': 'complete', 'subset_count': 5}))
    if snapshot is not None:
        (tmp_path / 'master_snapshot_check.json').write_text(json.dumps(snapshot))
    # A sentinel confirms protection is checked before any test-performance read.
    directory = tmp_path / 'training' / '20261104' / 'clean'
    directory.mkdir(parents=True)
    (directory / 'evaluation_complete.json').write_text('UNREADABLE PERFORMANCE SENTINEL')
    protocol = {'run_id': 'synthetic-only', 'task': 'CAPTION_DOWNSTREAM_RUN', 'sources': {'noisy_pool_run': '/synthetic/noisy_pool'}}
    result = summary.build_results(tmp_path, protocol, 'completed')
    assert result['status'] == 'artifact_incomplete'
    assert result['evidence_status'] == 'artifact-incomplete'
    assert result['metrics'] is None and result['statistics'] is None
    assert 'Master snapshot protection' in result['reason']
