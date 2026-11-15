"""CPU contract checks; never import torch or load caption models."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('caption_training', Path(__file__).parents[1] / 'scripts/experiments/run_caption_downstream_training.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_frozen_membership_and_identical_method_budget():
    a = runner.subset_memberships(range(5000))
    b = runner.subset_memberships(reversed(range(5000)))
    assert a == b
    assert len(a['subsets']) == 10
    assert all(len(s['image_ids']) == len(set(s['image_ids'])) == 3500 for s in a['subsets'])
    assert len({tuple(s['image_ids']) for s in a['subsets']}) == 10
    assert runner.prescribed_steps(17503) == 4370
    with pytest.raises(ValueError, match='5000'):
        runner.subset_memberships(range(4999))


def test_predeclared_timing_gates():
    assert runner.timing_decision(1000, 100)['subset_count'] == 10
    reduced = runner.timing_decision(1500, 100)
    assert reduced['subset_count'] == 5 and reduced['passed']
    assert not runner.timing_decision(3000, 300)['passed']


def test_gpu_lease_before_model_import(monkeypatch):
    monkeypatch.delenv('VL_DEDUP_GPU_LEASE', raising=False)
    with pytest.raises(RuntimeError, match='VL_DEDUP_GPU_LEASE'):
        runner.runtime(None)


def test_dependent_runs_enforce_clean_gate_and_reduction(tmp_path):
    runner.dependent_gate(tmp_path, runner.SEEDS[0], 'clean')
    with pytest.raises(RuntimeError, match='Clean subset1'):
        runner.dependent_gate(tmp_path, runner.SEEDS[0], 'no_dedup')
    runner.write_json(tmp_path / 'caption_training_gate.json', {
        'passed': True, 'correctness_passed': True, 'subset_count': 5})
    runner.dependent_gate(tmp_path, runner.SEEDS[4], 'b_cov')
    with pytest.raises(RuntimeError, match='excluded'):
        runner.dependent_gate(tmp_path, runner.SEEDS[5], 'clean')


def test_combined_projection_also_reduces_before_failure():
    reduced = runner.timing_decision(1000, 600)
    assert reduced['subset_count'] == 5 and reduced['passed']
    assert reduced['reduction_reason'] == 'train_eval_80_runs_exceeds_35_hours'
    assert runner.timing_decision(1500, 100)['reduction_reason'] == 'training_80_runs_exceeds_30_hours'
    assert runner.timing_decision(1000, 0)['reduction_reason'] is None
    assert not runner.timing_decision(1000, 2200)['passed']


def test_actual_runtime_gate_counts_each_receipt_once(tmp_path):
    directory = tmp_path / 'training' / str(runner.SEEDS[0]) / 'clean'
    runner.write_json(directory / 'training_complete.json', {'train_seconds': 90000})
    runner.write_json(directory / 'evaluation_complete.json', {'eval_seconds': 1000})
    assert runner.actual_runtime_gate(tmp_path)['actual_seconds'] == 91000
    assert runner.actual_runtime_gate(tmp_path)['actual_seconds'] == 91000
    other = tmp_path / 'training' / str(runner.SEEDS[0]) / 'no_dedup'
    runner.write_json(other / 'training_complete.json', {'train_seconds': 36000})
    with pytest.raises(RuntimeError, match='35 GPU hours'):
        runner.actual_runtime_gate(tmp_path)
    import json
    gate = json.loads((tmp_path / 'caption_runtime_gate.json').read_text())
    assert gate == {'status': 'failed', 'actual_seconds': 127000, 'limit_seconds': 126000}

def test_completed_predictions_resume_without_loading_model(tmp_path, monkeypatch):
    import json
    import pandas as pd
    # Import the runner directly, independently of the other test fixtures.
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('caption_downstream_resume', Path(__file__).resolve().parents[1] / 'scripts/experiments/run_caption_downstream_training.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    directory = tmp_path / 'training/20261104/clean'
    directory.mkdir(parents=True)
    (directory / 'training_complete.json').write_text('{}')
    (directory / 'evaluation_progress.json').write_text(json.dumps({'elapsed_seconds': 10.}))
    (directory / 'predictions.json').write_text(json.dumps([
        {'image_id': i, 'caption': 'a saved caption'} for i in range(5000)]))
    pd.DataFrame({'original_image_id': range(5000), 'image_path': ['unused']*5000}).to_parquet(tmp_path / 'test.parquet')
    monkeypatch.setenv('VL_DEDUP_GPU_LEASE', '1')
    def forbidden(*args):
        raise AssertionError('Completed predictions must not reload a model')
    monkeypatch.setattr(runner, 'runtime', forbidden)
    def score(argv, check):
        path = Path(argv[argv.index('--output')+1])
        path.write_text(json.dumps({'status': 'complete', 'metrics': {m: .5 for m in runner.METRICS}}))
    monkeypatch.setattr(runner.subprocess, 'run', score)
    args = type('Args', (), dict(run=str(tmp_path), evaluator='fake.py', evaluator_args='[]', python='python'))()
    result = runner.generate(args, {'seed': 20261104}, 'clean')
    assert result['test_images'] == 5000
    assert result['eval_seconds'] >= 10.
