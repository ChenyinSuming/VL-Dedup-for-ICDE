"""Small CPU identity, selection, and reporting contracts."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('caption_sets', Path(__file__).parents[1] / 'scripts/experiments/run_caption_downstream_sets.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_exact_graph_counts_and_clip_representative():
    assert runner.graph_counts([0, 0, 0, 3], [1, 1, 2, 2]) == {
        'TP': 1, 'predicted_positive': 3, 'truth_positive': 2, 'exact_F1': '2/5'}
    assert runner.representatives(['c', 'a', 'b'], [0, 0, 1], {'a': .5, 'b': .2, 'c': .5}) == ['a', 'b']


def test_state_coverage_requires_complete_disjoint_join():
    exports = [{'record_ids': ['a', 'b'], 'b': {'retained': ['a'], 'pending': [], 'deleted': ['b']}}]
    assert runner.arm_members(exports, 'b', ['a', 'b'])['deleted'] == ['b']
    with pytest.raises(ValueError, match='global identity'):
        runner.arm_members(exports, 'b', ['a', 'b', 'c'])
    exports[0]['b']['pending'] = ['b']
    with pytest.raises(ValueError, match='conservation'):
        runner.arm_members(exports, 'b', ['a', 'b'])


def test_completion_and_threshold_dependency_gate(tmp_path):
    with pytest.raises(RuntimeError, match='missing'):
        runner.require_complete(tmp_path, 'stage_e_complete.json')
    runner.dump(tmp_path / 'stage_e_complete.json', {'status': 'running'})
    with pytest.raises(RuntimeError, match='not complete'):
        runner.require_complete(tmp_path, 'stage_e_complete.json')
    runner.dump(tmp_path / 'stage_e_complete.json', {'status': 'complete'})
    runner.require_complete(tmp_path, 'stage_e_complete.json')


def test_reporting_zero_caption_images_and_original_removal():
    frame = pd.DataFrame([
        dict(record_id='a', original_image_id=1, image_path='x', is_copy=False, caption_swapped=False),
        dict(record_id='b', original_image_id=1, image_path='y', is_copy=True, caption_swapped=True),
        dict(record_id='c', original_image_id=2, image_path='z', is_copy=False, caption_swapped=False)])
    report = runner.composition(frame, {'b'}, [1, 2, 3])
    assert report['captions_per_original_image_histogram'] == {'0': 2, '1': 1}
    assert report['retained_copies'] == report['retained_swapped'] == 1
    assert report['removed_original_noncopy_nonswapped'] == 2


def test_test_images_reuse_r52_then_extract_only_missing(tmp_path, monkeypatch):
    import zipfile
    original = tmp_path / 'existing.jpg'
    original.write_bytes(b'existing')
    frame = pd.DataFrame([{'image_id': 1, 'image_path': str(original)}])
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **k: frame)
    zip_class = zipfile.ZipFile
    for partition in ('train2017', 'val2017'):
        with zip_class(tmp_path / f'{partition}.zip', 'w') as archive:
            if partition == 'train2017':
                archive.writestr('train2017/000000000002.jpg', b'missing-only')
    monkeypatch.setattr(zipfile, 'ZipFile', lambda path: zip_class(tmp_path / Path(path).name))
    run = tmp_path / 'run'
    resolved = runner.resolve_test_images(run, [{'cocoid': 1}, {'cocoid': 2}])
    assert resolved[1] == original
    assert resolved[2].read_bytes() == b'missing-only'
    assert [p.name for p in (run / 'test_images').iterdir()] == ['000000000002.jpg']


def test_completed_test_membership_reuses_without_loading_sources(tmp_path):
    runner.dump(tmp_path / 'test_membership_complete.json', {'status': 'complete', 'images': 5000})
    (tmp_path / 'test.parquet').write_bytes(b'already-produced')
    runner.produce_test(tmp_path, tmp_path / 'absent-karpathy.json')
