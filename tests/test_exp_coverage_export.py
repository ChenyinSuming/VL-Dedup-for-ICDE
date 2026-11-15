import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('coverage_export', Path(__file__).resolve().parents[1] / 'scripts/experiments/run_coverage_export.py')
coverage_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage_export)


def test_no_deletions():
    original = dict(retained=['a'], pending=['h'], deleted=[])
    def unexpected(*args):
        raise AssertionError('No check needed')
    assert coverage_export.coverage_export(original, {}, unexpected) == original


def test_all_covered_preserves_b():
    original = dict(retained=['a'], pending=['h'], deleted=['b', 'c'])
    assert coverage_export.coverage_export(original, {'b': 2, 'c': 1}, lambda *_: True) == original


def test_iterative_reference_grows_and_held_unchanged():
    original = dict(retained=['a'], pending=['h'], deleted=['b', 'c'])
    seen = []
    def check(rid, retained):
        seen.append((rid, list(retained)))
        return 'b' in retained
    result = coverage_export.coverage_export(original, {'b': 2, 'c': 1}, check)
    assert seen == [('b', ['a']), ('c', ['a', 'b'])]
    assert result == dict(retained=['a', 'b'], pending=['h'], deleted=['c'])
    assert original == dict(retained=['a'], pending=['h'], deleted=['b', 'c'])


def test_noniterative_does_not_extend_reference():
    original = dict(retained=['a'], pending=['h'], deleted=['b', 'c'])
    seen = []
    def check(rid, retained):
        seen.append(retained)
        return False
    result = coverage_export.coverage_export(original, {'b': 2, 'c': 1}, check, iterative=False)
    assert seen == [['a'], ['a']]
    assert result == dict(retained=['a', 'b', 'c'], pending=['h'], deleted=[])


@pytest.mark.parametrize('answer', [False, None])
def test_false_or_null_is_retained(answer):
    original = dict(retained=['a'], pending=[], deleted=['b'])
    assert coverage_export.coverage_export(original, {'b': 1}, lambda *_: answer)['retained'] == ['a', 'b']


def test_clip_order_and_record_id_tie():
    original = dict(retained=['a'], pending=[], deleted=['d', 'c', 'b'])
    seen = []
    def check(rid, retained):
        seen.append(rid)
        return True
    coverage_export.coverage_export(original, {'b': 1, 'c': 2, 'd': 2}, check)
    assert seen == ['c', 'd', 'b']


def test_r61_parser_invalid_is_conservative():
    parsed = coverage_export.coverage_check.parse_output('{"covered":true,"reason":"yes"}', 'length')
    assert not parsed['valid'] and parsed['covered'] is None
    original = dict(retained=['a'], pending=['h'], deleted=['b'])
    result = coverage_export.coverage_export(original, {'b': 1}, lambda *_: parsed['covered'])
    assert result == dict(retained=['a', 'b'], pending=['h'], deleted=[])


def test_dual_ai_ranges_keep_null_separate():
    votes = [[True, True], [True, False], [False, False], [None, True], [None, False], [None, None]]
    assert coverage_export.aggregate_votes(votes) == dict(deleted=6, covered_strict=1, covered_loose=3,
        uncovered_strict=1, uncovered_loose=3, any_null=3)
