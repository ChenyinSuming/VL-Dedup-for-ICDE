"""Checks of the released export, retained-set audit and downstream results."""
from pathlib import Path

import pandas as pd

from scripts import retained_set_export

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
RESULTS = ROOT / 'results'


def test_export_rebuilds_from_released_scores_verdicts_and_coverage_answers():
    released = pd.read_csv(ART / 'exports' / 'retained_set_vl_dedup.csv.gz').set_index('record_id').status
    rebuilt = retained_set_export.rebuild()
    assert len(rebuilt) == 38102
    assert all(released[record] == status for record, status in rebuilt.items())


def test_coverage_check_keeps_uncovered_records_for_later_checks():
    clip = {'a': 0.9, 'b': 0.8, 'c': 0.7}
    arm = {'retained': ['a'], 'held': [], 'deleted': ['b', 'c']}
    seen = []

    def covered(record, retained):
        seen.append((record, tuple(retained)))
        return record == 'c'

    result = retained_set_export.coverage_check(arm, clip, covered)
    assert seen == [('b', ('a',)), ('c', ('a', 'b'))]
    assert result == {'retained': ['a', 'b'], 'held': [], 'deleted': ['c']}


def test_exact_repeats_keep_the_highest_clipscore():
    clip = {'a': 0.2, 'b': 0.9, 'c': 0.5}
    text = {'a': 'A dog.', 'b': 'a  dog', 'c': 'a cat'}
    arm = {'retained': ['a', 'b', 'c'], 'held': [], 'deleted': []}
    assert retained_set_export.delete_exact_repeats(arm, clip, text) == {
        'retained': ['b', 'c'], 'held': [], 'deleted': ['a']}


def test_retained_set_summary_matches_audit_labels():
    summary = pd.read_csv(RESULTS / 'retained_set_summary.csv').set_index(['policy', 'stratum'])
    deletions = pd.read_csv(ART / 'audits' / 'retained_set_deletions.csv')
    for policy in ('keep_one', 'vl_dedup', 'vl_dedup_no_check'):
        rows = deletions[deletions.export == policy]
        lost = (rows.auditor_1 == 'lost').astype(int) + (rows.auditor_2 == 'lost').astype(int)
        assert len(rows) == summary.loc[(policy, 'all'), 'deleted']
        assert (lost == 2).sum() == summary.loc[(policy, 'all'), 'lost_strict']
        assert (lost >= 1).sum() == summary.loc[(policy, 'all'), 'lost_lenient']
    assert tuple(summary.loc[('vl_dedup', 'all'), ['retained', 'held', 'deleted']]) == (714, 148, 91)


def test_downstream_compositions_follow_from_training_sets():
    pool = pd.read_csv(ART / 'downstream' / 'pool_train.csv.gz').set_index('record_id')
    table = pd.read_csv(RESULTS / 'downstream.csv').set_index('retained_set')
    for name, row in table.iterrows():
        ids = pd.read_csv(ART / 'downstream' / 'training_sets' / f'{name}.csv.gz').record_id
        chosen = pool.loc[ids]
        assert len(chosen) == row.records
        assert chosen.is_copy.sum() == row.copies
        assert chosen.caption_swapped.sum() == row.swapped
