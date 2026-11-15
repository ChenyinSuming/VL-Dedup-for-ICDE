"""Frozen-score flag-aware representative and matched-budget contracts."""
import itertools

import pytest

from scripts.experiments.run_flag_rep_export import flag_rep, matched_controls, validate_arm
from scripts.experiments.support import export_cluster
from scripts.run_combined_actions import pair_action


def decisions(ids, value='duplicate'):
    return dict.fromkeys(itertools.combinations(sorted(ids), 2), value)


@pytest.mark.parametrize('judgment', ['duplicate', 'keep_augment', 'mismatch', 'non_duplicate'])
def test_unflagged_representative_exactly_replays_b(judgment):
    ids = ['a', 'b', 'c']
    clips, scores = dict(a=1, b=.5, c=.2), dict(a=.8, b=.7, c=.1)
    frozen = decisions(ids, judgment)
    actions = {p: pair_action(p, scores, v, .5) for p, v in frozen.items()}
    rep, result = flag_rep(ids, clips, scores, frozen, .5)
    assert rep == 'a'
    assert result == export_cluster(ids, 'a', actions)['b']
    assert 'c' in result['pending']


def test_flagged_representative_replaced_with_clip_then_id_tie():
    ids = ['c', 'b', 'a']
    rep, result = flag_rep(ids, dict(a=1, b=.7, c=.7), dict(a=.1, b=.5, c=.6), decisions(ids), .5)
    assert rep == 'b'
    assert result == dict(retained=['b'], pending=['a', 'c'], deleted=[])
    validate_arm(ids, result)


def test_all_flagged_cluster_held_without_representative():
    rep, result = flag_rep(['a', 'b'], dict(a=1, b=2), dict(a=.1, b=.2), decisions(['a', 'b']), .5)
    assert rep is None
    assert result == dict(retained=[], pending=['a', 'b'], deleted=[])


@pytest.mark.parametrize('score,retained,pending', [(.1, [], ['a']), (.5, ['a'], [])])
def test_singleton_strict_flag(score, retained, pending):
    _, result = flag_rep(['a'], dict(a=.9), dict(a=score), {}, .5)
    assert result == dict(retained=retained, pending=pending, deleted=[])
    validate_arm(['a'], result)


def test_missing_pair_stops():
    with pytest.raises(KeyError):
        flag_rep(['a', 'b'], dict(a=1, b=2), dict(a=1, b=1), {}, .5)


def test_states_disjoint_and_cover_records():
    for bits in itertools.product([.1, .5], repeat=3):
        ids = ['a', 'b', 'c']
        _, arm = flag_rep(ids, dict(a=3, b=2, c=1), dict(zip(ids, bits)), decisions(ids), .5)
        validate_arm(ids, arm)
        assert set(arm['pending']) >= {r for r, v in zip(ids, bits) if v < .5}


def test_controls_match_deleted_plus_held_budget():
    ids = ['a', 'b', 'c', 'd', 'e']
    b_fr_cov = dict(retained=['b', 'c', 'd'], pending=['a'], deleted=['e'])
    budget = len(b_fr_cov['pending']) + len(b_fr_cov['deleted'])
    controls = matched_controls(ids, {r: i for i, r in enumerate(ids)}, budget)
    assert controls['clipscore_fr'] == ['c', 'd', 'e']
    assert controls == matched_controls(list(reversed(ids)), {r: i for i, r in enumerate(ids)}, budget)
    assert all(len(ids) - len(kept) == budget for kept in controls.values())
