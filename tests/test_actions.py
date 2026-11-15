import itertools
import pytest
from scripts import run_combined_actions as m

def test_train_percentile_uses_validation_reference():
    clip = dict(validation_a=0.1, validation_b=0.3, t1=0.2, t2=0.5)
    probabilities = {r: dict(probability=p, valid=True) for (r, p) in dict(validation_a=0.2, validation_b=0.6, t1=0.4, t2=0.8).items()}
    scores = m.combined_scores(clip, probabilities, ['validation_a', 'validation_b'])
    assert scores['t1'] == 0.5
    assert scores['t2'] == 1.0
    clip['t3'] = 0.15
    probabilities['t3'] = dict(probability=0.3, valid=True)
    assert m.combined_scores(clip, probabilities, ['validation_a', 'validation_b'])['t1'] == scores['t1']

def test_representative_uses_clipscore_then_id():
    ids = ['a', 'b', 'c']
    (clip, combined) = (dict(a=0.8, b=0.9, c=0.9), dict(a=0.9, b=0.2, c=0.8))
    decisions = {p: 'duplicate' for p in itertools.combinations(ids, 2)}
    (representative, result) = m.export(ids, clip, combined, decisions)
    assert representative == 'b'
    assert result['b']['retained'] == ['b']

def test_vlm_review_only_at_or_above_threshold_and_low_augment_blocked():
    pair = ('a', 'b')
    assert m.classify_pair(pair, dict(a=m.TF, b=0.8), 'mismatch') == 'vlm_mismatch_review'
    assert m.classify_pair(pair, dict(a=m.TF / 2, b=0.8), 'mismatch') == 'score_mismatch'
    assert m.classify_pair(pair, dict(a=m.TF / 2, b=0.8), 'keep_augment') == 'score_mismatch'
    (_, result) = m.export(['a', 'b'], dict(a=0.9, b=0.8), dict(a=0.8, b=m.TF / 2), {pair: 'keep_augment'})
    assert result['b'] == dict(retained=['a'], held=['b'], deleted=[])

def test_export_closure_held_any_member_and_conservation():
    ids = ['a', 'b', 'c', 'd', 'e', 'f']
    clip = dict(a=0.9, b=0.8, c=0.7, d=0.6, e=0.5, f=0.4)
    combined = dict(a=0.8, b=0.8, c=0.8, d=0.8, e=0.8, f=0.01)
    decisions = {p: 'duplicate' for p in itertools.combinations(ids, 2)}
    decisions.update({('a', 'b'): 'keep_augment', ('b', 'c'): 'keep_augment', ('d', 'e'): 'mismatch'})
    (_, result) = m.export(ids, clip, combined, decisions)
    assert result['b'] == dict(retained=['a', 'b', 'c'], held=['d', 'e', 'f'], deleted=[])
    for arm in result.values():
        sets = [set(arm[k]) for k in ('retained', 'held', 'deleted')]
        assert set.union(*sets) == set(ids)
        assert sum(map(len, sets)) == len(ids)
    (_, single) = m.export(['a'], clip, combined, {})
    assert single['b'] == dict(retained=['a'], held=[], deleted=[])

def test_invalid_probability_kept_in_combined_denominator():
    clip = dict(validation_a=0.1, validation_b=0.3, train=0.2)
    p = dict(validation_a=dict(probability=0.2, valid=True), validation_b=dict(probability=0.6, valid=True), train=dict(probability=None, valid=False))
    assert m.combined_scores(clip, p, ['validation_a', 'validation_b'])['train'] == pytest.approx(0.75)
