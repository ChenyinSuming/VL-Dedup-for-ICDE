import itertools
import pytest
from scripts import run_audit_ranges as audit_ranges

def answer(a, b):
    return {'a': {'label': a}, 'b': {'label': b}}

def test_caption_errors_all_label_combinations():
    for labels in itertools.product(('correct', 'local_error', 'whole_mismatch', 'unsure'), repeat=4):
        (first, second) = (answer(*labels[:2]), answer(*labels[2:]))
        errors = [x in {'local_error', 'whole_mismatch'} for x in labels]
        assert audit_ranges.caption_errors_error(first, second, 'strict') == (errors[0] and errors[2] or (errors[1] and errors[3]))
        assert audit_ranges.caption_errors_error(first, second, 'loose') == any(errors)
        for frozen_value in (False, True):
            assert audit_ranges.caption_errors_error(first, second, 'protocol', frozen_value) is frozen_value

@pytest.mark.parametrize('labels,expected', [(('correct', 'correct'), False), (('correct', 'local_error'), True), (('whole_mismatch', 'correct'), True), (('local_error', 'whole_mismatch'), True)])
def test_protocol_human_any_noncorrect(labels, expected):
    assert audit_ranges.human_error(answer(*labels)) is expected

@pytest.mark.parametrize('first,second', list(itertools.product(('lost', 'uncertain', 'covered_single', 'covered_joint'), repeat=2)))
def test_retained_set_lost_rules(first, second):
    assert audit_ranges.retained_set_lost(first, second, 'strict') == (first == 'lost' and second == 'lost')
    assert audit_ranges.retained_set_lost(first, second, 'loose') == (first == 'lost' or second == 'lost')

def test_combined_uses_validation_reference_and_midranks():
    records = [{'record_id': str(i), 'partition': 'validation' if i < 2 else 'test'} for i in range(4)]
    clip = {str(i): {'clipscore': c} for (i, c) in enumerate([0, 2, 0, -1])}
    prob = [{'record_id': str(i), 'probability': p, 'valid': valid} for (i, (p, valid)) in enumerate([(0.2, True), (0.8, True), (0.2, True), (None, False)])]
    (scores, n) = audit_ranges.combined_scores(records, clip, prob)
    assert n == 2
    assert scores == {'0': 0.25, '1': 0.75, '2': 0.25, '3': 0.5}

@pytest.mark.parametrize('scores,expected', [({'a': 0.06, 'b': 0.5}, True), ({'a': 0.05, 'b': 0.5}, False), ({'a': 0.5, 'b': 0.05}, False), ({'a': 0.059, 'b': 0.5}, True)])
def test_combined_min_endpoint_strict_boundary(scores, expected):
    assert audit_ranges.in_combined({'record_id_a': 'a', 'record_id_b': 'b'}, scores, 0.059) is expected
