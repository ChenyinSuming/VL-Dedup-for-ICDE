import numpy as np
import pytest
from scripts import score_caption_mismatches as m

def test_yes_no_token_variants():
    tokens = [dict(text=t, logprob=float(np.log(p)), rank=i) for (i, (t, p)) in enumerate([('yes', 0.2), (' Yes', 0.1), ('YES', 0.1), (' no', 0.2), ('No', 0.1), ('yesterday', 0.2), ('yes.', 0.1)], 1)]
    result = m.yes_probability(tokens)
    assert result['valid'] and result['probability'] == pytest.approx(4 / 7)
    assert m.yes_probability([dict(text=' yes', logprob=-1, rank=21)])['probability'] is None

def test_invalid_and_single_class():
    assert m.yes_probability([dict(text='Sure', logprob=-0.1, rank=1)])['valid'] is False
    assert m.effective_probability(None) == 1.0
    assert m.yes_probability([dict(text='no', logprob=-2, rank=1)])['probability'] == 0
    assert m.yes_probability([dict(text='yes', logprob=-2, rank=1)])['probability'] == 1
    pairs = [dict(record_id_a='a', record_id_b='b')]
    assert m.pair_scores(pairs, {'a': 1.0, 'b': 1.0}) == [1.0]
    assert m.pair_scores(pairs, {'a': 1.0, 'b': 0.1}) == [0.1]

def test_component_scores():
    (rel, cross, text) = m.component_scores([0.2, 0.4, 0.9], [[1, 0], [0, 1], [1, 0]], [[1, 0], [0, 1], [1, 0]], [[1, 0], [0, 1], [1, 0]])
    np.testing.assert_allclose(rel, [-0.2, 0, 0.5])
    np.testing.assert_allclose(cross, [2 / 3, 1 / 3, 2 / 3])
    np.testing.assert_allclose(text, [0.5, 0, 0.5])

def test_validation_percentile_ties_and_test_extension():
    np.testing.assert_allclose(m.percentile([0.1, 0.2, 0.2, 0.4], [0, 0.1, 0.2, 0.3, 0.4, 1]), [0, 0.125, 0.5, 0.75, 0.875, 1])
