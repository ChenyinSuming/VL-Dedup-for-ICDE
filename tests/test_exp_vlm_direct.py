"""Frozen VLM_DIRECT_RUN identity, work partition, graph order and pairwise contracts."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiments import run_vlm_direct as direct
from scripts.run_graph_baselines import pairwise_counts


def test_identity_uses_visual_field_and_rejects_failures():
    row = dict(status='completed', raw_return=json.dumps({'same_visual_entity': 'yes'}),
               parsed=dict(contract_valid=True, decision='mismatch'))
    assert direct.identity(row)
    row['parsed']['contract_valid'] = False
    assert not direct.identity(row)
    row['parsed']['contract_valid'] = True
    row['status'] = 'failed'
    assert not direct.identity(row)
    row['status'] = 'completed'
    for raw in ('not JSON', '{}', '{"same_visual_entity":"uncertain"}', '{"same_visual_entity":"no"}'):
        row['raw_return'] = raw
        assert not direct.identity(row)


def test_reuse_and_new_cover_population_once():
    work = [dict(record_id_a=a, record_id_b=b) for a, b in [('b','a'),('a','c'),('c','b')]]
    reused, new = direct.partition_work(work, {('a','b'): {}, ('other','pair'): {}})
    assert reused == {('a','b')}
    assert {direct.key(r) for r in new} == {('a','c'),('b','c')}
    assert len(new)+len(reused) == len(work)
    with pytest.raises(ValueError, match='Duplicate'):
        direct.partition_work(work+[work[0]], {})


def test_vlm_k5_orders_by_image_score_then_ids_and_caps_components():
    a, b = np.array([4,3,1,0,2,5]), np.array([5,4,2,1,3,6])
    labels, order, retained, counts = direct.vlm_graph(7, a, b, [.8,.8,.9,.9,.9,.8], 5)
    assert order.tolist() == [3,2,4,1,0,5]
    assert retained.tolist() == [True,True,True,True,False,True]
    assert max(np.unique(labels, return_counts=True)[1]) == 5
    assert len(set(labels[:5])) == 1 and labels[5] == labels[6] != labels[4]
    assert counts['cap_rejected_edges'] == 1


def test_pairwise_counts_conserve_all_pairs():
    result = pairwise_counts([0,0,1,1,1], ['a','a','a','b','b'])
    assert [result[k] for k in ('tp','fp','fn','tn')] == [2,2,2,4]
    assert sum(result[k] for k in ('tp','fp','fn','tn')) == result['pairs'] == 10
    assert result['tp']+result['fn'] == result['true_positive_pairs'] == 4


def test_resume_preserves_stop_and_rejects_foreign_or_wrong_phase():
    a = [dict(record_id_a='a',record_id_b='b')]
    b = [dict(record_id_a='b',record_id_b='c')]
    direct.validate_new_history(a,b,[dict(a[0],phase='A')],dict(status='passed'))
    with pytest.raises(ValueError,match='time gate stopped'):
        direct.validate_new_history(a,b,[],dict(status='stopped'))
    for done in ([dict(record_id_a='x',record_id_b='y',phase='A')],
                 [dict(a[0],phase='B')], [dict(a[0],phase='A')]*2):
        with pytest.raises(ValueError,match='terminal identity'):
            direct.validate_new_history(a,b,done,None)
