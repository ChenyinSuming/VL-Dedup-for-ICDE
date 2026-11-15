"""Small independently checkable contracts for the K_RULES_RUN diagnostic."""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiments import run_k_rules as k
from scripts.cluster_with_size_cap import build_graph


@pytest.mark.parametrize('sizes,expected', [
    ([1]+[2]*19+[40], (40, 2, 40, 20, 78)),
    ([2]*18+[3], (3, 3, 3, 19, 39)),
    ([2]*20+[3], (3, 2, 3, 21, 43)),
    ([7, 6, 12, 2, 1], (12, 12, 12, 4, 27)),
])
def test_nearest_rank_statistics(sizes, expected):
    result = k.size_statistics(sizes)
    assert tuple(result[x] for x in ['max','p95_entity','p95_record','entities_ge2','records_ge2']) == expected
    expanded = sorted(s for s in sizes if s>=2 for _ in range(s))
    assert result['p95_record'] == expanded[(95*len(expanded)+99)//100-1]


@pytest.mark.parametrize('cap,expected', [
    (2, [{0,1}, {2,3}, {4,5}, {6}]),
    (3, [{0,1}, {2,3}, {4,5,6}]),
    (4, [{0,1,2,3}, {4,5,6}]),
    ('inf', [{0,1,2,3,4,5,6}]),
])
def test_existing_union_find_matches_hand_graph(cap, expected):
    # Pair bridges arrive before the final singleton edge; ordering matters.
    a,b = np.array([(0,1),(2,3),(4,5),(1,2),(3,4),(5,6)]).T
    labels,retained,_,_ = build_graph(7,a,b,cap)
    assert {frozenset(np.flatnonzero(labels==c)) for c in set(labels)} == {frozenset(c) for c in expected}
    assert np.array_equal(retained,labels[a]==labels[b])
    data = dict(setting=dict(setting=11),ids=pd.Index(range(7)),a=a,b=b,
                frame=pd.DataFrame(dict(record_id=range(7),visual_group_id=['a','a','a','b','b','c','c'])))
    result = k.measure(data,'all',cap)
    assert result['clusters'] == len(expected)
    assert result['max_cluster'] == max(map(len,expected))
    assert result['U'] == sum(len(c)*(len(c)-1)//2 for c in expected)


def test_pair_counts_conserve_and_match_enumeration():
    labels = np.array([0,0,0,3,3,5])
    truth = np.array(['a','a','b','b','b','c'])
    result = k.graph.pairwise_counts(labels,truth)
    expected = Counter_pairs(labels,truth)
    assert {key:result[key] for key in expected} == expected
    assert result['tp']+result['fn'] == 4
    assert sum(result[key] for key in ['tp','fp','fn','tn']) == 15


def Counter_pairs(labels, truth):
    counts = dict(tp=0,fp=0,fn=0,tn=0)
    for a,b in combinations(range(len(labels)),2):
        positive,same = truth[a]==truth[b],labels[a]==labels[b]
        counts['tp' if same and positive else 'fp' if same else 'fn' if positive else 'tn'] += 1
    return counts


def test_selection_uses_exact_fraction_and_small_k_ties():
    rows = [dict(K=4,exact_f1='1/2'),dict(K=3,exact_f1='1/2'),
            dict(K=5,exact_f1='50000000001/100000000000'),dict(K='inf',exact_f1='1/2')]
    assert k.select(rows,[3,4,'inf']) == 3
    assert k.select(rows,[3,4,5,'inf']) == 5


def test_m_is_path_node_but_not_evaluation_endpoint():
    labels,_,_,_ = build_graph(3,np.array([0,1]),np.array([1,2]),3)
    truth = pd.DataFrame(dict(record_id=['a','b','c'],visual_group_id=['x','y','x'],original_level=['L0','M','L0']))
    result = k.graph.evaluate(labels,truth,pd.Index(['a','b','c']))
    assert (result['records'],result['pairs'],result['tp'],result['fp']) == (2,1,1,0)
    assert k.graph.structure(labels)['U'] == 3
