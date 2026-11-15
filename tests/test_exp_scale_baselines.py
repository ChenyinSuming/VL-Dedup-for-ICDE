"""SCALE_BASELINES_RUN native group projection and integer pairwise invariants."""
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiments.run_scale_baselines import groups_to_components, metrics, check_budget
from scripts.run_graph_baselines import pairwise_counts


def test_groups_merge_overlaps_and_preserve_singletons():
    labels = groups_to_components(8, [[3, 1], [2, 3], [5, 6], [], [5]])
    assert labels.tolist() == [0, 1, 1, 1, 4, 5, 5, 7]


def test_group_outside_corpus_rejected():
    with pytest.raises(ValueError):
        groups_to_components(3, [[0, 3]])


@pytest.mark.parametrize('native_groups', [[], [[0, 1, 2, 3, 4, 5]], [[0, 2], [2, 3], [1, 4]]])
def test_pairwise_counts_match_enumeration_and_existing(native_groups):
    labels = groups_to_components(6, native_groups)
    truth = np.array(['a', 'a', 'b', 'a', 'b', 'c'])
    actual = metrics(labels, truth)
    expected = dict(tp=0, fp=0, fn=0, tn=0)
    for a, b in itertools.combinations(range(6), 2):
        predicted, same = labels[a] == labels[b], truth[a] == truth[b]
        expected['tp' if predicted and same else 'fp' if predicted else 'fn' if same else 'tn'] += 1
    assert {key: actual[key] for key in expected} == expected
    existing = pairwise_counts(labels, truth)
    assert all(actual[key] == value for key, value in existing.items())
    assert sum(expected.values()) == actual['pairs'] == 15
    assert actual['clusters'] + actual['delete_if_one_per_cluster'] == 6
    assert actual['tp'] + actual['fp'] == actual['U']


def test_projection_keeps_paths_through_other_partitions():
    labels = groups_to_components(4, [[0, 1], [1, 2]])
    projected = metrics(labels[[0, 2, 3]], np.array(['a', 'a', 'b']))
    assert (projected['tp'], projected['fp'], projected['fn']) == (1, 0, 0)


def test_gate_rounding_does_not_change_counts():
    assert check_budget(dict(f1=328/4543, tp=328), dict(f1=.072199, tp=328)) == {}
    assert 'tp' in check_budget(dict(tp=327), dict(tp=328))
