import numpy as np
import pandas as pd
import pytest
from scripts import build_candidates as candidates
from scripts import build_candidates
from scripts.run_action_checks import normalize_text, pr_curve, select_threshold

def test_edges_to_frame_uses_unordered_stable_edge_id():
    edges = {}
    candidates.update_edge(edges, 0, 1, 'phash', {'phash_distance': 0}, np.array(['train', 'train']))
    manifest = pd.DataFrame({'record_id': ['z', 'a']})
    frame = candidates.edges_to_frame(edges, manifest)
    assert frame.loc[0, 'edge_id'] == 'a__z'

def test_text_exact_edges_skip_cross_partition_pairs():
    manifest = pd.DataFrame({'record_id': ['train_a', 'train_b', 'test_a'], 'text': ['same caption', 'same caption', 'same caption'], 'partition': ['train', 'train', 'test']})
    edges = {}
    added = candidates.build_text_exact_edges(manifest, max_neighbors=0, edges=edges)
    frame = candidates.edges_to_frame(edges, manifest)
    assert added == 1
    assert len(frame) == 1
    assert frame.loc[0, 'record_id_a'] == 'train_a'
    assert frame.loc[0, 'record_id_b'] == 'train_b'

def test_threshold_uses_strict_interval_midpoint():
    selected = select_threshold([0.1, 0.3, 0.7, 0.9], [True, False, False, False])
    assert selected['threshold'] == pytest.approx(0.2)
    assert selected['exact_F1'] == '1'
    assert selected['optimal_intervals'][0]['lower'] == 0.1
    assert selected['optimal_intervals'][0]['upper'] == 0.3

def test_threshold_merges_adjacent_optimal_intervals():
    selected = select_threshold([0, 1, 1, 1, 2], [True, True, False, False, False])
    assert selected['exact_F1'] == '2/3'
    assert selected['threshold'] == 1
    assert len(selected['optimal_intervals']) == 1
    assert selected['optimal_intervals'][0]['upper'] == 2

def test_threshold_does_not_span_disconnected_optima():
    selected = select_threshold([0, 1, 2, 3, 4], [True, False, False, True, False])
    assert selected['exact_F1'] == '2/3'
    assert len(selected['optimal_intervals']) == 2
    assert selected['threshold'] == 0.5

def test_pr_ties_and_count_conservation_against_direct_predictions():
    scores = np.array([0.1, 0.1, 0.5, 0.9, 0.9])
    labels = np.array([True, False, True, False, True])
    curve = pr_curve(scores, labels)
    assert len(curve) == 4
    assert curve[0]['precision'] == 1 and curve[0]['recall'] == 0
    for p in curve:
        pred = scores < p['threshold']
        assert (p['TP'], p['FP'], p['FN'], p['TN']) == (int((pred & labels).sum()), int((pred & ~labels).sum()), int((~pred & labels).sum()), int((~pred & ~labels).sum()))
        assert p['TP'] + p['FN'] == 3
        assert p['FP'] + p['TN'] == 2
    assert curve[-1]['TP'] == 3 and curve[-1]['FP'] == 2
    ap = sum(((b['recall'] - a['recall']) * b['precision'] for (a, b) in zip(curve, curve[1:])))
    assert ap == pytest.approx(53 / 90)

@pytest.mark.parametrize('text, expected', [(' Ａ Cat,\nSITS! ', 'a cat sits'), ('a-cat', 'a cat'), ('Cat.', 'cat'), ('foo_bar', 'foo_bar'), ('Straße', 'straße'), ('...', '')])
def test_exact_stage_b_normalization(text, expected):
    assert normalize_text is build_candidates.normalize_text
    assert normalize_text(text) == expected
