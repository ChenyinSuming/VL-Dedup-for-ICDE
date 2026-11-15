from sklearn.isotonic import IsotonicRegression
from itertools import combinations
import numpy as np
import pandas as pd
import pytest
from scripts import train_visual_classifier as m
from scripts.cluster_with_size_cap import build_graph, edge_order, same_partition

def test_feature_order_and_values():
    assert m.FEATURES == ['clip_image', 'clip_text', 'phash_similarity', 'raw_text_equal', 'normalized_text_equal', 'clipscore_min', 'clipscore_mean', 'clipscore_difference', 'cross_clip_min', 'cross_clip_mean', 'sscd_cosine']
    frame = pd.DataFrame(dict(record_id=['a', 'b'], text=['x', 'x'], image_path=['a.jpg', 'a.jpg']))
    image = np.array([[1, 0], [0, 1]], dtype=np.float32)
    text = np.array([[1, 0], [1, 0]], dtype=np.float32)
    base = m.hgb_sscd.Features.__new__(m.hgb_sscd.Features)
    base.ids = pd.Index(frame.record_id)
    base.observable = m.hgb_sscd.ObservableFeatures(frame, image, text, [0, 2 ** 64 - 1])
    base.sscd = np.array([[1, 0], [0.6, 0.8]], dtype=np.float32)
    feature = m.Features.__new__(m.Features)
    (feature.base, feature.ids) = (base, base.ids)
    pairs = pd.DataFrame(dict(record_id_a=['a'], record_id_b=['b']))
    actual = feature.transform(pairs)
    np.testing.assert_allclose(actual, [[0, 1, 0, 1, 1, 0, 0.5, 1, 0, 0.5, 0.6]])
    base.observable.assets[1] = 'different.jpg'
    np.testing.assert_array_equal(actual, feature.transform(pairs))

@pytest.mark.parametrize('exclude_m', [True, False])
def test_contingency_matches_bruteforce(exclude_m):
    ids = pd.Index(list('abcdef'))
    truth = pd.DataFrame(dict(record_id=list('abcdef'), visual_group_id=['x', 'x', 'x', 'y', 'y', 'z'], original_level=['L2', 'M', 'L2', 'HN', 'HN', 'M']))
    components = np.array([1, 1, 1, 1, 2, 3])
    actual = m.pairwise_counts(components, truth, ids, exclude_m)
    chosen = truth[truth.original_level.ne('M')] if exclude_m else truth
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    for (a, b) in combinations(chosen.itertuples(), 2):
        y = a.visual_group_id == b.visual_group_id
        pred = components[ids.get_loc(a.record_id)] == components[ids.get_loc(b.record_id)]
        counts['tp' if y and pred else 'fn' if y else 'fp' if pred else 'tn'] += 1
    assert {k: actual[k] for k in counts} == counts
    assert actual['n'] == sum(counts.values()) == len(chosen) * (len(chosen) - 1) // 2

def test_cap_rejects_cross_component_edge_with_sizes():
    (labels, kept, rejected, counts) = build_graph(4, np.array([0, 2, 1]), np.array([1, 3, 2]), 2)
    assert kept.tolist() == [True, True, False]
    assert rejected.tolist() == [[2, 2, 2]]
    assert labels[0] == labels[1] and labels[2] == labels[3] and (labels[1] != labels[2])
    assert counts['cap_rejected_edges'] == 1

def test_redundant_edge_retained_and_three_counts_conserved():
    (labels, kept, _, counts) = build_graph(4, np.array([0, 1, 0, 2]), np.array([1, 2, 2, 3]), 3)
    assert kept.tolist() == [True, True, True, False]
    assert counts == dict(union_merges=2, same_root_redundant_edges=1, cap_rejected_edges=1, retained_direct_edges=3)
    assert sum((counts[k] for k in ['union_merges', 'same_root_redundant_edges', 'cap_rejected_edges'])) == 4
    assert labels[0] == labels[2] != labels[3]

def test_infinity_equals_uncapped_union_including_isolate():
    (a, b) = (np.array([0, 1, 0, 3]), np.array([1, 2, 2, 4]))
    (labels, kept, _, counts) = build_graph(6, a, b, 'inf')
    unlimited = build_graph(6, a, b, 6)[0]
    assert same_partition(labels, unlimited)
    assert same_partition(labels, [8, 8, 8, 4, 4, 7])
    assert not same_partition(labels, [8, 8, 8, 4, 4, 4])
    assert kept.all() and counts['cap_rejected_edges'] == 0

def test_edge_order_calibrated_then_raw_then_public_ids():
    (a, b) = (np.array([3, 2, 0, 1]), np.array([1, 0, 1, 0]))
    order = edge_order(np.arange(4), np.array([0.8, 0.8, 0.7, 0.8]), a, b, np.array([0.9, 0.8, 0.99, 0.8]))
    assert order.tolist() == [0, 3, 1, 2]

def test_k_uses_exact_f1_and_smaller_cap_for_ties():
    rows = [dict(k=k, pairwise=dict(exact_f1='1/3')) for k in m.GRID]
    assert m.select_k(rows[::-1]) == 2
    rows[-1]['pairwise']['exact_f1'] = '10000000000000001/30000000000000000'
    assert m.select_k(rows) == 'inf'

def test_classifier_probabilities_and_isotonic_calibration():

    class Model:
        classes_ = ['hard_negative', 'keep_augment_candidate', 'duplicate']

        def predict_proba(self, x):
            return np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    np.testing.assert_allclose(m.hgb_sscd.raw_probability(Model(), None), [0.3, 0.9])
    calibration = IsotonicRegression(out_of_bounds='clip').fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    np.testing.assert_allclose(calibration.predict([0, 0.1, 0.2, 0.3, 0.4, 1]), [0, 0, 0.5, 0.5, 1, 1])
