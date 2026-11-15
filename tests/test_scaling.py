from itertools import combinations
import numpy as np
import pandas as pd
import pytest
from scripts import run_pipeline_scaling as MODULE

def test_review_count_includes_singletons_and_counts_unordered_pairs():
    assert MODULE.review_counts([9, 9, 9, 4, 4, 7]) == dict(clusters=3, max_cluster=3, U=4)

def test_review_count_uses_python_integer_beyond_int32():
    assert MODULE.review_counts(np.zeros(100000, dtype=np.int32))['U'] == 4999950000

def test_subset_adapter_keeps_visual_classifier_eleven_features_and_sscd():
    frame = pd.DataFrame(dict(record_id=['A', 'C'], text=['alpha', 'beta'], image_path=['/tmp/same.jpg', '/tmp/same.jpg']))
    image = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    text = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    sscd = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    feature = MODULE.visual_classifier_features(frame, image, text, np.array([0, 1], dtype=np.uint64), sscd)
    pairs = pd.DataFrame(dict(record_id_a=['A'], record_id_b=['C']))
    actual = feature.transform(pairs)
    assert actual.shape == (1, 11)
    assert actual[0, -1] == np.float32(0.6)
    assert actual[0, 2] == np.float32(63 / 64)
    assert actual[0, 5] == 0 and actual[0, 8] == 1

@pytest.mark.parametrize('threshold', [0.2, 0.5])
def test_routing_matches_pair_enumeration_and_keeps_equality(threshold):
    labels = np.array([0, 0, 0, 3, 3, 5])
    scores = np.array([0.1, 0.2, 0.9, 0.5, 0.8, 0.0], dtype=np.float64)
    actual = MODULE.below_threshold_counts(labels, scores, threshold)
    pairs = [(a, b) for (a, b) in combinations(range(len(labels)), 2) if labels[a] == labels[b]]
    below = sum((min(float(scores[a]), float(scores[b])) < threshold for (a, b) in pairs))
    assert actual == dict(U=len(pairs), below_t_new=below, vlm_call_pairs=len(pairs) - below)

def test_routing_does_not_round_frozen_threshold_to_float32():
    score = np.float32(0.2)
    threshold = np.nextafter(float(score), np.inf)
    assert MODULE.below_threshold_counts([0, 0, 0], [score, 0.3, 0.5], threshold) == dict(U=3, below_t_new=2, vlm_call_pairs=1)
