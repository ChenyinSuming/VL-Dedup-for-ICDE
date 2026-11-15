import itertools
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_candidates as builder
from scripts.experiments.run_ann_scale import HNSW, embedding_edges, metrics, subset_positions
from scripts.cluster_with_size_cap import build_graph
from scripts.experiments.support import build_hnsw_edges


@pytest.mark.parametrize('backend', ['hnsw', 'exact'])
def test_embedding_channels_match_existing_builder(backend):
    rng = np.random.default_rng(72)
    frame = pd.DataFrame(dict(record_id=[f'r{i:02}' for i in range(24)], partition='global'))
    expected, actual = {}, {}
    builder.faiss.omp_set_num_threads(8)
    for channel, k in [('clip_image', 6), ('clip_text', 3)]:
        vectors = rng.normal(size=(len(frame), 8)).astype(np.float32)
        vectors[2] = vectors[0]  # Retain the existing tie/self behavior.
        builder.faiss.normalize_L2(vectors)
        parts = frame.partition.to_numpy()
        if backend == 'hnsw':
            added, _ = build_hnsw_edges(builder, vectors, channel, k, expected, parts, HNSW)
        else:
            added = builder.build_embedding_edges(vectors, channel, channel+'_score', k, None, expected, parts)
        count, phases = embedding_edges(builder, vectors, channel, k, actual, parts, backend)
        assert count == added
        assert phases['index_add']['wall_seconds'] >= 0
        assert phases['search']['wall_seconds'] >= 0
    pd.testing.assert_frame_equal(builder.edges_to_frame(expected, frame), builder.edges_to_frame(actual, frame))
    assert all(a < b for a, b in actual)
    assert len(actual) == len({frozenset(pair) for pair in actual})
    assert any(len(value['channels']) == 2 for value in actual.values())


@pytest.mark.parametrize('target', [5, 10, 18])
def test_whole_image_prefix_and_original_record_order(target):
    images = np.array([30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 40, 40, 40, 40, 40, 40])
    positions, count = subset_positions(images, target)
    shuffled = np.random.Generator(np.random.PCG64(20261106)).permutation(np.unique(images))
    chosen, total = [], 0
    for image in shuffled:
        chosen.append(image)
        total += np.count_nonzero(images == image)
        if total >= target:
            break
    assert count == len(chosen)
    assert np.array_equal(positions, np.flatnonzero(np.isin(images, chosen)))
    assert len(positions) >= target
    assert len(positions) - np.count_nonzero(images == chosen[-1]) < target
    for image in chosen:
        assert np.count_nonzero(images[positions] == image) == np.count_nonzero(images == image)


@pytest.mark.parametrize('cap', [3, 'inf'])
def test_graph_counts_and_projection_conserve_pairs(cap):
    groups = np.array(['a', 'a', 'a', 'b', 'b', 'c'])
    a, b = np.array([0, 1, 2, 3, 4]), np.array([1, 2, 3, 4, 5])
    labels, retained, rejected, counts = build_graph(6, a, b, cap)
    result = metrics(labels, groups)
    brute = dict(tp=0, fp=0, fn=0, tn=0)
    for left, right in itertools.combinations(range(6), 2):
        predicted, truth = labels[left] == labels[right], groups[left] == groups[right]
        brute['tp' if predicted and truth else 'fp' if predicted else 'fn' if truth else 'tn'] += 1
    assert {key: result[key] for key in brute} == brute
    assert sum(brute.values()) == 15
    assert counts['retained_direct_edges'] + len(rejected) == len(a)
    assert retained.sum() == counts['union_merges'] + counts['same_root_redundant_edges']
    mask = np.array([True, True, False, True, False, True])
    assert metrics(labels[mask], groups[mask])['pairs'] == 6
