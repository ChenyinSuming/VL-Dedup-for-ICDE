import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import time
import numpy as np
import pandas as pd
GRID = [2, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 'inf']

def edge_order(selected, scores, a, b, raw_positive=None):
    keys = (np.maximum(a[selected], b[selected]), np.minimum(a[selected], b[selected]))
    if raw_positive is not None:
        assert raw_positive.shape == selected.shape
        keys += (-raw_positive,)
    return selected[np.lexsort((*keys, -scores[selected]))]

def same_partition(left, right):
    if len(left) != len(right):
        return False
    (left, right) = (pd.factorize(np.asarray(left))[0], pd.factorize(np.asarray(right))[0])
    return len(np.unique(np.column_stack([left, right]), axis=0)) == len(np.unique(left)) == len(np.unique(right))

def build_graph(n, a, b, cap, deadline=None):
    (parent, sizes) = (list(range(n)), [1] * n)
    retained = np.ones(len(a), dtype=bool)
    (rejected, merges, redundant) = ([], 0, 0)
    limit = n if cap == 'inf' else cap

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (i, (u, v)) in enumerate(zip(a, b)):
        if deadline is not None and i % 4096 == 0 and (time.perf_counter() >= deadline):
            raise TimeoutError('Graph construction deadline exceeded')
        (x, y) = (find(int(u)), find(int(v)))
        if x == y:
            redundant += 1
        elif sizes[x] + sizes[y] > limit:
            retained[i] = False
            rejected.append((i, sizes[x], sizes[y]))
        else:
            if sizes[x] < sizes[y] or (sizes[x] == sizes[y] and x > y):
                (x, y) = (y, x)
            parent[y] = x
            sizes[x] += sizes[y]
            merges += 1
    labels = np.array([find(i) for i in range(n)], dtype=np.int32)
    assert np.array_equal(retained, labels[a] == labels[b])
    counts = dict(union_merges=merges, same_root_redundant_edges=redundant, cap_rejected_edges=len(rejected), retained_direct_edges=int(retained.sum()))
    assert merges + redundant + len(rejected) == len(a)
    assert merges + redundant == int(retained.sum())
    if cap != 'inf':
        assert max(np.unique(labels, return_counts=True)[1]) <= cap
    return (labels, retained, np.array(rejected, dtype=np.int64).reshape(-1, 3), counts)
