import numpy as np

def kwikcluster(n, a, b, constraints=()):
    adjacency = [[] for _ in range(n)]
    for (x, y) in zip(a, b):
        adjacency[x].append(int(y))
        adjacency[y].append(int(x))
    forbidden = {}
    for (x, y) in constraints:
        forbidden.setdefault(int(x), set()).add(int(y))
        forbidden.setdefault(int(y), set()).add(int(x))
    labels = np.full(n, -1, dtype=np.int32)
    rejected = 0
    for pivot in range(n):
        if labels[pivot] >= 0:
            continue
        labels[pivot] = pivot
        group = {pivot}
        for neighbor in sorted(adjacency[pivot]):
            if labels[neighbor] >= 0:
                continue
            if forbidden.get(neighbor, set()) & group:
                rejected += 1
                continue
            labels[neighbor] = pivot
            group.add(neighbor)
    return (labels, rejected)
