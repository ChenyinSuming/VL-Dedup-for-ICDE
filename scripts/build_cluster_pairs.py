import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
ROOT = REPO_ROOT
RES = RESOURCES
RUN = RES / 'runs/caption_actions'
REPORT = ROOT / str(REPORTS / 'caption_actions')

def stamp():
    return datetime.now(timezone.utc).isoformat()

def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def write_rows(path, rows):
    with Path(path).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

def key(a, b):
    return tuple(sorted((a, b)))

def row_key(row):
    return key(row['record_id_a'], row['record_id_b'])

def pair_id(identity):
    return 'stage_e_clusters__' + '__'.join(identity)

def partition(identity, parts):
    (a, b) = (parts[rid] for rid in identity)
    return a if a == b else 'cross:' + '|'.join(sorted((a, b)))

def enumerate_pairs(ids, components, parts):
    mapping = component_mapping(ids, components)
    groups = defaultdict(list)
    for (rid, component) in mapping.items():
        groups[component].append(rid)
    rows = []
    for (component, members) in groups.items():
        for identity in combinations(sorted(members), 2):
            part = partition(identity, parts)
            rows.append(dict(pair_id=pair_id(identity), record_id_a=identity[0], record_id_b=identity[1], component=component, in_test=part == 'test', partition=part))
    return (sorted(rows, key=row_key), Counter(map(len, groups.values())), groups)

def by_partition(identities, parts):
    counts = Counter((partition(p, parts) for p in identities))
    return {p: counts[p] for p in ('test', 'validation', 'train', *sorted(set(counts) - {'test', 'validation', 'train'}))}

def account():
    import numpy as np
    import pyarrow.parquet as pq
    cfg = json.loads((REPORT / 'settings.json').read_text())
    dataset = RES / 'datasets/benchmark'
    parts = {r['record_id']: r['partition'] for r in read_rows(dataset / 'private/lineage.jsonl')}
    ids = pq.read_table(RES / cfg['inputs']['record_order'].split(':')[0], columns=['record_id']).column(0).to_pylist()
    components = np.load(RES / cfg['inputs']['components'])
    assert len(ids) == len(set(ids)) == len(components) == len(parts)
    assert set(ids) == set(parts)
    (pairs, sizes, groups) = enumerate_pairs(ids, components, parts)
    RUN.mkdir(parents=True, exist_ok=True)
    write_rows(RUN / 'U_new.jsonl', pairs)
    result = dict(components=sum(sizes.values()), size_counts=dict(sizes), U_new=len(pairs), by_partition=by_partition({row_key(r) for r in pairs}, parts))
    write_json(REPORT / 'results.json', result)
    return result

def component_mapping(ids, components):
    if len(ids) != len(components) or len(set(ids)) != len(ids):
        raise ValueError('Record/component mapping length or uniqueness mismatch')
    return dict(zip(ids, map(int, components), strict=True))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.parse_args()
    account()
