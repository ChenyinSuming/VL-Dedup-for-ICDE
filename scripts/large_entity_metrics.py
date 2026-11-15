import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
ROOT = REPO_ROOT
RES = RESOURCES
REPORT = ROOT / str(REPORTS / 'large_entities')
LARGE = RES / 'runs/large_entities'
ORIGINAL = RES / 'runs/capped_graphs'
OUT = LARGE / 'pairwise'
GRID = [2, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 'inf']
GRAPHS = ['large_entities', 'original']
MODELS = ['HGB', 'LR10']

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def dump(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def choose_two(counts):
    return sum((int(n) * (int(n) - 1) // 2 for n in counts))

def pairwise_counts(components, visual_groups):
    (components, visual_groups) = (np.asarray(components), np.asarray(visual_groups))
    if components.ndim != 1 or visual_groups.ndim != 1 or len(components) != len(visual_groups):
        raise ValueError('Component and group vectors must be aligned one-dimensional arrays')
    if pd.isna(visual_groups).any() or pd.isna(components).any():
        raise ValueError('Missing group/component identity')
    (_, component_codes, component_sizes) = np.unique(components, return_inverse=True, return_counts=True)
    (_, group_codes, group_sizes) = np.unique(visual_groups, return_inverse=True, return_counts=True)
    (_, cells) = np.unique(np.column_stack([component_codes, group_codes]), axis=0, return_counts=True)
    tp = choose_two(cells)
    (predicted_positive, true_positive) = (choose_two(component_sizes), choose_two(group_sizes))
    (fp, fn) = (predicted_positive - tp, true_positive - tp)
    pairs = len(components) * (len(components) - 1) // 2
    tn = pairs - tp - fp - fn
    assert min(tp, fp, fn, tn) >= 0
    denominator = 2 * tp + fp + fn
    exact = Fraction(2 * tp, denominator) if denominator else Fraction(0)
    return dict(records=len(components), pairs=pairs, true_positive_pairs=true_positive, predicted_positive_pairs=predicted_positive, tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / predicted_positive if predicted_positive else 0.0, recall=tp / true_positive if true_positive else 0.0, f1=float(exact), exact_f1=str(exact))

def select_k(rows):
    if {r['k'] for r in rows} != set(GRID) or len(rows) != len(GRID):
        raise ValueError('Selection requires the complete fixed K grid')
    return min(rows, key=lambda r: (-Fraction(r['exact_f1']), GRID.index(r['k'])))

def load_truth(graph, partition, out=OUT):
    columns = ['record_id', 'partition', 'visual_group_id']
    if graph == 'large_entities':
        truth = pd.read_parquet(LARGE / 'lineage.parquet', columns=columns, filters=[('partition', '==', partition)])
    else:
        pattern = re.compile('"partition"\\s*:\\s*"' + partition + '"')
        rows = []
        with (RES / 'datasets/benchmark/private/lineage.jsonl').open() as stream:
            for line in stream:
                if pattern.search(line):
                    row = json.loads(line)
                    rows.append({k: row[k] for k in columns})
        truth = pd.DataFrame(rows)
    assert len(truth) and truth.record_id.is_unique and truth.partition.eq(partition).all()
    assert truth.visual_group_id.notna().all()
    return truth

def inputs():
    result = {}
    for (graph, root) in [('large_entities', LARGE), ('original', ORIGINAL)]:
        path = root / ('public_manifest.parquet' if graph == 'large_entities' else 'component_row_ids.parquet')
        ids = pd.Index(pd.read_parquet(path, columns=['record_id']).record_id)
        assert len(ids) == (61000 if graph == 'large_entities' else 50000)
        assert ids.is_unique and ids.is_monotonic_increasing
        frozen = read(root / 'graphs_frozen.json')
        cells = {(r['model'], r['k']): r for r in frozen['cells']}
        assert len(cells) == 32 and set(cells) == {(m, k) for m in MODELS for k in GRID}
        result[graph] = dict(ids=ids, cells=cells, root=str(root), row_ids=str(path), graph_frozen_at_utc=frozen['frozen_at_utc'])
    assert set(result['original']['ids']) <= set(result['large_entities']['ids'])
    return result

def evaluate_partition(graph, source, partition):
    truth = load_truth(graph, partition)
    positions = source['ids'].get_indexer(truth.record_id)
    assert (positions >= 0).all()
    groups = truth.visual_group_id.to_numpy()
    rows = []
    for model in MODELS:
        for cap in GRID:
            cell = source['cells'][model, cap]
            labels = np.load(Path(cell['artifacts']) / 'components.npy', mmap_mode='r')
            assert labels.shape == (len(source['ids']),)
            stats = pairwise_counts(labels[positions], groups)
            rows.append(dict(graph=graph, partition=partition, model=model, k=cap, **stats))
    return rows

def flat_row(row):
    return dict(row)

def run():
    protocol = read(REPORT / 'settings.json')
    assert protocol['inputs']['grid'] == GRID
    source = inputs()
    dump(OUT / 'settings.json', protocol)
    results = dict(run_id=protocol['run_id'], sources={g: {k: v for (k, v) in s.items() if k not in ['ids', 'cells']} for (g, s) in source.items()}, validation_first_computed_at_utc=now(), validation=[], selections={})
    for graph in GRAPHS:
        results['validation'] += evaluate_partition(graph, source[graph], 'validation')
    for graph in GRAPHS:
        for model in MODELS:
            rows = [r for r in results['validation'] if r['graph'] == graph and r['model'] == model]
            selected = dict(select_k(rows), frozen_at_utc=now(), validation_cells=16)
            key = graph + '__' + model
            results['selections'][key] = selected
            dump(OUT / ('selection_' + key + '.json'), selected)
    dump(OUT / 'validation_frozen.json', results)
    frozen = dict(frozen_at_utc=now(), validation_cells=64, selections=results['selections'])
    dump(OUT / 'selections_frozen.json', frozen)
    results['selections_frozen_at_utc'] = frozen['frozen_at_utc']
    results['test_first_computed_at_utc'] = now()
    results['test'] = []
    for graph in GRAPHS:
        results['test'] += evaluate_partition(graph, source[graph], 'test')
    results['completed_at_utc'] = now()
    dump(OUT / 'results.json', results)
    dump(REPORT / 'pairwise_results.json', results)
    for partition in ['validation', 'test']:
        pd.DataFrame([flat_row(r) for r in results[partition]]).to_csv(REPORT / f'pairwise_{partition}.csv', index=False)
    print(json.dumps({k: {f: s[f] for f in ['k', 'tp', 'fp', 'fn', 'f1']} for (k, s) in results['selections'].items()}), flush=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run()
if __name__ == '__main__':
    main()
