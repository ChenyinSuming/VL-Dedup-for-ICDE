import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
from collections import defaultdict
from pathlib import Path
import re
import sys
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts import large_entity_metrics as base
(RES, REPORT, LARGE, ORIGINAL) = (base.RES, base.REPORT, base.LARGE, base.ORIGINAL)
(GRID, GRAPHS, MODELS) = (base.GRID, base.GRAPHS, base.MODELS)
OUT = LARGE / 'pairwise'
FIELDS = ['records', 'pairs', 'true_positive_pairs', 'predicted_positive_pairs', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1', 'exact_f1']

def inherit_levels(original, new):
    columns = ['record_id', 'partition', 'visual_group_id', 'original_level']
    original = original[columns].copy()
    if original.isna().any().any() or not original.record_id.is_unique:
        raise ValueError('Missing or duplicate original identity/level')
    indexed = original.set_index('record_id')
    groups = defaultdict(list)
    for row in original.to_dict('records'):
        groups[row['visual_group_id']].append(row)
    original['level_source_record_id'] = original.record_id
    original['is_new'] = False
    additions = []
    for row in new:
        source_id = row['caption_source_record_id']
        source = indexed.loc[source_id]
        members = groups[row['visual_group_id']]
        assert len(members) == 5 and all((r['original_level'] in {'L0', 'L2'} for r in members))
        assert source['partition'] == row['partition'] and source['visual_group_id'] == row['visual_group_id']
        additions.append({k: row[k] for k in ['record_id', 'partition', 'visual_group_id']} | dict(original_level=source['original_level'], level_source_record_id=source_id, is_new=True))
    result = pd.concat([original, pd.DataFrame(additions)], ignore_index=True) if additions else original
    assert result.record_id.is_unique
    return result

def original_metadata(partition):
    pattern = re.compile('"partition"\\s*:\\s*"' + partition + '"')
    rows = []
    import json
    with (RES / 'datasets/benchmark/private/lineage.jsonl').open() as stream:
        for line in stream:
            if pattern.search(line):
                row = json.loads(line)
                rows.append({k: row[k] for k in ['record_id', 'partition', 'visual_group_id', 'original_level']})
    return pd.DataFrame(rows)

def metadata(graph, partition):
    original = original_metadata(partition)
    new = []
    if graph == 'large_entities':
        import json
        pattern = re.compile('"partition"\\s*:\\s*"' + partition + '"')
        with (LARGE / 'new_records.jsonl').open() as stream:
            for line in stream:
                if pattern.search(line):
                    new.append(json.loads(line))
        assert len(new) == 5500
    result = inherit_levels(original, new)
    assert len(result) and result.partition.eq(partition).all()
    return result

def filtered_counts(labels, truth, exclude_m=True):
    if truth.original_level.isna().any():
        raise ValueError('Missing original_level')
    mask = truth.original_level.ne('M').to_numpy() if exclude_m else np.ones(len(truth), dtype=bool)
    counts = base.pairwise_counts(np.asarray(labels)[mask], truth.visual_group_id.to_numpy()[mask])
    return dict(records_before_exclusion=len(truth), excluded_m_records=int((~mask).sum()), **counts)

def evaluate_partition(graph, source, partition):
    truth = metadata(graph, partition)
    truth.to_parquet(OUT / f'metadata_{graph}_{partition}.parquet', index=False)
    positions = source['ids'].get_indexer(truth.record_id)
    assert (positions >= 0).all()
    rows = []
    for model in MODELS:
        for cap in GRID:
            labels = np.load(Path(source['cells'][model, cap]['artifacts']) / 'components.npy', mmap_mode='r')
            counts = filtered_counts(labels[positions], truth)
            rows.append(dict(graph=graph, partition=partition, model=model, k=cap, **counts))
    return rows

def run_evaluation():
    OUT.mkdir(parents=True, exist_ok=True)
    source = base.inputs()
    results = dict(validation=[], selections={})
    for graph in GRAPHS:
        results['validation'] += evaluate_partition(graph, source[graph], 'validation')
    for graph in GRAPHS:
        for model in MODELS:
            rows = [r for r in results['validation'] if r['graph'] == graph and r['model'] == model]
            selection = base.select_k(rows)
            key = graph + '__' + model
            results['selections'][key] = selection
            base.dump(OUT / ('selection_' + key + '.json'), selection)
    base.dump(OUT / 'selections_frozen.json', dict(selections=results['selections']))
    results['test'] = []
    for graph in GRAPHS:
        results['test'] += evaluate_partition(graph, source[graph], 'test')
    base.dump(OUT / 'results.json', results)
    base.dump(REPORT / 'pairwise_results.json', results)
    for part in ['validation', 'test']:
        pd.DataFrame(results[part]).to_csv(REPORT / f'pairwise_{part}.csv', index=False)

def main():
    run_evaluation()
if __name__ == '__main__':
    main()
