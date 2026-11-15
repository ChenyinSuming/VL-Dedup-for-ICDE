import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, REPORTS
from collections import Counter
from fractions import Fraction
from pathlib import Path
import sys
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from scripts import run_graph_baselines as graph
import numpy as np
import pandas as pd
import pyarrow as pa
from threadpoolctl import threadpool_limits
REPORT = ROOT / str(REPORTS / 'mv_products')
OUT = graph.RUNS / 'mv_products'

def product_records(truth):
    return truth[truth.partition.eq('test') & truth.original_level.ne('M') & truth.source_dataset.eq('stanford_online_products')]

def measure(labels, truth, ids):
    positions = ids.get_indexer(truth.record_id)
    assert (positions >= 0).all() and truth.record_id.is_unique
    selected = labels[positions]
    result = graph.pairwise_counts(selected, truth.visual_group_id.to_numpy())
    (tp, fp, fn) = [result[k] for k in ('tp', 'fp', 'fn')]
    precision = Fraction(tp, tp + fp) if tp + fp else Fraction(0)
    recall = Fraction(tp, tp + fn) if tp + fn else Fraction(0)
    return result | dict(precision=float(precision), recall=float(recall), exact_precision=str(precision), exact_recall=str(recall), largest_cluster=max(Counter(selected).values(), default=0))

def calculate(protocol):
    manifest = pd.read_parquet(protocol['full_graph']['node_order_manifest'], columns=['record_id'])
    ids = pd.Index(manifest.record_id)
    assert len(ids) == 58550 and ids.is_unique and ids.is_monotonic_increasing
    mapping = pd.read_parquet(graph.MV / 'external_baselines/input_mapping.parquet', columns=['record_id', 'row_index'])
    assert np.array_equal(mapping.record_id, ids)
    assert np.array_equal(mapping.row_index, np.arange(len(ids)))
    truth = graph.metadata('MV', 'test')
    overall = truth[truth.partition.eq('test') & truth.original_level.ne('M')]
    products = product_records(truth)
    (rows, universes) = ([], {})
    for (scope, records) in [('mv', overall), ('products', products)]:
        universe = graph.pairwise_counts(np.arange(len(records)), records.visual_group_id.to_numpy())
        keys = ['records', 'pairs', 'true_positive_pairs']
        universes[scope] = {k: universe[k] for k in keys}
    for spec in protocol['rows']:
        labels = np.load(spec['components'], mmap_mode='r')
        assert labels.shape == (len(ids),) and np.issubdtype(labels.dtype, np.integer)
        (mv, sop) = (measure(labels, overall, ids), measure(labels, products, ids))
        assert all((sop[k] <= mv[k] for k in ('records', 'pairs', 'true_positive_pairs', 'tp', 'fp', 'fn', 'tn')))
        rows.append({k: spec[k] for k in ('id', 'method', 'k', 'cluster_source', 'components')} | dict(mv=mv, products=sop, global_largest_cluster=max(Counter(labels).values())))
        print(spec['id'], sop['tp'], sop['fp'], sop['fn'], f"{sop['f1']:.6f}", sop['largest_cluster'], flush=True)
    return dict(run_id=protocol['run_id'], products_universe=universes['products'], mv_universe=universes['mv'], rows=rows)

def main():
    protocol = graph.read(REPORT / 'settings.json')
    with threadpool_limits(limits=4):
        pa.set_cpu_count(4)
        pa.set_io_thread_count(4)
        result = calculate(protocol)
    graph.dump(OUT / 'results.json', result)
    graph.dump(REPORT / 'results.json', result)
if __name__ == '__main__':
    main()
