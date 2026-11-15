"""Recompute every count of the graph table (results/graph.csv) from the released cluster labels and benchmark manifests.

Usage:
    python scripts/recompute_graph_metrics.py            # compare with results/graph.csv
    python scripts/recompute_graph_metrics.py --write    # overwrite results/graph.csv

Inputs:
    benchmark/records.csv.gz, benchmark/pairs.csv.gz
    benchmark/multiview/records.csv.gz, benchmark/multiview/pairs.csv.gz
    artifacts/clusters/<surface>.parquet   one cluster label per record and method

The left block of the graph table is pairwise F1 over all record pairs of the test records that are
not M records (swapped captions). Two records are a true pair when they share a visual group.
Products restricts the multi-view surface to the product (SOP) records. The right block counts
the labeled test pairs of each relation whose two records share a cluster.
"""
import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'benchmark'
CLUSTERS = ROOT / 'artifacts' / 'clusters'
RESULT = ROOT / 'results' / 'graph.csv'

METHODS = ['data_juicer', 'semhash', 'semdedup', 'caption_minhash', 'b2_sscd', 'b6_fusion',
           'hgb_kinf', 'hgb_k5', 'vl_dedup_kinf', 'vl_dedup_k5']
EDITED = ['original', 'mixed', 'mild', 'strong']
RELATIONS = ['duplicate', 'keep_augment_candidate', 'text_duplicate_only', 'hard_negative', 'mismatch']


def choose2(counts):
    return sum(int(c) * (int(c) - 1) // 2 for c in counts)


def pairwise(labels, groups):
    """True positive, false positive and false negative record pairs of a clustering."""
    _, c, c_sizes = np.unique(labels, return_inverse=True, return_counts=True)
    _, g, g_sizes = np.unique(groups, return_inverse=True, return_counts=True)
    _, cells = np.unique(np.column_stack([c, g]), axis=0, return_counts=True)
    tp = choose2(cells)
    return tp, choose2(c_sizes) - tp, choose2(g_sizes) - tp


def compute():
    records = pd.read_csv(BENCH / 'records.csv.gz')
    pairs = pd.read_csv(BENCH / 'pairs.csv.gz')
    mv_records = pd.read_csv(BENCH / 'multiview' / 'records.csv.gz')
    mv_pairs = pd.read_csv(BENCH / 'multiview' / 'pairs.csv.gz')

    test = records[records.partition.eq('test') & records.level.ne('M')]
    mv_new = mv_records[mv_records.partition.eq('test')].assign(source_dataset='stanford_online_products')
    mv_test = pd.concat([test, mv_new], ignore_index=True)[['record_id', 'visual_group_id', 'source_dataset']]
    products = mv_test[mv_test.source_dataset.eq('stanford_online_products')]
    test_pairs = pairs[pairs.partition.eq('test')]
    mv_test_pairs = mv_pairs[mv_pairs.partition.eq('test')]
    same_product = mv_test_pairs.visual_relation.eq('same').to_numpy()

    labels = {s: pd.read_parquet(CLUSTERS / f'{s}.parquet').set_index('record_id')
              for s in EDITED + ['multiview']}
    rows = []
    for m in METHODS:
        row = dict(method=m)
        for s in EDITED:
            tp, fp, fn = pairwise(labels[s][m].loc[test.record_id].to_numpy(), test.visual_group_id.to_numpy())
            row.update({f'{s}_tp': tp, f'{s}_fp': fp, f'{s}_fn': fn})
        for scope, truth in [('mv', mv_test), ('products', products)]:
            tp, fp, fn = pairwise(labels['multiview'][m].loc[truth.record_id].to_numpy(), truth.visual_group_id.to_numpy())
            row.update({f'{scope}_tp': tp, f'{scope}_fp': fp, f'{scope}_fn': fn})
        lab = labels['original'][m]
        joined = lab.loc[test_pairs.record_id_a].to_numpy() == lab.loc[test_pairs.record_id_b].to_numpy()
        for rel in RELATIONS:
            sel = test_pairs.relation.eq(rel).to_numpy()
            row[f'joined_{rel}'], row[f'total_{rel}'] = int(joined[sel].sum()), int(sel.sum())
        lab = labels['multiview'][m]
        joined = lab.loc[mv_test_pairs.record_id_a].to_numpy() == lab.loc[mv_test_pairs.record_id_b].to_numpy()
        row.update(joined_same_product=int(joined[same_product].sum()), total_same_product=int(same_product.sum()),
                   joined_two_products=int(joined[~same_product].sum()), total_two_products=int((~same_product).sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    new = compute()
    for s in EDITED + ['mv', 'products']:
        new[f'{s}_f1'] = [float(Fraction(2 * tp, 2 * tp + fp + fn)) for tp, fp, fn in
                          zip(new[f'{s}_tp'], new[f'{s}_fp'], new[f'{s}_fn'])]
    if args.write:
        new.to_csv(RESULT, index=False)
        print(f'wrote {RESULT}')
        return 0
    released = pd.read_csv(RESULT)
    counts = [c for c in released.columns if c != 'method' and not c.endswith('_f1')]
    diff = (released.set_index('method')[counts] != new.set_index('method')[counts]).sum().sum()
    f1_diff = max(abs(released[c] - new[c]).max() for c in released.columns if c.endswith('_f1'))
    print(new[['method'] + [f'{s}_f1' for s in EDITED + ['mv', 'products']]].round(4).to_string(index=False))
    print(f'{len(counts) * len(new)} counts compared, {diff} differ; largest F1 difference {f1_diff:.2e}')
    return 1 if diff or f1_diff > 1e-12 else 0


if __name__ == '__main__':
    sys.exit(main())
