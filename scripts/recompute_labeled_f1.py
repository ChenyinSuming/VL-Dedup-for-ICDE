"""Compute the Labeled column of Table IV: co-membership F1 on the labeled test pairs.

A pair is positive when its two records show the same visual entity. Pairs with a swapped caption
are left out, which leaves 9,494 labeled test pairs.

    python scripts/recompute_labeled_f1.py            # compare with results/graph_labeled_pairs.csv
    python scripts/recompute_labeled_f1.py --write    # write results/graph_labeled_pairs.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'results' / 'graph_labeled_pairs.csv'
METHODS = ['data_juicer', 'semhash', 'semdedup', 'caption_minhash', 'b2_sscd', 'b6_fusion',
           'hgb_kinf', 'hgb_k5', 'vl_dedup_kinf', 'vl_dedup_k5']


def compute():
    pairs = pd.read_csv(ROOT / 'benchmark' / 'pairs.csv.gz')
    pairs = pairs[(pairs.partition == 'test') & (pairs.relation != 'mismatch')]
    clusters = pd.read_parquet(ROOT / 'artifacts' / 'clusters' / 'original.parquet')
    if 'record_id' in clusters.columns:
        clusters = clusters.set_index('record_id')
    rows = []
    for method in METHODS:
        same = clusters[method].reindex(pairs.record_id_a).to_numpy() == clusters[method].reindex(pairs.record_id_b).to_numpy()
        positive = (pairs.visual_relation == 'same').to_numpy()
        tp, fp, fn = int((same & positive).sum()), int((same & ~positive).sum()), int((~same & positive).sum())
        rows.append(dict(method=method, pairs=len(pairs), tp=tp, fp=fp, fn=fn, f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0))
    return pd.DataFrame(rows)


def main():
    new = compute()
    if '--write' in sys.argv:
        new.to_csv(RESULT, index=False)
    released = pd.read_csv(RESULT)
    assert new[['tp', 'fp', 'fn']].equals(released[['tp', 'fp', 'fn']])
    print(new.to_string(index=False))


if __name__ == '__main__':
    main()
