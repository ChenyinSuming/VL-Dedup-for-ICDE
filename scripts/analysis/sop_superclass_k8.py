import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if __name__ == '__main__':
    from scripts.vl_paths import RESOURCES, ANALYSIS
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd
    from pathlib import Path
    R = RESOURCES / 'runs/sop_scale'
    corpus = pd.read_parquet(R / 'corpus.parquet')
    cls = corpus.class_id.values
    sup = corpus.super_class_id.values
    for k in (5, 8):
        lab = np.load(R / f'classifier_LR_K{k}_components.npy')
        order = np.argsort(lab, kind='stable')
        sl = lab[order]
        groups = np.split(order, np.flatnonzero(np.diff(sl)) + 1)
        n = same = 0
        for g in groups:
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if cls[g[i]] != cls[g[j]]:
                        n += 1
                        same += sup[g[i]] == sup[g[j]]
        print(k, 'false joins', n, 'same superclass', same, round(same / n, 4))
