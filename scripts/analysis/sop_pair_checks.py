import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if __name__ == '__main__':
    from scripts.vl_paths import RESOURCES, ANALYSIS
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    import sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    R = RESOURCES / 'runs/sop_scale'
    A = R / 'stage_a_amendment2'

    def c2(x):
        x = np.asarray(x, dtype=np.int64)
        return x * (x - 1) // 2
    corpus = pd.read_parquet(R / 'corpus.parquet')
    acorp = pd.read_parquet(A / 'corpus.parquet')
    assert corpus.record_id.is_monotonic_increasing
    assert (acorp.record_id.values == corpus.record_id.values).all()
    n = len(corpus)
    cls = corpus.class_id.values
    part = corpus.evaluation_partition.values
    print('n', n, 'products', corpus.class_id.nunique(), 'pos', int(c2(corpus.class_id.value_counts().values).sum()))
    for p in ['validation', 'test', 'train']:
        m = part == p
        print(p, int(m.sum()), corpus[m].class_id.nunique(), int(c2(corpus[m].class_id.value_counts().values).sum()))
    scopes = {'all': np.ones(n, bool), 'validation': part == 'validation', 'test': part == 'test'}
    runs = [('classifier_LR', 5), ('classifier_LR', 8), ('classifier_LR', 'inf'), ('classifier_HGB', 5), ('classifier_HGB', 8), ('classifier_HGB', 'inf'), ('B2', 'inf'), ('B6', 'inf')]
    out = {}
    for (meth, k) in runs:
        lab = np.load(R / f'{meth}_K{k}_components.npy')
        gmax = int(np.bincount(lab).max())
        for (s, m) in scopes.items():
            df = pd.DataFrame({'l': lab[m], 'c': cls[m]})
            tp = int(c2(df.groupby(['l', 'c']).size().values).sum())
            pr = int(c2(df.groupby('l').size().values).sum())
            po = int(c2(df.groupby('c').size().values).sum())
            (fp, fn) = (pr - tp, po - tp)
            P = tp / pr if pr else 0
            Rr = tp / po
            F = 2 * tp / (pr + po)
            out[f'{meth}|{k}|{s}'] = dict(U=pr, TP=tp, FP=fp, FN=fn, P=round(P, 6), R=round(Rr, 6), F1=round(F, 6), max=int(df.groupby('l').size().max()), clusters=int(df.l.nunique()))
        out[f'{meth}|{k}|global_max'] = gmax
    for (k, v) in out.items():
        print(k, v)
    cand = pd.read_parquet(A / 'candidates.parquet')
    print('cand cols', list(cand.columns), len(cand))
    (ia, ib) = ([c for c in cand.columns if c.endswith('_a') or c in ('a', 'i', 'left')][:1], None)
    print(cand.head(3).to_string())
    sys.stdout.flush()
    ph = np.load(A / 'pHash.npy')
    print('phash', ph.shape, ph.dtype)
    text = corpus.text.values
    sup = corpus.super_class_id.values

    def popcount64(x):
        x = x.astype(np.uint64)
        cnt = np.zeros(x.shape, np.int64)
        for sh in range(0, 64, 8):
            cnt += np.array([bin(i).count('1') for i in range(256)], np.int64)[(x >> np.uint64(sh) & np.uint64(255)).astype(np.int64)]
        return cnt
    for (meth, k) in [('classifier_LR', 5), ('classifier_LR', 8), ('B6', 'inf'), ('B2', 'inf')]:
        lab = np.load(R / f'{meth}_K{k}_components.npy')
        order = np.argsort(lab, kind='stable')
        sl = lab[order]
        bounds = np.flatnonzero(np.diff(sl)) + 1
        groups = np.split(order, bounds)
        (A_, B_) = ([], [])
        for g in groups:
            if len(g) < 2 or len(g) > 20:
                continue
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if cls[g[i]] != cls[g[j]]:
                        A_.append(g[i])
                        B_.append(g[j])
        (A_, B_) = (np.array(A_), np.array(B_))
        if len(A_) == 0:
            print(meth, k, 'no fp in small clusters')
            continue
        ham = popcount64(ph[A_] ^ ph[B_])
        same_sup = (sup[A_] == sup[B_]).mean()
        same_txt = (text[A_] == text[B_]).mean()
        test_m = (part[A_] == 'test') & (part[B_] == 'test')
        print(meth, k, 'FP pairs (clusters<=20)', len(A_), 'ham<=8', int((ham <= 8).sum()), 'ham==0', int((ham == 0).sum()), 'same_super', round(float(same_sup), 4), 'same_text', round(float(same_txt), 4), 'test FP', int(test_m.sum()), 'test ham<=8', int(((ham <= 8) & test_m).sum()))
