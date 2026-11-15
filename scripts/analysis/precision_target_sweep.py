import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if __name__ == '__main__':
    from scripts.vl_paths import REPO_ROOT, ANALYSIS
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    import os
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[k] = '4'
    import sys, json, re
    sys.path.insert(0, str(REPO_ROOT))
    import numpy as np, pandas as pd
    from fractions import Fraction
    from scripts import train_hgb_classifier as hgb_sscd
    from scripts import train_visual_classifier as visual_classifier
    from scripts import run_sop_multiview as reference
    from scripts.pair_features import binary_metrics
    (DATA, SOURCE, classifier_run, GRID, ENDS) = (hgb_sscd.DATA, hgb_sscd.SOURCE, visual_classifier.OUT, visual_classifier.GRID, hgb_sscd.ENDS)
    MV = reference.OUT

    def part(path, p):
        pat = re.compile('\\"partition\\"\\s*:\\s*\\"' + p + '"')
        return pd.DataFrame([json.loads(l) for l in open(path) if pat.search(l)])
    lin_val = part(DATA / 'private/lineage.jsonl', 'validation')
    lin_test = part(DATA / 'private/lineage.jsonl', 'test')
    rel_val = part(DATA / 'sealed/relation_applicability.jsonl', 'validation')
    ids = pd.Index(pd.read_parquet(SOURCE / 'public_manifest.parquet').record_id)
    assert ids.is_monotonic_increasing and len(ids) == 50000

    def pw(labels, truth, ids):
        t = truth[truth.original_level.ne('M')]
        pos = ids.get_indexer(t.record_id)
        assert (pos >= 0).all()
        return hgb_sscd.pairwise_counts(np.asarray(labels)[pos], t.visual_group_id.to_numpy())

    def choose(y, score, recalled, pi):
        el = []
        for thr in np.unique(np.r_[np.linspace(0, 1, 201), score]):
            m = binary_metrics(y, recalled & (score >= thr))
            if m['tp'] + m['fp'] >= 20 and m['precision'] >= pi:
                el.append((m['recall'], float(thr)))
        return max(el, default=(0.0, 1.000001), key=lambda x: (x[0], x[1]))[1]
    surf = {}
    for s in ['original', 'mixed', 'mild', 'strong']:
        (_, a, b, codes) = hgb_sscd.candidate_input(s, ids)
        surf[s] = dict(a=a, b=b, codes=codes, cal=np.load(classifier_run / f'{s}_classifier_LR_calibrated.npy', mmap_mode='r'), raw=np.load(classifier_run / f'{s}_classifier_LR_raw.npy', mmap_mode='r'), n=50000, ids=ids)
    o = surf['original']
    known = rel_val.visual_relation_mask.to_numpy(dtype=bool)
    yv = rel_val.visual_relation.eq('same').to_numpy()[known]
    (va, vb) = [ids.get_indexer(rel_val[c]) for c in ENDS]
    vc = (np.minimum(va, vb).astype(np.int64) * 50000 + np.maximum(va, vb))[known]
    order = np.argsort(o['codes'])
    pos = np.searchsorted(o['codes'], vc, sorter=order)
    pos = np.clip(pos, 0, len(order) - 1)
    hit = o['codes'][order[pos]] == vc
    recalled = hit
    score = np.zeros(len(vc))
    score[hit] = np.asarray(o['cal'])[order[pos[hit]]]
    print('calibration pairs', len(yv), 'recalled', int(recalled.sum()), flush=True)

    def graph(S, thr, cap):
        (cal, raw) = (np.asarray(S['cal']), np.asarray(S['raw']))
        sel = np.flatnonzero(cal >= thr)
        ordered = hgb_sscd.edge_order(sel, cal, S['a'], S['b'], raw[sel])
        (labels, _, _, _) = hgb_sscd.build_graph(S['n'], S['a'][ordered], S['b'][ordered], cap)
        return (labels, len(sel))
    pub = pd.read_parquet(MV / 'public_manifest.parquet')
    mids = pd.Index(pub.record_id)
    assert len(mids) == 58550 and mids.is_monotonic_increasing
    mc = pd.read_parquet(MV / 'candidates.parquet', columns=ENDS)
    (ma, mb) = [mids.get_indexer(mc[c]) for c in ENDS]
    mvS = dict(a=ma, b=mb, cal=np.load(MV / 'visual_classifier_classifier/visual_classifier_classifier_LR_score.npy', mmap_mode='r'), raw=np.load(MV / 'visual_classifier_classifier/visual_classifier_classifier_LR_raw.npy', mmap_mode='r'), n=58550)
    new = pd.read_parquet(MV / 'new_manifest.parquet')
    cols = ['record_id', 'partition', 'visual_group_id', 'original_level', 'source_dataset']
    added = new[new.partition.eq('test')].copy().assign(original_level='SOP_multiview', source_dataset='stanford_online_products')
    mv_truth = pd.concat([lin_test[cols], added[cols]], ignore_index=True)
    sop_truth = mv_truth[mv_truth.source_dataset.eq('stanford_online_products')]
    out = []
    for pi in [0.95, 0.97, 0.98, 0.99, 0.995]:
        thr = choose(yv, score, recalled, pi)
        rows = []
        for cap in GRID:
            (labels, nsel) = graph(o, thr, cap)
            rows.append(dict(k=cap, f1=pw(labels, lin_val, ids)['exact_f1']))
        kd = min(rows, key=lambda r: (-Fraction(r['f1']), GRID.index(r['k'])))['k']
        res = dict(pi=pi, tau=thr, K=kd, val_f1=float(Fraction(max(rows, key=lambda r: Fraction(r['f1']))['f1'])))
        for s in ['original', 'mixed', 'mild', 'strong']:
            (labels, nsel) = graph(surf[s], thr, kd)
            c = pw(labels, lin_test, ids)
            res[s] = round(float(Fraction(c['exact_f1'])), 6)
            res['S_' + s] = nsel
        (labels, nsel) = graph(mvS, thr, kd)
        c = pw(labels, mv_truth, mids)
        sp = pw(labels, sop_truth, mids)
        res['MV'] = round(float(Fraction(c['exact_f1'])), 6)
        res['MV_products'] = round(float(Fraction(sp['exact_f1'])), 6)
        res['MV_records'] = c.get('records')
        res['MV_pos'] = c['tp'] + c['fn']
        print(json.dumps(res), flush=True)
        out.append(res)
    json.dump(out, open(ANALYSIS / 'precision_target_sweep.json', 'w'), indent=1)
