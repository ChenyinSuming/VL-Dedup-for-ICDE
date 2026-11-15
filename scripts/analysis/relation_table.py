import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if __name__ == '__main__':
    from scripts.vl_paths import REPO_ROOT, REPORTS, ANALYSIS
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    import os
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[k] = '4'
    import sys, json, re
    sys.path.insert(0, str(REPO_ROOT))
    import numpy as np, pandas as pd
    from fractions import Fraction
    from scripts import train_hgb_classifier as hgb_sscd
    from scripts import train_visual_classifier as visual_classifier
    from scripts import run_sop_multiview as reference
    (DATA, SOURCE, classifier_run, baseline_surfaces, native_run) = (hgb_sscd.DATA, hgb_sscd.SOURCE, visual_classifier.OUT, visual_classifier.baseline_surfaces, hgb_sscd.native_baselines)
    GB = hgb_sscd.RES / 'runs/graph_baselines'
    MV = reference.OUT

    def part(path, p):
        pat = re.compile('\\"partition\\"\\s*:\\s*\\"' + p + '"')
        return pd.DataFrame([json.loads(l) for l in open(path) if pat.search(l)])
    lin = part(DATA / 'private/lineage.jsonl', 'test')
    rel = part(DATA / 'sealed/relation_applicability.jsonl', 'test')
    ids = pd.Index(pd.read_parquet(SOURCE / 'public_manifest.parquet').record_id)

    def load(path, idpath=None):
        lab = np.load(path)
        if idpath is None:
            assert len(lab) == 50000
            return lab
        rid = pd.read_parquet(idpath, columns=['record_id']).record_id
        m = pd.Series(lab, index=rid).reindex(ids)
        assert m.notna().all()
        return m.to_numpy()

    def specs(s):
        native_baselines = native_run / s
        sd = GB / ('SemDeDup/original/50/0.005_components.npy' if s == 'original' else f'SemDeDup/{s}/components.npy')
        b = baseline_surfaces if s == 'original' else baseline_surfaces / 'per_surface_candidates'
        return [('Data-Juicer', native_baselines / 'data_juicer_components.npy', native_baselines / 'input_mapping.parquet'), ('SemHash', native_baselines / 'semhash_0.9_components.npy', native_baselines / 'input_mapping.parquet'), ('SemDeDup', sd, None), ('Caption MinHash', GB / 'caption_Minhash/original/components.npy', None), ('B2', b / f'{s}_B2_inf_components.npy', None), ('B6', b / f'{s}_B6_inf_components.npy', None), ('HGB inf', classifier_run / f'graphs/{s}/classifier_HGB/inf/components.npy', None), ('HGB 5', classifier_run / f'graphs/{s}/classifier_HGB/5/components.npy', None), ('VL inf', classifier_run / f'graphs/{s}/classifier_LR/inf/components.npy', None), ('VL 5', classifier_run / f'graphs/{s}/classifier_LR/5/components.npy', None)]
    truth = lin[lin.original_level.ne('M')]
    tpos = ids.get_indexer(truth.record_id)
    (ra, rb) = [ids.get_indexer(rel[c]) for c in ['record_id_a', 'record_id_b']]
    out = {}
    for s in ['original', 'strong']:
        for (name, path, idp) in specs(s):
            lab = load(path, idp)
            f1 = float(Fraction(hgb_sscd.pairwise_counts(lab[tpos], truth.visual_group_id.to_numpy())['exact_f1']))
            co = lab[ra] == lab[rb]
            rates = {r: [int(co[(rel.relation == r).to_numpy()].sum()), int((rel.relation == r).sum())] for r in ['duplicate', 'keep_augment_candidate', 'text_duplicate_only', 'hard_negative', 'mismatch']}
            out[s, name] = dict(f1=round(f1, 4), **rates)
            print(s, name, round(f1, 4), rates, flush=True)
    pub = pd.Index(pd.read_parquet(MV / 'public_manifest.parquet').record_id)
    tp = pd.read_parquet(MV / 'test_pairs.parquet')
    sop = tp[tp.origin.eq('multiview_SOP')]
    (pa_, pb_) = [pub.get_indexer(sop[c]) for c in ['record_id_a', 'record_id_b']]
    same = sop.visual_relation.eq('same').to_numpy()
    mvp = json.load(open(str(REPORTS / 'mv_products/settings.json')))['rows']
    for r in mvp:
        lab = np.load(r['components'])
        assert len(lab) == 58550
        co = lab[pa_] == lab[pb_]
        print('MV', r['method'], r['k'], 'same', int(co[same].sum()), int(same.sum()), 'diff', int(co[~same].sum()), int((~same).sum()), flush=True)
