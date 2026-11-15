import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if __name__ == '__main__':
    from scripts.vl_paths import REPO_ROOT, ANALYSIS
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    import importlib.util
    import sys
    ROOT = str(REPO_ROOT)
    sys.path.insert(0, ROOT)
    spec = importlib.util.spec_from_file_location('_fj_coco', ROOT + '/scripts/run_coco_scale.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    import joblib
    import numpy as np
    import pandas as pd
    from scripts import run_graph_baselines as graph
    from scripts.cluster_with_size_cap import build_graph
    frame = pd.read_parquet(m.SOURCE / 'public_manifest.parquet')
    candidates = pd.read_parquet(m.SOURCE / 'candidates.parquet')
    image = np.load(m.SOURCE / 'features/clip/image_emb.npy', mmap_mode='r')
    text = np.load(m.SOURCE / 'features/clip/text_emb.npy', mmap_mode='r')
    hashes = pd.read_parquet(m.SOURCE / 'features/phash/phash.parquet')
    assert hashes.record_id.tolist() == frame.record_id.tolist()
    ph = hashes.phash_int.to_numpy(dtype=np.uint64)
    sscd = np.load(m.RES / 'runs/edit_stress/original/sscd.npy', mmap_mode='r')
    bundle = joblib.load(m.visual_classifier / 'classifier_LR.joblib')
    with m.configure_threads(8):
        (raw, cal) = m.predict(frame, candidates, image, text, ph, sscd, bundle)
    (ids, a, b, sel) = m.positive_edges(frame, candidates, raw, cal, 2 / 3)
    (labels, _, _, _) = build_graph(len(frame), a, b, 5)
    truth = graph.metadata('original', 'test')
    truth = truth[truth.original_level.ne('M')]
    pos = ids.get_indexer(truth.record_id)
    print('counts', graph.pairwise_counts(labels[pos], truth.visual_group_id.to_numpy()))
    fpos = pd.Index(frame.record_id).get_indexer(ids)
    ph_s = ph[fpos]
    txt_s = frame.text.to_numpy()[fpos]
    img_s = frame.image_path.to_numpy()[fpos]
    emb_s = None
    t = truth.assign(pos=pos, lab=labels[pos])
    pairs = []
    for (lab, g) in t.groupby('lab'):
        if len(g) < 2:
            continue
        rows = g.to_dict('records')
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i]['visual_group_id'] != rows[j]['visual_group_id']:
                    pairs.append((rows[i], rows[j]))
    print('fp pairs', len(pairs))

    def ham(x, y):
        return bin(int(x) ^ int(y)).count('1')
    out = []
    for (r, s) in pairs:
        (i, j) = (r['pos'], s['pos'])
        (fi, fj) = (fpos[i], fpos[j])
        out.append(dict(a=r['record_id'], b=s['record_id'], src=r['visual_group_id'][:4] + '-' + s['visual_group_id'][:4], lv=r['original_level'] + '-' + s['original_level'], ga=r['visual_group_id'], gb=s['visual_group_id'], ham=ham(ph_s[i], ph_s[j]), same_text=txt_s[i] == txt_s[j], sscd=float(np.dot(sscd[fi], sscd[fj])), ta=txt_s[i], tb=txt_s[j], ia=img_s[i], ib=img_s[j]))
    df = pd.DataFrame(out)
    print(df.src.value_counts().to_dict())
    print('ham<=8', int((df.ham <= 8).sum()), 'ham==0', int((df.ham == 0).sum()), 'same_text', int(df.same_text.sum()))
    print('ham dist', df.ham.describe().to_dict())
    print('sscd', df.sscd.describe().to_dict(), 'sscd>=0.8288', int((df.sscd >= 0.828769862651825).sum()))
    print(df.lv.value_counts().head(10).to_dict())
    for (src, g) in df.groupby('src'):
        print(src, len(g), 'ham<=8', int((g.ham <= 8).sum()), 'same_text', int(g.same_text.sum()), 'sscd>=thr', int((g.sscd >= 0.828769862651825).sum()))
    print('clusters with fp', t[t.record_id.isin(set(df.a) | set(df.b))].lab.nunique())
    df.drop(columns=['ia', 'ib']).to_csv(ANALYSIS / 'benchmark_false_joins.csv', index=False)
    print(df.sort_values('ham').head(8)[['a', 'b', 'src', 'ham', 'sscd', 'ta', 'tb']].to_string())
    print(df.sort_values('ham').tail(8)[['a', 'b', 'src', 'ham', 'sscd', 'ta', 'tb']].to_string())
