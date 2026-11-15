import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if __name__ == '__main__':
    from scripts.vl_paths import REPO_ROOT, ANALYSIS
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    exec(open(REPO_ROOT / 'scripts/analysis/precision_target_sweep.py').read().split('def graph(')[0])
    lv = np.unique(score[recalled])
    print('levels >= 0.3:')
    for t in lv[lv >= 0.3]:
        m = binary_metrics(yv, recalled & (score >= t))
        print(round(float(t), 6), 'accepted', m['tp'] + m['fp'], 'tp', m['tp'], 'fp', m['fp'], 'P', round(m['precision'], 6), 'R', round(m['recall'], 6))
    for pi in [0.8, 0.85, 0.9, 0.93, 0.95, 0.99, 0.995, 0.997, 0.998, 0.999, 1.0]:
        print('pi', pi, 'tau', choose(yv, score, recalled, pi))
