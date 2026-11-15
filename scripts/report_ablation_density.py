import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import Counter
from fractions import Fraction
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('ablation_density', Path(__file__).with_name('run_ablation_density.py'))
ablation_density = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation_density)
(OUT, REPORT) = (ablation_density.OUT, ablation_density.REPORT)
(read, dump) = (ablation_density.read, ablation_density.dump)

def independent_counts(labels, truth, ids):
    truth = truth[truth.original_level.ne('M')]
    c = labels[ids.get_indexer(truth.record_id)]
    g = truth.visual_group_id.tolist()
    choose = lambda values: sum((n * (n - 1) // 2 for n in values))
    tp = choose(Counter(zip(c, g)).values())
    fp = choose(Counter(c).values()) - tp
    fn = choose(Counter(g).values()) - tp
    return dict(tp=tp, fp=fp, fn=fn, f1=float(Fraction(2 * tp, 2 * tp + fp + fn)))

def main():
    result = dict(feature_groups=ablation_density.GROUPS, ablation_selection=read(REPORT / 'ablation_selection_frozen.json'), ablation=read(REPORT / 'ablation_results.json'), retrieval=read(REPORT / 'retrieval_results.json'), density_selection=read(REPORT / 'density_selection_frozen.json'), density=read(REPORT / 'density_results.json'), curves=read(REPORT / 'density_curves.json'))
    dump(REPORT / 'results.json', result)
if __name__ == '__main__':
    with ablation_density.stage('report'):
        main()
