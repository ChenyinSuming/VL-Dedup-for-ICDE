import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT
from pathlib import Path
import sys
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from scripts import run_graph_baselines as g

def main():
    frozen = g.read(g.REPORT / 'selection_frozen.json')
    a1 = g.read(g.OUT / 'test_A1.json')
    timing = g.read(g.OUT / 'timing_A1.json')
    a2 = g.read(g.OUT / 'test_A2_SemDeDup.json')['cells'] + g.read(g.OUT / 'test_A2_Minhash.json')['cells']
    result = dict(selection=frozen, A1=a1['cells'], A1_timing=timing['runs'], A2=a2)
    g.dump(g.REPORT / 'results.json', result)
if __name__ == '__main__':
    main()
