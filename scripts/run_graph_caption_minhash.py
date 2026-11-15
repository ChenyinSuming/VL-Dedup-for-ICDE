import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT
import importlib
import importlib.metadata as md
from pathlib import Path
import sys
import types
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from scripts import run_graph_baselines as g
import numpy as np
import pandas as pd

def namespace(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module

def official_operator():
    assert md.version('py-data-juicer') == '1.5.3'
    site = Path(md.distribution('py-data-juicer').locate_file('')) / 'data_juicer'
    for (name, path) in [('data_juicer', site), ('data_juicer.utils', site / 'utils'), ('data_juicer.ops', site / 'ops'), ('data_juicer.ops.common', site / 'ops/common'), ('data_juicer.ops.deduplicator', site / 'ops/deduplicator')]:
        namespace(name, path)

    class Registry:

        def register_module(self, *args, **kwargs):
            return lambda cls: cls

    class Base:

        def __init__(self, *args, **kwargs):
            self.text_key = kwargs.get('text_key', 'text')
    base = types.ModuleType('data_juicer.ops.base_op')
    (base.OPERATORS, base.Deduplicator) = (Registry(), Base)
    sys.modules[base.__name__] = base
    constant = types.ModuleType('data_juicer.utils.constant')
    constant.HashKeys = type('HashKeys', (), {'minhash': '__dj__minhash'})
    sys.modules[constant.__name__] = constant
    lazy = types.ModuleType('data_juicer.utils.lazy_loader')
    lazy.LazyLoader = importlib.import_module
    sys.modules[lazy.__name__] = lazy
    model = types.ModuleType('data_juicer.utils.model_utils')

    def unused_sentencepiece(*args, **kwargs):
        raise RuntimeError('Default space splitter must not load sentencepiece')
    model.prepare_sentencepiece_model = unused_sentencepiece
    sys.modules[model.__name__] = model
    module = importlib.import_module('data_juicer.ops.deduplicator.document_minhash_deduplicator')
    return (module.DocumentMinhashDeduplicator(), module.__file__)

class NativeDataset(list):

    def __getitem__(self, key):
        return [row[key] for row in self] if isinstance(key, str) else super().__getitem__(key)

    def remove_columns(self, columns):
        return NativeDataset(({k: v for (k, v) in row.items() if k not in columns} for row in self))

    def filter(self, function, with_indices=False, **kwargs):
        assert with_indices
        selected = [i for (i, row) in enumerate(self) if function(row, i)]
        (union_find,) = [c.cell_contents for c in function.__closure__ if hasattr(c.cell_contents, 'parent') and hasattr(c.cell_contents, 'find')]
        self.labels = np.asarray([union_find.find(i) for i in range(len(self))], dtype=np.int32)
        result = NativeDataset((self[i] for i in selected))
        result.labels = self.labels
        assert len(selected) == len(set(self.labels))
        return result

def main():
    output = dict(run_id=g.OUT.name, implementation='Data-Juicer 1.5.3 default document_minhash_deduplicator', cells=[])
    for surface in ['original', 'MV']:
        frame = pd.read_parquet((g.SOURCE if surface == 'original' else g.MV) / 'public_manifest.parquet')
        ids = pd.Index(frame.record_id)
        target = g.OUT / 'caption_Minhash' / surface
        import time
        (start, cpu) = (time.perf_counter(), time.process_time())
        (op, source) = official_operator()
        samples = NativeDataset()
        for (index, (record_id, text)) in enumerate(zip(frame.record_id, frame.text)):
            samples.append(op.compute_hash({'record_id': record_id, 'text': text}))
        (selected, traced) = op.process(samples)
        timing = dict(wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu)
        labels = selected.labels
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / 'components.npy', labels)
        for evaluation_surface in ['original', 'mixed', 'mild', 'strong'] if surface == 'original' else ['MV']:
            output['cells'].append(dict(method='caption_Minhash', surface=evaluation_surface, components=str(target / 'components.npy'), timing=timing, **g.corpus_metrics(labels, ids, evaluation_surface)))
        g.dump(g.OUT / 'test_A2_Minhash.json', output)
if __name__ == '__main__':
    main()
