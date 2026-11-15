"""Exercise the official SPICE field contract without loading real scorers."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest

spec = importlib.util.spec_from_file_location('caption_downstream_eval', Path(__file__).resolve().parents[1] / 'scripts/experiments/run_caption_downstream_eval.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_official_spice_recall_field_and_macro_average(tmp_path, monkeypatch):
    class Tokenizer:
        def tokenize(self, values):
            return values

    class Scorer:
        def compute_score(self, refs, captions):
            return .5, [.4, .6]

    class Bleu:
        def __init__(self, n):
            assert n == 4

        def compute_score(self, refs, captions):
            return [.1, .2, .3, .4], [[.1, .1], [.2, .2], [.3, .3], [.4, .4]]

    class Spice:
        def compute_score(self, refs, captions):
            return .4, [{'All': {'f': .3, 'pr': .6, 're': .2}},
                        {'All': {'f': .5, 'pr': .7, 're': .4}}]

    modules = [('tokenizer.ptbtokenizer', 'PTBTokenizer', Tokenizer),
               ('bleu.bleu', 'Bleu', Bleu), ('cider.cider', 'Cider', Scorer),
               ('meteor.meteor', 'Meteor', Scorer), ('spice.spice', 'Spice', Spice)]
    monkeypatch.setitem(sys.modules, 'pycocoevalcap', types.ModuleType('pycocoevalcap'))
    for path, name, cls in modules:
        module = types.ModuleType('pycocoevalcap.' + path)
        setattr(module, name, cls)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(evaluation, 'chair_evaluate', lambda *a: dict(
        overall_metrics={'CHAIRi': .1, 'CHAIRs': .2}, sentences=[]))
    refs = {i: [{'caption': 'reference'}] for i in [1, 2]}
    res = {i: [{'caption': 'prediction'}] for i in [1, 2]}
    result = dict(metrics={}, timings_seconds={}, per_image=[])
    evaluation.evaluate(types.SimpleNamespace(java_home=tmp_path), [1, 2], refs, res, result, tmp_path)
    assert result['metrics']['SPICE recall'] == pytest.approx(.3)
    assert [r['SPICE recall'] for r in result['per_image']] == [.2, .4]
