#!/usr/bin/env python3
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
"""CPU-only official caption evaluation; check-only never launches Java or scorers.

Input: COCO list [{"image_id": int, "caption": str}], exactly Karpathy test5000.
Run with the independent CAPTION_DOWNSTREAM_RUN caption environment, never the NOISY_POOL_RUN environment.
"""
import argparse
import ast
import contextlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True

NOISY_POOL_RUN = (RESOURCES / 'runs/caption_pilot')
CAPTION_DOWNSTREAM_RUN = (RESOURCES / 'runs/caption_downstream')
METRICS = ['CIDEr', 'BLEU-4', 'METEOR', 'SPICE', 'SPICE recall', 'CHAIR_i', 'CHAIR_s']


def read_contract(captions, karpathy):
    images = json.loads(karpathy.read_text())['images']
    test = [im for im in images if im['split'] == 'test']
    ids = [im['cocoid'] for im in test]
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise ValueError('Original Karpathy test membership must contain 5000 unique COCO IDs')
    refs = {}
    for im in test:
        sentences = im['sentences']
        if len(sentences) < 5 or any(not isinstance(s.get('raw'), str) for s in sentences):
            raise ValueError('Each test image requires every original raw reference (at least five)')
        refs[im['cocoid']] = [{'caption': s['raw']} for s in sentences]
    rows = json.loads(captions.read_text())
    if not isinstance(rows, list) or any(
        not isinstance(r, dict) or type(r.get('image_id')) is not int
        or not isinstance(r.get('caption'), str) or not r['caption'].strip() for r in rows
    ):
        raise ValueError('Captions must be a COCO list with integer image_id and nonempty caption')
    predicted = [r['image_id'] for r in rows]
    if len(predicted) != 5000 or len(set(predicted)) != 5000 or set(predicted) != set(ids):
        raise ValueError('Predictions must match all and only original Karpathy test5000 IDs')
    return sorted(ids), refs, {r['image_id']: [{'caption': r['caption']}] for r in rows}


def dependencies(args):
    missing = []
    for name in ['pycocoevalcap', 'numpy', 'nltk', 'pattern', 'lib2to3']:
        if importlib.util.find_spec(name) is None:
            missing.append('Python module: ' + name)
    java = args.java_home / 'bin/java'
    if not args.java_home.is_relative_to(CAPTION_DOWNSTREAM_RUN):
        missing.append('JAVA_HOME must belong to the independent CAPTION_DOWNSTREAM_RUN environment')
    if not java.is_file() or not os.access(java, os.X_OK):
        missing.append('Executable: ' + str(java))
    chair = args.chair_source / 'utils/chair.py'
    for path in [chair, args.chair_source / 'data/synonyms.txt'] + [
        args.coco_annotations / (kind + '_' + split + '2014.json')
        for kind in ['captions', 'instances'] for split in ['train', 'val']
    ]:
        if not path.is_file():
            missing.append('File: ' + str(path))
    spec = importlib.util.find_spec('pycocoevalcap')
    if spec is not None:
        root = Path(next(iter(spec.submodule_search_locations))).resolve()
        if not root.is_relative_to(CAPTION_DOWNSTREAM_RUN):
            missing.append('pycocoevalcap must be installed in the independent CAPTION_DOWNSTREAM_RUN environment: ' + str(root))
        for rel in ['tokenizer/stanford-corenlp-3.4.1.jar', 'meteor/meteor-1.5.jar',
                    'meteor/data/paraphrase-en.gz', 'spice/spice-1.0.jar',
                    'spice/lib/stanford-corenlp-3.6.0.jar',
                    'spice/lib/stanford-corenlp-3.6.0-models.jar']:
            if not (root / rel).is_file():
                missing.append('File: ' + str(root / rel))
    if importlib.util.find_spec('nltk') is not None:
        import nltk
        # Pattern checks these directory names on import; a zip-only WordNet
        # makes it invoke the downloader even though NLTK itself can read the zip.
        for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab/english/',
                         'corpora/wordnet', 'corpora/wordnet_ic', 'corpora/sentiwordnet']:
            try:
                nltk.data.find(resource)
            except LookupError:
                missing.append('NLTK data: ' + resource)
    return missing


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('A required metric is nonfinite; no zero substitution is permitted')
    return value


def chair_evaluate(args, ids, res, metrics, per_image, temp):
    # Upstream is Python 2. Convert syntax only, without changing its scoring rules.
    # misc imports unrelated LM/IM modules and imports chair recursively; its only
    # name used by chair.py is os. Supply that explicitly rather than loading misc.
    from lib2to3.refactor import RefactoringTool, get_fixers_from_package
    source = (args.chair_source / 'utils/chair.py').read_text()
    converted = str(RefactoringTool(get_fixers_from_package('lib2to3.fixes')).refactor_string(source, 'chair.py'))
    tree = ast.parse(converted)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == 'misc')]
    namespace = {'__name__': 'caption_downstream_original_chair', 'os': os}
    exec(compile(tree, str(args.chair_source / 'utils/chair.py'), 'exec'), namespace)
    # The upstream container requires these unused reporting keys; None records
    # that ROUGE and BLEU1-3 are not requested, rather than inventing scores.
    overall = {'Bleu_1': None, 'Bleu_2': None, 'Bleu_3': None, 'ROUGE_L': None,
               'Bleu_4': metrics['BLEU-4'], **{k: metrics[k] for k in ['CIDEr', 'METEOR', 'SPICE']}}
    payload = {'overall': overall, 'imgToEval': {
        str(i): {'image_id': i, 'caption': res[i][0]['caption'], **per_image[i],
                 'Bleu_1': None, 'Bleu_2': None, 'Bleu_3': None,
                 'Bleu_4': per_image[i]['BLEU-4'], 'ROUGE_L': None} for i in ids}}
    cap_file = temp / 'chair_input.json'
    cap_file.write_text(json.dumps(payload))
    old = Path.cwd()
    try:
        os.chdir(args.chair_source)
        evaluator = namespace['CHAIR'](ids, str(args.coco_annotations))
        evaluator.get_annotations()
        return evaluator.compute_chair(str(cap_file))
    finally:
        os.chdir(old)


def evaluate(args, ids, refs, res, result, temp):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.spice.spice import Spice
    os.environ['JAVA_HOME'] = str(args.java_home)
    os.environ['PATH'] = str(args.java_home / 'bin') + os.pathsep + os.environ.get('PATH', '')
    per_image = {i: {'image_id': i} for i in ids}
    timings = result['timings_seconds']
    started = time.monotonic()
    tokenizer = PTBTokenizer()
    refs = {i: refs[i] for i in ids}
    res = {i: res[i] for i in ids}
    refs, token_res = tokenizer.tokenize(refs), tokenizer.tokenize(res)
    timings['PTBTokenizer'] = time.monotonic() - started
    for label, factory in [('CIDEr', Cider), ('BLEU-4', lambda: Bleu(4)), ('METEOR', Meteor), ('SPICE', Spice)]:
        started = time.monotonic()
        scorer = factory()
        try:
            score, values = scorer.compute_score(refs, token_res)
            if label == 'BLEU-4':
                score, values = score[3], values[3]
            result['metrics'][label] = finite(score)
            if len(values) != len(ids):
                raise ValueError(label + ' per-image count mismatch')
            for i, value in zip(ids, values):
                if label == 'SPICE':
                    per_image[i]['SPICE_categories'] = value
                    per_image[i][label] = finite(value['All']['f'])
                    per_image[i]['SPICE recall'] = finite(value['All']['re'])
                else:
                    per_image[i][label] = finite(value)
            if label == 'SPICE':
                result['metrics']['SPICE recall'] = sum(per_image[i]['SPICE recall'] for i in ids) / len(ids)
        finally:
            timings[label] = time.monotonic() - started
            if label == 'METEOR' and hasattr(scorer, 'meteor_p'):
                scorer.meteor_p.kill()
                scorer.meteor_p.wait()
        result['per_image'] = list(per_image.values())
    started = time.monotonic()
    chair = chair_evaluate(args, ids, res, result['metrics'], per_image, temp)
    timings['CHAIR'] = time.monotonic() - started
    result['chair_raw'] = chair
    result['metrics']['CHAIR_i'] = finite(chair['overall_metrics']['CHAIRi'])
    result['metrics']['CHAIR_s'] = finite(chair['overall_metrics']['CHAIRs'])
    for row in chair['sentences']:
        per_image[row['image_id']]['CHAIR'] = row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captions', '--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--karpathy-json', type=Path, default=NOISY_POOL_RUN / 'dataset_coco.json')
    parser.add_argument('--chair-source', type=Path, default=NOISY_POOL_RUN / 'upstream_chair')
    parser.add_argument('--coco-annotations', type=Path, default=CAPTION_DOWNSTREAM_RUN / 'resources/coco2014/annotations')
    parser.add_argument('--java-home', type=Path, default=CAPTION_DOWNSTREAM_RUN / 'caption-env')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    for name in ['captions', 'output', 'karpathy_json', 'chair_source', 'coco_annotations', 'java_home']:
        setattr(args, name, getattr(args, name).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'blocked', 'metrics': {k: None for k in METRICS},
              'timings_seconds': {}, 'per_image': [], 'paths': {
                  k: str(getattr(args, k)) for k in ['captions', 'karpathy_json', 'chair_source', 'coco_annotations', 'java_home']},
              'implementation': 'official pycocoevalcap/PTBTokenizer; original LisaAnne/Hallucination CHAIR, Python3 syntax adaptation; CIDEr raw; SPICE recall macro All.re',
              'hash_fields': 'N/A: no file hashes'}
    started = time.monotonic()
    exit_code = 1
    try:
        ids, refs, res = read_contract(args.captions, args.karpathy_json)
        if args.karpathy_json != (NOISY_POOL_RUN / 'dataset_coco.json'):
            original_ids, original_refs, _ = read_contract(args.captions, NOISY_POOL_RUN / 'dataset_coco.json')
            if ids != original_ids or refs != original_refs:
                raise ValueError('Supplied Karpathy references differ from immutable NOISY_POOL_RUN original')
        result['contract'] = {'images': len(ids), 'reference_count': sum(map(len, refs.values())), 'reference_count_histogram': {str(n): sum(len(v) == n for v in refs.values()) for n in sorted({len(v) for v in refs.values()})}, 'membership': 'original Karpathy test', 'images_with_extra_references': [i for i in ids if len(refs[i]) > 5]}
        result['missing_dependencies'] = dependencies(args)
        if result['missing_dependencies']:
            result['reason'] = 'Missing dependencies; no scorers launched or resources downloaded'
        elif args.check_only:
            result['status'] = 'dependencies_ready'
            exit_code = 0
        else:
            # Preserve Python scorer output alongside the machine-readable JSON.
            with tempfile.TemporaryDirectory(prefix='caption_eval_', dir=args.output.parent) as work:
                with args.output.with_suffix('.log').open('w') as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        evaluate(args, ids, refs, res, result, Path(work))
            result['status'] = 'complete'
            exit_code = 0
    except Exception as exc:
        result['reason'] = str(exc)
        result['traceback'] = traceback.format_exc()
    finally:
        result['timings_seconds']['total'] = time.monotonic() - started
        # Nested SPICE categories may contain NaN for undefined subcategories.
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean(v) for v in obj]
            if hasattr(obj, 'item'):
                obj = obj.item()
            return None if isinstance(obj, float) and not math.isfinite(obj) else obj
        args.output.write_text(json.dumps(clean(result), indent=2, allow_nan=False) + '\n')
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
