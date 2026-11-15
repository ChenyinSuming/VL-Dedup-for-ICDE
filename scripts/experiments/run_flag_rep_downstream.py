#!/usr/bin/env python3
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
"""FLAG_REP_RUN controller and worker using unchanged CAPTION_DOWNSTREAM_RUN ClipCap train/evaluate bodies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_flag_rep_export import RUN, CAPTION_DOWNSTREAM_RUN, REPORT, RES, dump, load

PYTHON = str(CAPTION_DOWNSTREAM_RUN / 'caption-env/bin/python')
GPU_LOCK = str(RES / 'locks/gpu.lock')
METHODS = ('b_fr_cov', 'clipscore_fr', 'b_fr', 'random_fr')


def flag_rep_gate(run, seed, method):
    if Path(run) != RUN or load(RUN / 'training_sets_complete.json')['status'] != 'complete':
        raise RuntimeError('FLAG_REP_RUN training sets are not complete')
    if not 20261104 <= seed <= 20261113:
        raise RuntimeError('Seed outside frozen ten subsets')
    if method == 'b_cov' and seed == 20261104:
        return
    if method not in METHODS or not load(RUN / 'retraining_gate.json')['passed']:
        raise RuntimeError('FLAG_REP_RUN retraining correctness gate failed')


def worker(command, seed, method):
    from scripts.experiments import run_caption_downstream_training as training
    # The CAPTION_DOWNSTREAM_RUN gate concerns an 80-run Clean-first campaign. FLAG_REP_RUN has its own
    # predeclared b_cov retraining gate; numerical train/generate bodies are reused.
    training.dependent_gate = flag_rep_gate
    subset = next(s for s in load(RUN / 'subsets.json')['subsets'] if s['seed'] == seed)
    args = SimpleNamespace(run=str(RUN), gpt2=str(CAPTION_DOWNSTREAM_RUN / 'caption_models/gpt2'),
           clip=str(CAPTION_DOWNSTREAM_RUN / 'caption_models/ViT-B-32.pt'), checkpoint_every=100,
           python=PYTHON, evaluator=str(ROOT / 'scripts/run_caption_downstream_eval.py'),
           evaluator_args=json.dumps(['--karpathy-json', str(RES / 'runs/caption_pilot/dataset_coco.json'),
             '--chair-source', str(RES / 'runs/caption_pilot/upstream_chair'),
             '--coco-annotations', str(CAPTION_DOWNSTREAM_RUN / 'coco_annotations'), '--java-home', str(CAPTION_DOWNSTREAM_RUN / 'caption-env')]))
    result = (training.train if command == 'train' else training.generate)(args, subset, method)
    if command == 'train' and result['steps'] != 4370:
        raise RuntimeError('FLAG_REP_RUN prescribed 4370-step gate failed')


def links():
    for target, source in [('training_sets/clean.parquet', 'training_sets/clean.parquet'),
                           ('training_sets/b_cov.parquet', 'training_sets/b_cov.parquet'),
                           ('test.parquet', 'test.parquet')]:
        destination = RUN / target
        if not destination.exists():
            destination.symlink_to(CAPTION_DOWNSTREAM_RUN / source)
    if load(RUN / 'subsets.json') != load(CAPTION_DOWNSTREAM_RUN / 'subsets.json'):
        raise RuntimeError('CAPTION_DOWNSTREAM_RUN subset membership drift')


def runtime_gate():
    scoring = load(RUN / 'scoring_complete.json')['elapsed_seconds']
    coverage = load(RUN / 'training_sets_complete.json').get('coverage_gpu_seconds', 0.)
    durations, current = [], 0.
    for seed, method in [(20261104, 'b_cov')] + [(s, m) for m in METHODS for s in range(20261104, 20261114)]:
        directory = RUN / 'training' / str(seed) / method
        train = directory / 'training_complete.json'
        evaluation = directory / 'evaluation_complete.json'
        if train.exists() and evaluation.exists():
            duration = load(train)['train_seconds'] + load(evaluation)['eval_seconds']
            durations.append(duration)
        else:
            if train.exists():
                current += load(train)['train_seconds']
            progress = directory / 'evaluation_progress.json'
            if progress.exists():
                current += load(progress).get('elapsed_seconds', 0.)
    per_run = sum(durations) / len(durations) if durations else 980.
    actual = scoring + coverage + sum(durations) + current
    projected = scoring + coverage + sum(durations) + per_run * (41 - len(durations))
    value = dict(passed=projected <= 14 * 3600 and actual <= 14 * 3600,
                 measured_gpu_seconds=actual, projected_gpu_seconds=projected,
                 completed_train_evals=len(durations), per_run_seconds=per_run)
    dump(RUN / 'runtime_gate.json', value)
    if not value['passed']:
        raise RuntimeError(f'14 GPU-hour gate failed: {value}')
    return value


def launch(command, seed, method):
    directory = RUN / 'training' / str(seed) / method
    marker = directory / ('training_complete.json' if command == 'train' else 'evaluation_complete.json')
    if marker.exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '4'
    env.update(NLTK_DATA=str(CAPTION_DOWNSTREAM_RUN / 'nltk_data'), TMPDIR=str(RUN / 'tmp'),
               XDG_CACHE_HOME=str(RUN / 'runtime_cache'), HF_HOME=str(RUN / 'runtime_cache/huggingface'),
               PYTHONDONTWRITEBYTECODE='1', VL_DEDUP_GPU_LEASE='1', TOKENIZERS_PARALLELISM='false')
    (RUN / 'tmp').mkdir(exist_ok=True)
    state = dict(status='running', phase=command, seed=seed, method=method, pid=os.getpid(),
                 updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    dump(RUN / 'workflow.json', state)
    with (directory / f'{command}.log').open('ab') as log:
        subprocess.run(['flock', GPU_LOCK, PYTHON, '-I', str(Path(__file__).resolve()), command,
                        '--seed', str(seed), '--method', method], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def retraining_gate():
    new = load(RUN / 'training/20261104/b_cov/evaluation_complete.json')
    old = load(CAPTION_DOWNSTREAM_RUN / 'training/20261104/b_cov/evaluation_complete.json')
    differences = {metric: new['metrics'][metric] - old['metrics'][metric] for metric in old['metrics']}
    value = dict(passed=abs(differences['CIDEr']) <= .02, seed=20261104,
                 old_metrics=old['metrics'], new_metrics=new['metrics'], differences=differences,
                 train_seconds=load(RUN / 'training/20261104/b_cov/training_complete.json')['train_seconds'],
                 eval_seconds=new['eval_seconds'], limit=.02)
    dump(RUN / 'retraining_gate.json', value)
    dump(REPORT / 'downstream/retraining_gate.json', value)
    if not value['passed']:
        raise RuntimeError(f'Retraining CIDEr difference {differences["CIDEr"]} exceeds .02')


def campaign():
    if load(RUN / 'training_sets_complete.json')['status'] != 'complete':
        raise RuntimeError('Construct and commit B1 training sets first')
    links()
    runtime_gate()
    launch('train', 20261104, 'b_cov')
    launch('generate', 20261104, 'b_cov')
    retraining_gate()
    runtime_gate()
    for method in METHODS:
        for seed in range(20261104, 20261114):
            runtime_gate()
            launch('train', seed, method)
            launch('generate', seed, method)
            runtime_gate()
    dump(RUN / 'training_campaign_complete.json', dict(status='complete', seeds=list(range(20261104, 20261114)),
            methods=list(METHODS), trained_evaluated=40, retraining_controls=1))
    dump(RUN / 'workflow.json', dict(status='complete', phase='B2', pid=os.getpid()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['campaign', 'train', 'generate'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--method')
    args = parser.parse_args()
    try:
        if args.command == 'campaign':
            campaign()
        else:
            worker(args.command, args.seed, args.method)
    except Exception as exc:
        dump(RUN / 'stop.json', dict(status='stopped', stage='B2', error_type=type(exc).__name__, error=str(exc)))
        raise
