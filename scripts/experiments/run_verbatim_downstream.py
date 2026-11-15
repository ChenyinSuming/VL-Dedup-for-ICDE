#!/usr/bin/env python3
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
"""VERBATIM_EXPORT_RUN resumable campaign using unchanged CAPTION_DOWNSTREAM_RUN numerical train/evaluate bodies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiments.run_verbatim_export import RUN, REPORT, CAPTION_DOWNSTREAM_RUN, FLAG_REP_RUN, RES, METHODS, dump, load

PYTHON=str(CAPTION_DOWNSTREAM_RUN/'caption-env/bin/python')
GPU_LOCK=str(RES/'locks/gpu.lock')
CONTROL='b_fr_cov'
SEEDS=range(20261104,20261114)


def dependent_gate(run,seed,method):
    if Path(run)!=RUN or load(RUN/'training_sets_complete.json')['status']!='complete':
        raise RuntimeError('VERBATIM_EXPORT_RUN training-set completion gate failed')
    if seed not in SEEDS:
        raise RuntimeError('Seed outside frozen subsets')
    if method==CONTROL and seed==20261104:
        return
    if method not in METHODS or not load(RUN/'retraining_gate.json')['passed']:
        raise RuntimeError('VERBATIM_EXPORT_RUN retraining correctness gate failed')


def worker(command,seed,method):
    from scripts.experiments import run_caption_downstream_training as training
    training.dependent_gate=dependent_gate
    subset=next(s for s in load(RUN/'subsets.json')['subsets'] if s['seed']==seed)
    args=SimpleNamespace(run=str(RUN),gpt2=str(CAPTION_DOWNSTREAM_RUN/'caption_models/gpt2'),
        clip=str(CAPTION_DOWNSTREAM_RUN/'caption_models/ViT-B-32.pt'),checkpoint_every=100,python=PYTHON,
        evaluator=str(ROOT/'scripts/run_verbatim_downstream_eval.py'),
        evaluator_args=json.dumps(['--karpathy-json',str(RES/'runs/caption_pilot/dataset_coco.json'),
          '--chair-source',str(RES/'runs/caption_pilot/upstream_chair'),
          '--coco-annotations',str(CAPTION_DOWNSTREAM_RUN/'coco_annotations'),'--java-home',str(CAPTION_DOWNSTREAM_RUN/'caption-env')]))
    value=(training.train if command=='train' else training.generate)(args,subset,method)
    if command=='train' and value['steps']!=4370:
        raise RuntimeError('Prescribed 4370-step gate failed')


def links():
    for target,source in [('training_sets/clean.parquet',CAPTION_DOWNSTREAM_RUN/'training_sets/clean.parquet'),
                          ('training_sets/b_fr_cov.parquet',FLAG_REP_RUN/'training_sets/b_fr_cov.parquet'),
                          ('test.parquet',CAPTION_DOWNSTREAM_RUN/'test.parquet')]:
        dest=RUN/target
        if not dest.exists():dest.symlink_to(source)
    if load(RUN/'subsets.json')!=load(CAPTION_DOWNSTREAM_RUN/'subsets.json'):
        raise RuntimeError('Frozen CAPTION_DOWNSTREAM_RUN subset drift')


def runtime_gate():
    durations=[];current=0.
    for seed,method in [(20261104,CONTROL)]+[(s,m) for m in METHODS for s in SEEDS]:
        d=RUN/'training'/str(seed)/method
        train=d/'training_complete.json'; evaluation=d/'evaluation_complete.json'
        if train.exists() and evaluation.exists():
            durations.append(load(train)['train_seconds']+load(evaluation)['eval_seconds'])
        else:
            if train.exists():current+=load(train)['train_seconds']
            progress=d/'evaluation_progress.json'
            if progress.exists():current+=load(progress).get('elapsed_seconds',0.)
    mean=sum(durations)/len(durations) if durations else 964.
    actual=sum(durations)+current
    projected=sum(durations)+mean*(41-len(durations))
    value=dict(passed=actual<=14*3600 and projected<=14*3600,measured_gpu_seconds=actual,
               projected_gpu_seconds=projected,completed_train_evals=len(durations),per_run_seconds=mean)
    dump(RUN/'runtime_gate.json',value)
    if not value['passed']:raise RuntimeError(f'14 GPU-hour gate failed: {value}')
    return value


def launch(command,seed,method):
    directory=RUN/'training'/str(seed)/method
    marker=directory/('training_complete.json' if command=='train' else 'evaluation_complete.json')
    if marker.exists():return
    directory.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='4'
    env.update(NLTK_DATA=str(CAPTION_DOWNSTREAM_RUN/'nltk_data'),TMPDIR=str(RUN/'tmp'),XDG_CACHE_HOME=str(RUN/'runtime_cache'),
       HF_HOME=str(RUN/'runtime_cache/huggingface'),PYTHONDONTWRITEBYTECODE='1',VL_DEDUP_GPU_LEASE='1',TOKENIZERS_PARALLELISM='false')
    (RUN/'tmp').mkdir(exist_ok=True)
    dump(RUN/'workflow.json',dict(status='running',phase=command,seed=seed,method=method,pid=os.getpid(),
         updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    with (directory/f'{command}.log').open('ab') as log:
        subprocess.run(['flock',GPU_LOCK,PYTHON,'-I',str(Path(__file__).resolve()),command,
                        '--seed',str(seed),'--method',method],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def retraining_gate():
    new=load(RUN/f'training/20261104/{CONTROL}/evaluation_complete.json')
    old=load(FLAG_REP_RUN/f'training/20261104/{CONTROL}/evaluation_complete.json')
    differences={m:new['metrics'][m]-old['metrics'][m] for m in old['metrics']}
    value=dict(passed=abs(differences['CIDEr'])<=.02,seed=20261104,method=CONTROL,
         old_metrics=old['metrics'],new_metrics=new['metrics'],differences=differences,
         train_seconds=load(RUN/f'training/20261104/{CONTROL}/training_complete.json')['train_seconds'],
         eval_seconds=new['eval_seconds'],limit=.02)
    dump(RUN/'retraining_gate.json',value);dump(REPORT/'downstream/retraining_gate.json',value)
    if not value['passed']:raise RuntimeError(f'Retraining gate failed: {differences}')


def campaign():
    if load(RUN/'training_sets_complete.json')['status']!='complete':raise RuntimeError('Complete B1 first')
    (RUN/'nvidia-smi-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    links();runtime_gate()
    launch('train',20261104,CONTROL);launch('generate',20261104,CONTROL)
    retraining_gate();runtime_gate()
    for method in METHODS:
        for seed in SEEDS:
            runtime_gate();launch('train',seed,method);launch('generate',seed,method);runtime_gate()
    dump(RUN/'training_campaign_complete.json',dict(status='complete',methods=list(METHODS),seeds=list(SEEDS),
         trained_evaluated=40,retraining_controls=1))
    dump(RUN/'workflow.json',dict(status='complete',phase='B2',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['campaign','train','generate']);p.add_argument('--seed',type=int);p.add_argument('--method');a=p.parse_args()
    try:
        campaign() if a.command=='campaign' else worker(a.command,a.seed,a.method)
    except Exception as exc:
        dump(RUN/'stop.json',dict(status='stopped',stage='B2',error_type=type(exc).__name__,error=str(exc)))
        raise
