#!/usr/bin/env python3
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
"""CAPTION_DOWNSTREAM_RUN local-only ClipCap runner. Invoke with the isolated RUN environment.

Inputs: training_sets/{method}.parquet: record_id, original_image_id,
image_path,text; test.parquet: original_image_id,image_path (5000 unique rows).
Local models: --gpt2 directory (language-pretrained GPT2 small only), --clip
OpenAI ViT-B/32 .pt. No caption checkpoints accepted or downloaded here.
Evaluator contract: Python script --predictions PATH --output PATH plus
--evaluator-args JSON list; returns status=complete and finite METRICS mapping.
The evaluator owns official PTB/COCO/SPICE/CHAIR integration, not this runner.
Every GPU command requires VL_DEDUP_GPU_LEASE=1; caller must hold campaign flock.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True

METHODS = ('no_dedup', 'keep_one', 'semdedup', 'clipscore', 'random', 'b', 'b_cov', 'clean')
METRICS = ('CIDEr', 'BLEU-4', 'METEOR', 'SPICE', 'SPICE recall', 'CHAIR_i', 'CHAIR_s')
SEEDS = list(range(20261104, 20261114))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def gpu_lease():
    if os.environ.get('VL_DEDUP_GPU_LEASE') != '1':
        raise RuntimeError('GPU phase requires VL_DEDUP_GPU_LEASE=1 under campaign flock')


def subset_memberships(ids):
    ids = sorted(set(int(x) for x in ids))
    if len(ids) != 5000:
        raise ValueError('Expected exactly 5000 clean training image IDs')
    return {'seeds': SEEDS, 'images_each': 3500, 'subsets': [
        {'seed': seed, 'image_ids': sorted(random.Random(seed).sample(ids, 3500))}
        for seed in SEEDS]}


def prescribed_steps(clean_count):
    return 10 * (clean_count // 40)


def timing_decision(train_seconds, eval_seconds):
    reason = None
    if train_seconds * 80 > 30 * 3600:
        reason = 'training_80_runs_exceeds_30_hours'
    elif (train_seconds + eval_seconds) * 80 > 35 * 3600:
        reason = 'train_eval_80_runs_exceeds_35_hours'
    count = 5 if reason else 10
    projected = (train_seconds + eval_seconds) * (8 * count)
    return {'subset_count': count, 'reduction_reason': reason, 'projected_train_eval_seconds': projected,
            'passed': projected <= 35 * 3600}


def dependent_gate(run, seed, method):
    if seed == SEEDS[0] and method == 'clean':
        return
    path = Path(run) / 'caption_training_gate.json'
    if not path.exists():
        raise RuntimeError('Complete Clean subset1 correctness/time gate first')
    gate = json.loads(path.read_text())
    if not gate['passed'] or not gate['correctness_passed']:
        raise RuntimeError('Clean correctness/time gate failed')
    if seed not in SEEDS[:gate['subset_count']]:
        raise RuntimeError('Subset excluded by prescribed time reduction')


def load_rows(path):
    import pandas as pd
    frame = pd.read_parquet(path)
    required = {'record_id', 'original_image_id', 'image_path', 'text'}
    if not required.issubset(frame.columns):
        raise ValueError(f'{path}: required columns {sorted(required)}')
    if frame.record_id.duplicated().any() or frame[list(required)].isnull().any().any():
        raise ValueError(f'{path}: null fields or duplicate record_id')
    return frame.sort_values('record_id').to_dict('records')


def prepare(run):
    rows = load_rows(run / 'training_sets/clean.parquet')
    if len(rows) != 25009:
        raise ValueError('Expected 25009 original clean captions')
    value = subset_memberships(r['original_image_id'] for r in rows)
    path = run / 'subsets.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Existing frozen subset membership differs')
    if not path.exists():
        write_json(path, value)
    return value


def runtime(args, directory=None):
    gpu_lease()
    if directory is None:
        directory = Path(args.run) / 'logs'
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
    (directory / f'nvidia-smi-{os.getpid()}-{time.time_ns()}.txt').write_text(snapshot.stdout)
    import torch
    import clip
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    if not Path(args.gpt2).is_dir() or not Path(args.clip).is_file():
        raise ValueError('Only local GPT2 directory and OpenAI CLIP .pt accepted')
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2, local_files_only=True)
    encoder, preprocess = clip.load(args.clip, device='cuda', jit=False)
    encoder.eval().requires_grad_(False)
    gpt = GPT2LMHeadModel.from_pretrained(args.gpt2, local_files_only=True)
    if gpt.config.n_embd != 768 or gpt.config.n_layer != 12:
        raise ValueError('Expected GPT2 small')
    class CaptionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gpt = gpt
            self.mapping = torch.nn.Sequential(torch.nn.Linear(512, 3840),
                torch.nn.Tanh(), torch.nn.Linear(3840, 7680))
        def prefix(self, features):
            return self.mapping(features).reshape(-1, 10, 768)
    return torch, tokenizer, encoder, preprocess, CaptionModel().cuda()


def encode_rows(torch, encoder, preprocess, rows, progress):
    from PIL import Image
    paths = sorted(set(str(r['image_path']) for r in rows))
    features = {}
    for offset in range(0, len(paths), 40):
        batch = paths[offset:offset + 40]
        images = []
        for path in batch:
            with Image.open(path) as image:
                images.append(preprocess(image.convert('RGB')))
        with torch.no_grad():
            values = encoder.encode_image(torch.stack(images).cuda()).float().cpu()
        if values.shape[1] != 512:
            raise ValueError('Expected 512-dimensional frozen ViT-B/32 features')
        features.update(zip(batch, values))
        write_json(progress, {'pid': os.getpid(), 'phase': 'encode', 'images': offset + len(batch)})
    return features


def train(args, subset, method):
    run = Path(args.run)
    seed = subset['seed']
    dependent_gate(run, seed, method)
    directory = run / 'training' / str(seed) / method
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / 'training_complete.json'
    if final.exists():
        return json.loads(final.read_text())
    gpu_lease()
    started = time.monotonic()
    ids = set(subset['image_ids'])
    clean = [r for r in load_rows(run / 'training_sets/clean.parquet') if int(r['original_image_id']) in ids]
    rows = [r for r in load_rows(run / f'training_sets/{method}.parquet') if int(r['original_image_id']) in ids]
    if len(rows) < 40:
        raise ValueError('Retained subset must contain a complete batch of 40')
    total = prescribed_steps(len(clean))
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch, tok, encoder, preprocess, model = runtime(args, directory)
    features = encode_rows(torch, encoder, preprocess, rows, directory / 'progress.json')
    del encoder
    tokens = [tok.encode(r['text'])[:40] for r in rows]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, min(5000, total), total)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    position, step, elapsed = 0, 0, 0.0
    checkpoint = directory / 'checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['seed'] != seed or state['method'] != method or state['total_steps'] != total or state['record_ids'] != [r['record_id'] for r in rows]:
            raise ValueError('Checkpoint input contract changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        order, position, step, elapsed = state['order'], state['position'], state['step'], state['elapsed_seconds']
        rng.setstate(state['stream_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    model.train()
    def save():
        state = dict(seed=seed, method=method, total_steps=total, record_ids=[r['record_id'] for r in rows],
            model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            order=order, position=position, step=step, stream_rng=rng.getstate(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), elapsed_seconds=elapsed + time.monotonic() - started)
        temporary = checkpoint.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(checkpoint)
    while step < total:
        if position + 40 > len(order):
            rng.shuffle(order)
            position = 0
        indices = order[position:position + 40]
        position += 40
        token_ids = torch.zeros((40, 40), dtype=torch.long, device='cuda')
        mask = torch.zeros((40, 40), device='cuda')
        for i, index in enumerate(indices):
            length = len(tokens[index])
            token_ids[i, :length] = torch.tensor(tokens[index], device='cuda')
            mask[i, :length] = 1
        prefix = model.prefix(torch.stack([features[str(rows[i]['image_path'])] for i in indices]).cuda())
        embeddings = torch.cat((prefix, model.gpt.transformer.wte(token_ids)), dim=1)
        attention = torch.cat((torch.ones((40, 10), device='cuda'), mask), dim=1)
        logits = model.gpt(inputs_embeds=embeddings, attention_mask=attention).logits[:, 9:-1]
        losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), token_ids.flatten(), reduction='none')
        loss = (losses * mask.flatten()).sum() / mask.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        if step % args.checkpoint_every == 0 or step == total:
            torch.cuda.synchronize()
            save()
            write_json(directory / 'progress.json', {'pid': os.getpid(), 'phase': 'train', 'step': step, 'total_steps': total, 'loss': loss.item()})
    receipt = {'seed': seed, 'method': method, 'steps': total, 'records': len(rows), 'clean_records': len(clean),
               'train_seconds': elapsed + time.monotonic() - started, 'checkpoint': str(checkpoint)}
    write_json(final, receipt)
    return receipt


def generate(args, subset, method):
    import pandas as pd
    dependent_gate(args.run, subset['seed'], method)
    directory = Path(args.run) / 'training' / str(subset['seed']) / method
    output = directory / 'predictions.json'
    receipt_path = directory / 'evaluation_complete.json'
    if receipt_path.exists():
        return json.loads(receipt_path.read_text())
    gpu_lease()
    started = time.monotonic()
    timing_path = directory / 'evaluation_progress.json'
    prior_seconds = json.loads(timing_path.read_text()).get('elapsed_seconds', 0.0) if timing_path.exists() else 0.0
    if not (directory / 'training_complete.json').exists():
        raise ValueError('Final prescribed training checkpoint required')
    if not args.evaluator:
        raise ValueError('Official seven-metric evaluator executable is required')
    test = pd.read_parquet(Path(args.run) / 'test.parquet').sort_values('original_image_id').to_dict('records')
    if len(test) != 5000 or len({int(r['original_image_id']) for r in test}) != 5000:
        raise ValueError('Expected frozen 5000-image test membership')
    predictions = json.loads(output.read_text()) if output.exists() else []
    if [r['image_id'] for r in predictions] != [int(r['original_image_id']) for r in test[:len(predictions)]]:
        raise ValueError('Partial predictions do not match frozen test order')
    if len(predictions) < len(test):
        torch, tok, encoder, preprocess, model = runtime(args, directory)
        model.load_state_dict(torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=False)['model'])
        model.eval()
        features = encode_rows(torch, encoder, preprocess, test[len(predictions):], directory / 'progress.json')
        with torch.no_grad():
            for row in test[len(predictions):]:
                embedding = model.prefix(features[str(row['image_path'])].unsqueeze(0).cuda())
                generated = []
                for _ in range(40):
                    token = model.gpt(inputs_embeds=embedding).logits[:, -1].argmax(-1)
                    value = token.item()
                    if value == tok.eos_token_id:
                        break
                    generated.append(value)
                    if '.' in tok.decode(generated):
                        break
                    embedding = torch.cat((embedding, model.gpt.transformer.wte(token).unsqueeze(1)), dim=1)
                caption = tok.decode(generated).split('.', 1)[0]
                if '.' in tok.decode(generated):
                    caption += '.'
                predictions.append({'image_id': int(row['original_image_id']), 'caption': caption})
                write_json(output, predictions)
                write_json(timing_path, {'elapsed_seconds': prior_seconds + time.monotonic() - started})
                write_json(directory / 'progress.json', {'pid': os.getpid(), 'phase': 'generate', 'images': len(predictions), 'total': 5000})
        del model, encoder
        torch.cuda.empty_cache()
    metrics_path = directory / 'metrics.json'
    try:
        subprocess.run([args.python, '-I', args.evaluator, *json.loads(args.evaluator_args),
                        '--predictions', str(output), '--output', str(metrics_path)], check=True)
    finally:
        write_json(timing_path, {'elapsed_seconds': prior_seconds + time.monotonic() - started})
    evaluated = json.loads(metrics_path.read_text())
    if evaluated.get('status') != 'complete':
        raise ValueError('Official evaluator did not complete')
    metrics = evaluated['metrics']
    if any(k not in metrics or not isinstance(metrics[k], (int, float)) or not math.isfinite(metrics[k]) for k in METRICS):
        raise ValueError('Evaluator must produce all seven finite official metrics')
    receipt = {'seed': subset['seed'], 'method': method, 'metrics': metrics,
               'eval_seconds': prior_seconds + time.monotonic() - started, 'test_images': len(predictions)}
    write_json(receipt_path, receipt)
    return receipt


def actual_runtime_gate(run):
    actual = 0.0
    for seed in SEEDS:
        for method in METHODS:
            directory = Path(run) / 'training' / str(seed) / method
            for name, key in (('training_complete.json', 'train_seconds'),
                              ('evaluation_complete.json', 'eval_seconds')):
                path = directory / name
                if path.exists():
                    actual += float(json.loads(path.read_text())[key])
    limit = 35 * 3600
    receipt = {'status': 'failed' if actual > limit else 'passed',
               'actual_seconds': actual, 'limit_seconds': limit}
    write_json(Path(run) / 'caption_runtime_gate.json', receipt)
    if actual > limit:
        raise RuntimeError('Actual completed training and evaluation exceeds 35 GPU hours')
    return receipt


def campaign(args):
    run = Path(args.run)
    subsets = prepare(run)['subsets']
    # Execute one isolated process per model, releasing GPU allocations on exit.
    def invoke(command, subset, method):
        actual_runtime_gate(run)
        argv = [args.python, '-I', str(Path(__file__).resolve()), command, '--run', str(run),
                '--seed', str(subset['seed']), '--method', method, '--gpt2', args.gpt2,
                '--clip', args.clip, '--checkpoint-every', str(args.checkpoint_every)]
        if args.evaluator:
            argv += ['--evaluator', args.evaluator, '--evaluator-args', args.evaluator_args]
        subprocess.run(argv, check=True)
        actual_runtime_gate(run)
    invoke('train', subsets[0], 'clean')
    invoke('generate', subsets[0], 'clean')
    first = run / 'training' / str(SEEDS[0]) / 'clean'
    training = json.loads((first / 'training_complete.json').read_text())
    evaluation = json.loads((first / 'evaluation_complete.json').read_text())
    decision = timing_decision(training['train_seconds'], evaluation['eval_seconds'])
    decision['clean_cider'] = evaluation['metrics']['CIDEr']
    decision['correctness_passed'] = decision['clean_cider'] >= .4
    write_json(run / 'caption_training_gate.json', decision)
    if not decision['passed'] or not decision['correctness_passed']:
        raise RuntimeError('Clean correctness or total projected time gate failed; dependent work stopped')
    for subset in subsets[:decision['subset_count']]:
        for method in METHODS:
            invoke('train', subset, method)
            invoke('generate', subset, method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'generate', 'campaign'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--gpt2')
    parser.add_argument('--clip')
    parser.add_argument('--evaluator', help='Python adapter for frozen official seven metrics')
    parser.add_argument('--evaluator-args', default='[]', help='JSON list of frozen evaluator resource arguments')
    parser.add_argument('--python', default=os.sys.executable, help='Isolated environment Python')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument('--method', choices=METHODS, default='clean')
    parser.add_argument('--checkpoint-every', type=int, default=100)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(Path(args.run))
        return
    gpu_lease()
    if not args.gpt2 or not args.clip or args.checkpoint_every < 1:
        parser.error('GPU commands require --gpt2, --clip and positive checkpoint interval')
    if args.command == 'campaign':
        if not args.evaluator:
            parser.error('Campaign requires --evaluator before starting Clean training')
        campaign(args)
    else:
        subset = next(s for s in prepare(Path(args.run))['subsets'] if s['seed'] == args.seed)
        (train if args.command == 'train' else generate)(args, subset, args.method)


if __name__ == '__main__':
    main()
