#!/usr/bin/env python3
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
"""A2: unchanged COMBINED_ACTIONS_RUN q scoring with frozen validation references."""
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import json

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_flag_rep_export import RUN, REPORT, COMBINED_ACTIONS_RUN, RES, load, rows, dump, writel
from scripts import run_combined_actions as combined_actions


def run():
    if os.environ.get('VL_DEDUP_GPU_LEASE') != '1':
        raise RuntimeError('GPU lock lease required')
    previous_path = RUN / 'scoring_progress.json'
    previous = load(previous_path) if previous_path.exists() else {}
    prior_seconds = previous.get('elapsed_seconds', 0.)
    begin = time.monotonic()
    progress = dict(status='preparing', completed=0, elapsed_seconds=prior_seconds,
                    prior_seconds=prior_seconds, nvidia_smi_before_load=subprocess.check_output(['nvidia-smi'], text=True))
    dump(RUN / 'scoring_progress.json', progress)
    parts, groups, clip = combined_actions.inputs()
    original = load(COMBINED_ACTIONS_RUN / 'record_scores.json')
    sampled = random.Random(20261106).sample(sorted(original), 200)
    singleton = sorted(ids[0] for ids in groups.values() if len(ids) == 1)
    expected = dict(train=9217, validation=1507, test=1174)
    actual = {p: sum(parts[r] == p for r in singleton) for p in expected}
    if actual != expected:
        raise RuntimeError(f'Singleton population gate: {actual} != {expected}')
    dataset = Path(combined_actions.cfg()['inputs']['dataset'])
    public = {r['record_id']: r for r in rows(dataset / 'public_records.jsonl')}
    lineage = {r['record_id']: r for r in rows(dataset / 'private/lineage.jsonl')}
    validation = sorted(r for r, v in original.items() if v['partition'] == 'validation')
    if len(validation) != 6132:
        raise RuntimeError('Frozen validation reference identity/count gate failed')
    reference_clip = [clip[r] for r in validation]
    reference_q = [original[r]['probability'] for r in validation if original[r]['valid']]
    writel(RUN / 'scoring_membership.jsonl', [dict(record_id=r, phase=phase, partition=parts[r])
           for phase, ids in [('rescore', sampled), ('singleton', singleton)] for r in ids])
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                      TRITON_CACHE_DIR=str(RUN / 'runtime_cache/triton'),
                      TORCHINDUCTOR_CACHE_DIR=str(RUN / 'runtime_cache/torchinductor'))
    sys.path.insert(0, os.environ.get('VL_DEDUP_VLLM_SITE', ''))
    from importlib.metadata import version
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    if version('vllm') != '0.11.0':
        raise RuntimeError('COMBINED_ACTIONS_RUN vLLM version gate failed')
    config = combined_actions.cfg()['inference']
    model_path = str(RES / 'models/Qwen2.5-VL-7B-Instruct')
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
                   min_pixels=config['min_pixels'], max_pixels=config['max_pixels'])
    model = LLM(model=model_path, dtype=config['dtype'], max_model_len=config['max_model_len'],
                max_num_seqs=config['max_num_seqs'], max_num_batched_tokens=config['max_num_batched_tokens'],
                enforce_eager=config['enforce_eager'], enable_chunked_prefill=config['enable_chunked_prefill'],
                limit_mm_per_prompt=config['limit_mm_per_prompt'],
                mm_processor_kwargs={'min_pixels': config['min_pixels'], 'max_pixels': config['max_pixels']},
                gpu_memory_utilization=config['gpu_memory_utilization'], disable_log_stats=True)
    sampling = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
    load_seconds = time.monotonic() - begin
    for phase, ids in [('rescore', sampled), ('singleton', singleton)]:
        output_path = RUN / f'{phase}_scores.jsonl'
        existing = rows(output_path) if output_path.exists() else []
        if [r['record_id'] for r in existing] != ids[:len(existing)]:
            raise RuntimeError('Scoring resume prefix identity failed')
        active = RUN / 'scoring_active_batch.json'
        if active.exists():
            raise RuntimeError('Unresolved admitted scoring batch; no automatic retry')
        phase_begin = time.monotonic()
        for offset in range(len(existing), len(ids), 16):
            block = ids[offset:offset + 16]
            inputs, pictures = [], []
            for rid in block:
                row = public[rid]
                path = row['images'][0]['image_path']
                picture = Image.open(path).convert('RGB')
                pictures.append(picture)
                text = combined_actions.cfg()['prompt'].format(caption=row['text'])
                messages = [{'role': 'user', 'content': [{'type': 'image', 'image': path}, {'type': 'text', 'text': text}]}]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs.append(dict(prompt=prompt, multi_modal_data={'image': [picture]}))
            dump(active, dict(phase=phase, record_ids=block))
            outputs = model.generate(inputs, sampling, use_tqdm=False)
            if len(outputs) != len(block):
                raise RuntimeError('Scoring output count mismatch')
            terminal = []
            for rid, output in zip(block, outputs, strict=True):
                first = output.outputs[0].logprobs[0]
                tokens = [dict(token_id=int(tid), text=processor.tokenizer.decode([tid]),
                               logprob=float(lp.logprob), rank=int(lp.rank))
                          for tid, lp in first.items() if lp.rank <= 20]
                prob = combined_actions.yes_probability(tokens)
                cp = float(mismatch_scores.percentile(reference_clip, [clip[rid]])[0])
                qp = float(mismatch_scores.percentile(reference_q, [mismatch_scores.effective_probability(prob['probability'])])[0]) if prob['valid'] else 1.
                terminal.append(dict(record_id=rid, partition=parts[rid], phase=phase, tokens=tokens,
                                     fusion=(cp + qp) / 2, **prob))
            with output_path.open('a') as stream:
                for row in terminal:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            active.unlink()
            for picture in pictures:
                picture.close()
            progress.update(status='running', phase=phase, completed=offset + len(block),
                            total=len(ids), elapsed_seconds=prior_seconds + time.monotonic() - begin, load_seconds=load_seconds)
            elapsed = time.monotonic() - phase_begin
            projection = progress['elapsed_seconds'] + elapsed / max(1, offset + len(block) - len(existing)) * (len(ids) - offset - len(block))
            if phase == 'rescore':
                projection += elapsed / max(1, offset + len(block) - len(existing)) * len(singleton)
            projection += 41 * 980
            progress['projected_all_gpu_hours'] = projection / 3600
            dump(RUN / 'scoring_progress.json', progress)
            if projection > 14 * 3600:
                raise RuntimeError(f'GPU time gate: projected {projection / 3600:.3f} >14 hours')
        if phase == 'rescore':
            rescored = rows(output_path)
            if any(r['valid'] != original[r['record_id']]['valid'] for r in rescored):
                raise RuntimeError('Rescore validity changed: preserve q outputs for diagnosis')
            comparisons = dict(n=200, seed=20261106,
                q_max_absolute_difference=max(abs(r['probability'] - original[r['record_id']]['probability']) for r in rescored if r['valid']),
                beta_max_absolute_difference=max(abs(r['fusion'] - original[r['record_id']]['fusion']) for r in rescored))
            dump(RUN / 'rescore_comparison.json', comparisons)
    scores = rows(RUN / 'singleton_scores.jsonl')
    result = dict(task='flag_rep', stage='A2',
                  rescore=load(RUN / 'rescore_comparison.json'), partitions={},
                  elapsed_seconds=prior_seconds + time.monotonic() - begin, benchmark_A1_unchanged=True)
    for part in expected:
        population = [r for r in scores if r['partition'] == part]
        flagged = [r for r in population if r['fusion'] < combined_actions.TF]
        result['partitions'][part] = dict(records=len(population), held=len(flagged),
                M_records=sum(lineage[r['record_id']]['original_level'] == 'M' for r in population),
                held_M_records=sum(lineage[r['record_id']]['original_level'] == 'M' for r in flagged))
    dump(REPORT / 'benchmark/results_a2.json', result)
    dump(RUN / 'scoring_complete.json', dict(status='complete', **result))
    progress.update(status='complete', elapsed_seconds=result['elapsed_seconds'])
    dump(RUN / 'scoring_progress.json', progress)


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        dump(RUN / 'stop.json', dict(status='stopped', stage='A2', error_type=type(exc).__name__, error=str(exc)))
        raise
