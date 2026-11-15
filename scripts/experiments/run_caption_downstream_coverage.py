"""CAPTION_DOWNSTREAM_RUN application of the unchanged COVERAGE_EXPORT_RUN coverage prompt, cache and export order."""
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_caption_downstream import RUN, dump, load, rows, writel
from scripts.experiments import run_coverage_export as coverage_export
from scripts.experiments import run_coverage_check as coverage_check


def main():
    assert os.environ.get('VL_DEDUP_GPU_LEASE') == '1'
    assert load(RUN / 'b_export_complete.json')['status'] == 'complete'
    if (RUN / 'coverage_complete.json').exists():
        return
    if (RUN / 'coverage_active_batch.json').exists():
        raise RuntimeError('Unresolved coverage batch: preserve outputs and reconcile without retry')
    begin = time.monotonic()
    protocol = load(ROOT / 'outputs/reports/coverage_export/protocol.json')
    config = protocol['model']
    records = {r['record_id']: r for r in rows(RUN / 'records.jsonl')}
    jobs = rows(RUN / 'b_exports.jsonl')
    cache = {}
    historical = Path(protocol['inputs']['coverage_check'])
    inputs = {(r['group'], r['unit_id']): r for r in rows(historical / 'inputs.jsonl')}
    for prediction in rows(historical / 'predictions.jsonl'):
        row = inputs[prediction['group'], prediction['unit_id']]
        key = json.dumps(dict(deleted_caption=row['deleted_caption'],
                             retained_captions=[r['caption'] for r in row['retained']]), ensure_ascii=False)
        cache[key] = dict(prediction, origin='coverage_check')
    output = RUN / 'coverage_predictions.jsonl'
    existing = rows(output) if output.exists() else []
    assert len({r['text_key'] for r in existing}) == len(existing)
    for prediction in existing:
        cache[prediction['text_key']] = prediction
    sys.path.insert(0, config['dependency_site'])
    os.environ.update(PYTHONPATH=config['dependency_site'], VLLM_WORKER_MULTIPROC_METHOD='spawn',
                      TRITON_CACHE_DIR=str(RUN / 'runtime_cache/triton'),
                      TORCHINDUCTOR_CACHE_DIR=str(RUN / 'runtime_cache/torchinductor'))
    # Keep vLLM UNIX socket paths short without touching the model resources.
    os.chdir(RUN)
    os.environ['VLLM_RPC_BASE_PATH'] = '.'
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    snapshot = subprocess.check_output(['nvidia-smi'], text=True)
    previous_path = RUN / 'coverage_progress.json'
    previous = load(previous_path) if previous_path.exists() else {}
    prior_seconds = previous.get('elapsed_seconds', previous.get('wall_seconds', 0.))
    dump(previous_path, dict(status='loading', gpu=snapshot, elapsed_seconds=prior_seconds))
    tokenizer = AutoTokenizer.from_pretrained(config['path'], local_files_only=True)
    kwargs = {k: v for k, v in config.items() if k not in ['name', 'path', 'backend', 'dependency_site']}
    model = LLM(model=config['path'], **kwargs)
    sampling = SamplingParams(temperature=0, top_p=1, max_tokens=384, n=1)
    checks = []
    session = dict(status='running', prior_terminal_requests=len(existing),
                   new_requests=0, invalid=sum(not r['valid'] for r in existing),
                   valid_null=sum(r['valid'] and r['covered'] is None for r in existing),
                   prior_elapsed_seconds=prior_seconds,
                   load_seconds=previous.get('load_seconds', 0.)+time.monotonic()-begin)

    def evaluate(requests):
        missing = {}
        for phase, rid, retained in requests:
            key = coverage_export.text_key(rid, retained, records)
            if key not in cache:
                missing.setdefault(key, (phase, rid, retained))
        items = list(missing.items())
        for offset in range(0, len(items), 16):
            block = items[offset:offset+16]
            rendered = []
            for key, _ in block:
                prompt = tokenizer.apply_chat_template([dict(role='user', content=protocol['prompt_prefix']+key)],
                                                      tokenize=False, add_generation_prompt=True)
                assert len(tokenizer.encode(prompt, add_special_tokens=False)) + 384 <= config['max_model_len']
                rendered.append(prompt)
            dump(RUN / 'coverage_active_batch.json', dict(text_keys=[k for k, _ in block]))
            outputs = model.generate(rendered, sampling, use_tqdm=False)
            assert len(outputs) == len(block)
            terminal = []
            for (key, (phase, rid, retained)), result in zip(block, outputs, strict=True):
                value = result.outputs[0]
                parsed = coverage_check.parse_output(value.text, value.finish_reason)
                terminal.append(dict(text_key=key, phase=phase, source_id=rid, retained_ids=retained,
                     raw=value.text, finish_reason=value.finish_reason, output_tokens=len(value.token_ids),
                     origin='caption_downstream', **parsed))
            with output.open('a') as stream:
                for row in terminal:
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            for row in terminal:
                cache[row['text_key']] = row
            session['new_requests'] += len(terminal)
            session['invalid'] += sum(not r['valid'] for r in terminal)
            session['valid_null'] += sum(r['valid'] and r['covered'] is None for r in terminal)
            session['wall_seconds'] = time.monotonic()-begin
            session['elapsed_seconds'] = prior_seconds+session['wall_seconds']
            dump(RUN / 'coverage_progress.json', session)
            (RUN / 'coverage_active_batch.json').unlink()
        answers = []
        for phase, rid, retained in requests:
            prediction = cache[coverage_export.text_key(rid, retained, records)]
            answer = prediction['covered'] if prediction['valid'] else None
            checks.append(dict(phase=phase, source_id=rid, retained_ids=retained,
                               covered=answer, valid=prediction['valid'], origin=prediction['origin']))
            answers.append(answer)
        return answers

    orders = {j['component']: sorted(j['b']['deleted'], key=lambda r: (-records[r]['clipscore'], r)) for j in jobs}
    requests = [('b_cov_one_pass', rid, sorted(j['b']['retained'])) for j in jobs for rid in orders[j['component']]]
    evaluate(requests)
    states = {j['component']: dict(retained=list(j['b']['retained']), pending=list(j['b']['pending']), deleted=[]) for j in jobs}
    for rank in range(max((len(order) for order in orders.values()), default=0)):
        active = [j for j in jobs if rank < len(orders[j['component']])]
        requests = [('b_cov', orders[j['component']][rank], sorted(states[j['component']]['retained'])) for j in active]
        for job, request, answer in zip(active, requests, evaluate(requests), strict=True):
            states[job['component']]['deleted' if answer is True else 'retained'].append(request[1])
    exports = []
    for job in jobs:
        def checked(rid, retained):
            value = cache[coverage_export.text_key(rid, retained, records)]
            return value['covered'] if value['valid'] else None
        result = coverage_export.coverage_export(job['b'], {r: records[r]['clipscore'] for r in job['record_ids']}, checked)
        assert result == {s: sorted(states[job['component']][s]) for s in coverage_export.STATES}
        assert result['pending'] == sorted(job['b']['pending'])
        assert set(job['b']['retained']) <= set(result['retained'])
        assert set(result['deleted']) <= set(job['b']['deleted'])
        exports.append(dict(job, b_cov=result))
    writel(RUN / 'coverage_exports.jsonl', exports)
    writel(RUN / 'coverage_checks.jsonl', checks)
    session.update(status='complete', logical_checks=len(checks),
                   total_terminal_requests=len(existing)+session['new_requests'], wall_seconds=time.monotonic()-begin)
    session['elapsed_seconds'] = prior_seconds+session['wall_seconds']
    dump(RUN / 'coverage_progress.json', session)
    dump(RUN / 'coverage_complete.json', session)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        dump(RUN / 'coverage_failure.json', dict(status='failed', error=type(exc).__name__, detail=str(exc)))
        raise
