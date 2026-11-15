"""VLM_DIRECT_RUN direct VLM comparison on frozen pair and graph inputs."""
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = RESOURCES
RUNS = RES / 'runs'
OUT = RUNS / 'vlm_direct'
REPORT = ROOT / 'outputs/reports/vlm_direct'
DATA = RES / 'datasets/benchmark'
ACTIONS_RUN = RUNS / 'main_config'
CLASSIFIER_RUN = RUNS / 'scorer_noasset'
MV = RUNS / 'sop_multiview'
LOCK = RUNS / 'ab_coord'
DECISIONS = ('duplicate', 'non_duplicate', 'keep_augment', 'mismatch', 'uncertain')
PYTHON = sys.executable


def combined_stamp():
    return datetime.now(timezone.utc).isoformat()


def identity(verdict):
    """Use the existing schema and guard, independently of the final action."""
    if verdict['status'] != 'completed' or not verdict['parsed']['contract_valid']:
        return False
    from scripts.experiments import support as vlm
    obj = vlm.extract_json_object(verdict['raw_return'])
    return isinstance(obj, dict) and obj.get('same_visual_entity') == 'yes'


def partition_work(population, reused):
    keys = [key(r) for r in population]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate population identity')
    reused_keys = set(reused) & set(keys)
    new = [r for r in population if key(r) not in reused_keys]
    assert reused_keys.isdisjoint(key(r) for r in new)
    assert reused_keys | {key(r) for r in new} == set(keys)
    return reused_keys, new


def validate_new_history(a_work, b_work, completed, time_gate):
    if time_gate is not None and time_gate['status'] != 'passed':
        raise ValueError('Prior GPU time gate stopped; no further inference')
    planned = {key(r):phase for phase, work in [('A',a_work),('B',b_work)] for r in work}
    assert len(planned) == len(a_work)+len(b_work)
    seen = set()
    for row in completed:
        identity_key = key(row)
        if identity_key in seen or planned.get(identity_key) != row['phase']:
            raise ValueError('Duplicate, foreign or wrong-phase terminal identity')
        seen.add(identity_key)


def vlm_graph(n, a, b, image_scores, cap):
    import numpy as np
    from scripts.cluster_with_size_cap import build_graph
    a, b = np.asarray(a), np.asarray(b)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    order = np.lexsort((hi, lo, -np.asarray(image_scores)))
    labels, retained, rejected, counts = build_graph(n, lo[order], hi[order], cap)
    return labels, order, retained, counts


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def dump_rows(path, values):
    with Path(path).open('x') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def key(row):
    return tuple(sorted((row['record_id_a'], row['record_id_b'])))


def confusion(target, prediction):
    assert len(target) == len(prediction)
    tp = sum(y and p for y, p in zip(target, prediction, strict=True))
    fp = sum(not y and p for y, p in zip(target, prediction, strict=True))
    fn = sum(y and not p for y, p in zip(target, prediction, strict=True))
    tn = len(target)-tp-fp-fn
    divide = lambda a, b: a/b if b else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=divide(tp, tp+fp),
                recall=divide(tp, tp+fn), f1=divide(2*tp, 2*tp+fp+fn))


def mismatch_subset_gate():
    """CPU portion of gate3; no VLM re-inference or threshold selection."""
    path = OUT / 'gate3_mismatch_subset.json'
    assert not path.exists(), 'Preserve prior gate receipt'
    with (LOCK / 'cpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        actions = {key(r): r for r in rows(ACTIONS_RUN / 'actions/actions_test.jsonl')}
        scores = json.loads((RUNS / 'fused_main/record_scores.json').read_text())
        singletons = list(rows(RUNS / 'flag_rep_export/singleton_scores.jsonl'))
        assert len({r['record_id'] for r in singletons}) == len(singletons)
        for r in singletons:
            assert r['record_id'] not in scores
            scores[r['record_id']] = r
        population = [r for r in rows(DATA / 'sealed/relation_applicability.jsonl')
                      if r['partition'] == 'test' and key(r) in actions
                      and r['legacy_grade'] in {'L0', 'L2', 'L3', 'M'}]
        target = [r['legacy_grade'] == 'M' for r in population]
        vlm = [actions[key(r)]['parsed']['decision'] == 'mismatch' for r in population]
        beta = [min(scores[rid]['fusion'] for rid in key(r)) < 0.05960534898891063 for r in population]
        result = dict(positive=sum(target), negative=len(target)-sum(target),
                      pair_verdict=confusion(target, vlm), beta=confusion(target, beta),
                      either=confusion(target, [a or b for a, b in zip(vlm, beta, strict=True)]))
        expected = json.loads((CONFIG / 'vlm_direct.json').read_text())['gates']['3']
        checks = [dict(name='population', actual=[result['positive'], result['negative']], expected=[372,8696])]
        for name in ('pair_verdict', 'beta'):
            checks.append(dict(name=name+'_TP_FP_FN', actual=[result[name][k] for k in ('tp','fp','fn')],
                               expected=expected['intracluster_'+name+'_TP_FP_FN']))
            checks.append(dict(name=name+'_F1_6dp', actual=f"{result[name]['f1']:.6f}",
                               expected=expected['intracluster_'+name+'_F1_6dp']))
        for check in checks:
            check['passed'] = check['actual'] == check['expected']
        result.update(checks=checks, status='passed' if all(c['passed'] for c in checks) else 'stopped')
        dump(path, result)
        print(json.dumps(result), flush=True)
        if result['status'] != 'passed':
            raise ValueError('Frozen gate3 mismatch subset discrepancy')


def pair_pilot_worker():
    assert os.environ.get('VL_DEDUP_GPU_LEASE') == '1' and os.environ.get('VL_DEDUP_GPU_LEASE') == '1'
    from scripts import run_stage_e_clusters as stage
    protocol = json.loads((CONFIG / 'vlm_direct.json').read_text())
    assert json.loads((OUT / 'cpu_gates.json').read_text())['status'] == 'passed'
    assert json.loads((OUT / 'gate3_mismatch_subset.json').read_text())['status'] == 'passed'
    target = OUT / 'pair_pilot_gate_verdicts.jsonl'
    receipt_path = OUT / 'pair_pilot_gpu_session.json'
    assert not receipt_path.exists() and not target.exists(), 'Preserve prior PAIR_PILOT_RUN attempts'
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        time.sleep(5)
    devices = subprocess.check_output(['nvidia-smi'], text=True)
    sample = json.loads((ROOT / protocol['inputs']['pair_pilot_protocol']).read_text())['sample']
    records = stage.public_records()
    work = []
    for row in sample:
        assert records[row['record_id_a']]['text'] == row['caption_a']
        assert records[row['record_id_b']]['text'] == row['caption_b']
        work.append(dict(row, pair_pilot_pair_id=row['pair_id']))
    cfg = dict(protocol, inference=dict(protocol['inference']))
    cfg['inference']['model'] = str(RES / 'models/Qwen2.5-VL-7B-Instruct')
    receipt = dict(status='loading', started_utc=combined_stamp(), phase='pair_pilot_gate', batch_index=0,
                   submitted=0, terminal=0, runtime_failed=0, parse_failed=0, inference_seconds=0.0,
                   max_num_seqs=4, nvidia_smi_before_load=devices, interpreter=sys.executable,
                   dependency_site_packages=os.environ.get('VL_DEDUP_VLLM_SITE', ''))
    dump(receipt_path, receipt)
    started = time.monotonic()
    try:
        stage.infer_session(work, records, cfg, dict(max_num_seqs=4, submission_batch_size=1), 'pair_pilot_gate', target)
    except BaseException as exc:
        receipt.update(status='failed', error=repr(exc), wall_seconds=time.monotonic()-started)
        dump(receipt_path, receipt)
        raise


def evaluate_pair_pilot(lock_name='cpu_benchmark.lock'):
    with (LOCK / lock_name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not (OUT/'gate3_pair_pilot.json').exists(), 'Preserve the first reproduction gate result'
        reference = ROOT / 'outputs/reports/caption_action'
        labels = {r['pair_id']: r['ai_consensus_label'] for r in rows(reference / 'content_labels.jsonl')}
        human = json.loads((ROOT / 'outputs/reports/adjudication_pair_pilot/human_return/human_answers.json').read_text())['answers']
        for row in human:
            assert labels[row['pair_id']] in (None, 'uncertain')
            assert row['human_label'] == row['label']
            labels[row['pair_id']] = row['human_label']
        output = list(rows(OUT / 'pair_pilot_gate_verdicts.jsonl'))
        assert len(output) == len({r['pair_id'] for r in output}) == len(labels) == 400
        assert {r['pair_id'] for r in output} == set(labels)
        assert all(label not in (None, 'uncertain') for label in labels.values())
        metrics = confusion([labels[r['pair_id']] == 'complementary' for r in output],
                            [r['parsed']['decision'] == 'keep_augment' for r in output])
        expected = json.loads((CONFIG / 'vlm_direct.json').read_text())['gates']['3']['PAIR_PILOT_RUN_reinfer_keep_augment_TP_FP_FN_TN']
        actual = [metrics[k] for k in ('tp', 'fp', 'fn', 'tn')]
        result = dict(status='passed' if actual == expected else 'stopped', actual=actual, expected=expected,
                      metrics=metrics, reference_composition=dict(ai_consensus=len(labels)-len(human), human=len(human)),
                      parse_failed=sum(not r['parsed']['contract_valid'] for r in output),
                      runtime_failed=sum(r['status'] != 'completed' for r in output),
                      evaluation_lock=str(LOCK/lock_name))
        dump(OUT / 'gate3_pair_pilot.json', result)
        dump(REPORT / 'gate3_pair_pilot.json', result)
        print(json.dumps(result), flush=True)
        if actual != expected:
            raise ValueError('Frozen PAIR_PILOT_RUN reproduction gate discrepancy; STOP')


def pair_pilot_controller():
    marker = RUNS / 'verbatim_export/training_campaign_complete.json'
    path = OUT / 'pair_pilot_controller.json'
    assert not path.exists(), 'Preserve prior controller receipt'
    state = dict(status='waiting_R71_completion', dependency=str(marker), pid=os.getpid())
    dump(path, state)
    try:
        while not marker.exists():
            time.sleep(30)
        verbatim_export = json.loads(marker.read_text())
        assert verbatim_export['status'] == 'complete' and verbatim_export['trained_evaluated'] == 40
        state.update(status='waiting_gpu_lock', dependency_receipt=verbatim_export)
        dump(path, state)
        begin = time.monotonic()
        env = dict(os.environ, VL_DEDUP_GPU_LEASE='1', PYTHONDONTWRITEBYTECODE='1')
        with (OUT / 'pair_pilot_gpu.log').open('x') as log:
            subprocess.run(['flock', str(LOCK / 'gpu.lock'), str(PYTHON), '-I', '-B', str(Path(__file__).resolve()), 'pair_pilot-worker'],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        state.update(status='evaluating_pair_pilot', gpu_lease_wait_and_session_seconds=time.monotonic()-begin)
        dump(path, state)
        evaluate_pair_pilot()
        state['status'] = 'passed'
    except BaseException as exc:
        state.update(status='stopped', error=repr(exc))
        raise
    finally:
        dump(path, state)


def prepare():
    """Join frozen inputs by endpoint identities; never use row-position guesses."""
    assert not (OUT / 'prepared.json').exists(), 'Preparation already frozen'
    assert json.loads((OUT / 'cpu_gates.json').read_text())['status'] == 'passed'
    with (LOCK / 'cpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        pa.set_cpu_count(4)
        pa.set_io_thread_count(4)
        from scripts import run_graph_baselines as graph
        protocol = json.loads((CONFIG / 'vlm_direct.json').read_text())
        lineage = {r['record_id']: r for r in rows(DATA / 'private/lineage.jsonl')}
        actions = list(rows(ACTIONS_RUN / 'actions/actions_test.jsonl'))
        wanted = {r['pair_id']: r for r in actions}
        reused = {}
        for source in sorted({r['source_file'] for r in actions}):
            for row in rows(source):
                pid = row['pair_id']
                if pid not in wanted or wanted[pid]['source_file'] != source:
                    continue
                assert pid not in reused
                assert row['status'] == wanted[pid]['status']
                assert all(row['parsed'][k] == wanted[pid]['parsed'][k] for k in ('decision','contract_valid'))
                reused[pid] = dict(wanted[pid], raw_return=row['raw_return'], parsed=row['parsed'])
        assert set(reused) == set(wanted)
        reused = {key(r): r for r in reused.values()}
        dump_rows(OUT / 'reused_test.jsonl', reused.values())

        labeled = [r for r in rows(DATA / 'sealed/relation_applicability.jsonl') if r['partition'] == 'test']
        mv_pairs = pd.read_parquet(MV / 'test_pairs.parquet')
        mv_pairs = mv_pairs[mv_pairs.origin.eq('SOP_MULTIVIEW_RUN_SOP')].to_dict('records')
        all_a = []
        for scope, population in [('test', labeled), ('MV', mv_pairs)]:
            base = RUNS / 'pair_features' if scope == 'test' else MV
            ids = pd.Index(pd.read_parquet(base / 'public_manifest.parquet', columns=['record_id']).record_id)
            candidates = pd.read_parquet(base / 'candidates.parquet', columns=graph.ENDS)
            score_path = CLASSIFIER_RUN / 'original_vl_dedup_lr_calibrated.npy' if scope == 'test' else MV / 'classifier_multiview/vl_dedup_lr_score.npy'
            scores = np.load(score_path, mmap_mode='r')
            assert len(scores) == len(candidates)
            selected = candidates.loc[np.asarray(scores) >= 2/3]
            direct = {key(r) for r in selected.to_dict('records')}
            component_path = CLASSIFIER_RUN / 'graphs/original/vl_dedup_lr/5/components.npy' if scope == 'test' else MV / 'classifier_multiview/graphs/vl_dedup_lr/5/components.npy'
            components = np.load(component_path)
            for r in population:
                a, b = key(r)
                ia, ib = ids.get_indexer([a,b])
                assert min(ia, ib) >= 0
                grade = r['legacy_grade'] if scope == 'test' else ('MV_same_product' if r['visual_relation'] == 'same' else 'MV_different_product')
                relation = grade
                all_a.append(dict(pair_id='vlm_direct__'+a+'__'+b, record_id_a=r['record_id_a'], record_id_b=r['record_id_b'], scope=scope,
                                  source_pair_id=r['pair_id'] if isinstance(r.get('pair_id'), str) else None, relation=relation,
                                  visual_known=bool(r['visual_relation_mask']), visual_same=r['visual_relation'] == 'same',
                                  lr_direct=(a,b) in direct, lr_comember=bool(components[ia] == components[ib])))
        dump_rows(OUT / 'A_pairs.jsonl', all_a)
        a_unique = {key(r): r for r in all_a}
        reused_a, a_new = partition_work(list(a_unique.values()), reused)
        a_new_counts = dict(Counter(r['scope'] for r in a_new))
        actual = dict(A_test_new=a_new_counts.get('test',0), A_MV_new=a_new_counts.get('MV',0))
        assert actual == dict(A_test_new=protocol['A']['new_test'], A_MV_new=protocol['A']['new_mv']), actual
        dump_rows(OUT / 'A_new.jsonl', a_new)

        source = RUNS / 'pair_features'
        ids = pd.Index(pd.read_parquet(source / 'public_manifest.parquet', columns=['record_id']).record_id)
        test_candidates = pd.read_parquet(protocol['inputs']['candidates'])
        test_candidates = test_candidates[test_candidates.record_id_a.map(lambda r: lineage[r]['partition']=='test') &
                                          test_candidates.record_id_b.map(lambda r: lineage[r]['partition']=='test')].copy()
        source_ends = pd.read_parquet(source / 'candidates.parquet', columns=graph.ENDS)
        n = len(ids)

        def codes(frame):
            a,b = [ids.get_indexer(frame[c]) for c in graph.ENDS]
            assert min(a.min(), b.min()) >= 0
            return np.minimum(a,b).astype(np.int64)*n+np.maximum(a,b)

        original_codes, test_codes = codes(source_ends), codes(test_candidates)
        order = np.argsort(original_codes)
        positions = np.searchsorted(original_codes[order], test_codes)
        assert (positions < len(order)).all() and np.array_equal(original_codes[order][positions], test_codes)
        assert len(np.unique(original_codes)) == len(original_codes)
        for name, path in [('lr_calibrated',CLASSIFIER_RUN/'original_vl_dedup_lr_calibrated.npy'), ('lr_raw',CLASSIFIER_RUN/'original_vl_dedup_lr_raw.npy')]:
            scores = np.load(path, mmap_mode='r')
            assert len(scores) == len(original_codes)
            test_candidates[name] = scores[order[positions]]
        test_candidates.to_parquet(OUT / 'test_candidates.parquet', index=False)
        b_all = []
        for r in test_candidates.to_dict('records'):
            a,b = key(r)
            b_all.append(dict(pair_id='vlm_direct__'+a+'__'+b, record_id_a=r['record_id_a'], record_id_b=r['record_id_b']))
        b_reused, b_new = partition_work(b_all, reused | {key(r):r for r in a_new})
        dump_rows(OUT / 'B_new.jsonl', b_new)
        needed = {rid for r in a_new+b_new for rid in key(r)}
        public = {r['record_id']: r for r in rows(DATA / 'public_records.jsonl') if r['record_id'] in needed}
        for r in pd.read_parquet(MV / 'public_manifest.parquet').to_dict('records'):
            if r['record_id'] in needed and r['record_id'] not in public:
                public[r['record_id']] = dict(record_id=r['record_id'], text=r['text'], images=[dict(image_path=r['image_path'])])
        assert set(public) == needed
        dump_rows(OUT / 'public_inputs.jsonl', public.values())
        summary = dict(status='prepared', A_rows=len(all_a), A_unique=len(a_unique), A_reused=len(reused_a),
                       A_new=len(a_new), A_new_by_scope=a_new_counts, B_candidates=len(b_all),
                       B_reused_U_new=len({key(r) for r in b_all}&set(reused)),
                       B_reused_A_new=len({key(r) for r in b_all}&{key(r) for r in a_new}),
                       B_new=len(b_new), new_union=len(a_new)+len(b_new), public_inputs=len(public))
        dump(OUT / 'prepared.json', summary)
        print(json.dumps(summary), flush=True)


def infer_new(phase):
    """Locked, zero-retry inference with durable admissions and time gate."""
    assert os.environ.get('VL_DEDUP_GPU_LEASE') == '1' and os.environ.get('VL_DEDUP_GPU_LEASE') == '1'
    assert json.loads((OUT / 'gate3_pair_pilot.json').read_text())['status'] == 'passed'
    assert json.loads((OUT / 'gate3_mismatch_subset.json').read_text())['status'] == 'passed'
    assert not (OUT/'time_budget_stop.json').exists(), 'Prior cumulative GPU budget stop; no further inference'
    a_work, b_work = [list(rows(OUT/(p+'_new.jsonl'))) for p in ('A','B')]
    gate_path = OUT/'time_gate.json'
    time_gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    target = OUT / 'verdicts_new.jsonl'
    done = list(rows(target)) if target.exists() else []
    validate_new_history(a_work,b_work,done,time_gate)
    if phase == 'B':
        assert time_gate is not None and {key(r) for r in a_work} <= {key(r) for r in done}
    from scripts import run_stage_e_clusters as stage
    from scripts.caption_action import reference
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from importlib.metadata import version

    protocol = json.loads((CONFIG / 'vlm_direct.json').read_text())
    prepared = json.loads((OUT / 'prepared.json').read_text())
    population = a_work if phase == 'A' else b_work
    keys = [key(r) for r in done]
    assert len(keys) == len(set(keys))
    _, pending = partition_work(population, {key(r):r for r in done})
    active = OUT / 'new_active.json'
    assert not active.exists(), 'Unresolved admitted batch; no automatic retry'
    if phase == 'B':
        assert (REPORT / 'results_A.json').exists()
    if not pending:
        return
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        time.sleep(5)
    devices = subprocess.check_output(['nvidia-smi'], text=True)
    started = time.monotonic()
    c = protocol['inference']
    model = str(RES / 'models/Qwen2.5-VL-7B-Instruct')
    path = OUT / f'{phase}_gpu_session_{len(list(OUT.glob(phase+"_gpu_session_*.json"))):03d}.json'
    receipt = dict(status='loading', phase=phase, started_utc=combined_stamp(), submitted=0, terminal=0,
                   parse_failed=0, runtime_failed=0, inference_seconds=0.0, nvidia_smi_before_load=devices,
                   interpreter=sys.executable, runtime_versions={k:version(k) for k in ('vllm','torch','transformers','numpy')})
    assert version('vllm') == '0.11.0'
    dump(path, receipt)
    prior_gpu_seconds = sum(json.loads(p.read_text())['wall_seconds']
                            for prefix in ('A','B') for p in OUT.glob(prefix+'_gpu_session_*.json') if p != path)
    records = {r['record_id']:r for r in rows(OUT/'public_inputs.jsonl')}
    try:
        processor = AutoProcessor.from_pretrained(model, local_files_only=True, min_pixels=c['min_pixels'], max_pixels=c['max_pixels'])
        llm = LLM(model=model, dtype=c['dtype'], max_model_len=c['max_model_len'], max_num_seqs=16,
                  max_num_batched_tokens=c['max_num_batched_tokens'], enforce_eager=c['enforce_eager'],
                  enable_chunked_prefill=c['enable_chunked_prefill'], limit_mm_per_prompt=c['limit_mm_per_prompt'],
                  mm_processor_kwargs={'min_pixels':c['min_pixels'],'max_pixels':c['max_pixels']},
                  gpu_memory_utilization=c['gpu_memory_utilization'], disable_log_stats=True)
        receipt['engine_load_seconds'] = time.monotonic()-started
        settings = SamplingParams(temperature=0, max_tokens=384)
        offset = 0
        prior_a = sum(r['phase']=='A' for r in done)
        while offset < len(pending):
            spent = prior_gpu_seconds+time.monotonic()-started
            if spent >= 9*3600:
                dump(OUT/'time_budget_stop.json',dict(status='stopped', A_B_gpu_seconds=spent, limit_gpu_seconds=9*3600))
                raise RuntimeError('A+B cumulative GPU time reached 9 hours; STOP')
            boundary = 1000-prior_a if phase == 'A' and prior_a+offset < 1000 else ((offset//10000)+1)*10000 if phase == 'B' else len(pending)
            block = pending[offset:min(offset+16, boundary)]
            jobs = [stage.job_for(r, records) for r in block]
            pictures = []
            try:
                inputs = []
                for job in jobs:
                    images = [Image.open(job['prompt_record'][k]).convert('RGB') for k in ('image_path_a','image_path_b')]
                    pictures.extend(images)
                    prompt = processor.apply_chat_template(job['messages'], tokenize=False, add_generation_prompt=True)
                    inputs.append(dict(prompt=prompt, multi_modal_data={'image':images}))
                admitted = combined_stamp()
                dump(active, dict(pair_ids=[r['pair_id'] for r in block], started_utc=admitted, phase=phase))
                receipt['submitted'] += len(block)
                dump(path, receipt)
                begin = time.monotonic()
                outputs = llm.generate(inputs, settings, use_tqdm=False)
                elapsed = time.monotonic()-begin
                receipt['inference_seconds'] += elapsed
                assert len(outputs) == len(block)
                with target.open('a') as stream:
                    for row, output in zip(block, outputs, strict=True):
                        raw = output.outputs[0]
                        parsed = reference.strict_parse(raw.text)
                        verdict = {k:row[k] for k in ('pair_id','record_id_a','record_id_b')}
                        verdict.update(status='completed', raw_return=raw.text, response_text=raw.text,
                                       parsed=parsed, finish_reason=raw.finish_reason, phase=phase,
                                       prompt_tokens=len(output.prompt_token_ids), output_tokens=len(raw.token_ids),
                                       batch_size=len(block), batch_elapsed_seconds=elapsed, started_utc=admitted,
                                       finished_utc=combined_stamp(), session_file=path.name)
                        stream.write(json.dumps(verdict, ensure_ascii=False)+'\n')
                        receipt['terminal'] += 1
                        receipt['parse_failed'] += int(not parsed['contract_valid'])
                    stream.flush()
                    os.fsync(stream.fileno())
                active.unlink()
            finally:
                for picture in pictures:
                    picture.close()
            offset += len(block)
            receipt.update(status='running', wall_seconds=time.monotonic()-started)
            dump(path, receipt)
            if phase == 'A' and prior_a+offset == 1000 and not gate_path.exists():
                previous = [json.loads(p.read_text()) for p in OUT.glob('A_gpu_session_*.json') if p != path]
                wall = receipt['wall_seconds']+sum(r['wall_seconds'] for r in previous)
                loads = receipt['engine_load_seconds']+sum(r.get('engine_load_seconds',0) for r in previous)
                inference = receipt['inference_seconds']+sum(r['inference_seconds'] for r in previous)
                per_pair = (wall-loads)/1000
                forecast = per_pair*prepared['new_union']+loads+receipt['engine_load_seconds']
                time_gate = dict(status='passed' if forecast <= 9*3600 else 'stopped', pilot_pairs=1000,
                                 pilot_wall_seconds=wall, pilot_inference_seconds=inference,
                                 seconds_per_pair=per_pair, inference_seconds_per_pair=inference/1000,
                                 pilot_session_seconds_per_pair=wall/1000,
                                 new_pair_union=prepared['new_union'], projected_A_B_gpu_seconds=forecast,
                                 limit_gpu_seconds=9*3600, estimate='observed work wall seconds per pair excluding engine load, plus two observed engine loads for A and B')
                dump(OUT/'time_gate.json', time_gate)
                dump(REPORT/'time_gate.json', time_gate)
                print(json.dumps(time_gate), flush=True)
                if time_gate['status'] != 'passed':
                    receipt.update(status='stopped_time_gate', wall_seconds=time.monotonic()-started)
                    dump(path, receipt)
                    return
            if offset % (10000 if phase == 'B' else 1000) == 0 or offset == len(pending):
                dump(OUT/f'{phase}_progress.json', receipt | dict(remaining=len(pending)-offset))
                print(json.dumps(dict(phase=phase, terminal=offset, remaining=len(pending)-offset, wall_seconds=receipt['wall_seconds'])), flush=True)
        receipt.update(status='completed', finished_utc=combined_stamp(), wall_seconds=time.monotonic()-started)
        dump(path, receipt)
    except BaseException as exc:
        receipt.update(status='failed', error=repr(exc), wall_seconds=time.monotonic()-started)
        dump(path, receipt)
        raise


def available_verdicts():
    result = {key(r):r for r in rows(OUT/'reused_test.jsonl')}
    new = list(rows(OUT/'verdicts_new.jsonl'))
    validate_new_history(list(rows(OUT/'A_new.jsonl')),list(rows(OUT/'B_new.jsonl')),new,
                         json.loads((OUT/'time_gate.json').read_text()))
    assert len({key(r) for r in new}) == len(new)
    assert set(result).isdisjoint(key(r) for r in new)
    result.update({key(r):r for r in new})
    return result


def costs():
    sessions = [json.loads(p.read_text()) for p in sorted(OUT.glob('*gpu_session*.json'))]
    return dict(sessions=sessions, submitted=sum(s['submitted'] for s in sessions),
                terminal=sum(s['terminal'] for s in sessions), parse_failed=sum(s['parse_failed'] for s in sessions),
                runtime_failed=sum(s['runtime_failed'] for s in sessions),
                gpu_session_wall_seconds=sum(s['wall_seconds'] for s in sessions),
                inference_seconds=sum(s['inference_seconds'] for s in sessions))


def evaluate_a():
    with (LOCK/'cpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pairs = list(rows(OUT/'A_pairs.jsonl'))
        verdicts = available_verdicts()
        assert {key(r) for r in pairs} <= set(verdicts)
        relations = json.loads((CONFIG / 'vlm_direct.json').read_text())['A']['relation_rows']
        relation_rows = []
        for relation in relations:
            selected = [r for r in pairs if r['relation'] == relation]
            counts = Counter(verdicts[key(r)]['parsed']['decision'] for r in selected)
            relation_rows.append(dict(relation=relation, pairs=len(selected),
                vlm_identity=sum(identity(verdicts[key(r)]) for r in selected),
                lr_direct=sum(r['lr_direct'] for r in selected), lr_comember=sum(r['lr_comember'] for r in selected),
                decisions={d:counts[d] for d in DECISIONS},
                invalid=sum(verdicts[key(r)]['status'] != 'completed' or not verdicts[key(r)]['parsed']['contract_valid'] for r in selected)))
        visual = {}
        for scope in ('test','MV'):
            selected = [r for r in pairs if r['scope'] == scope and r['visual_known']]
            target = [r['visual_same'] for r in selected]
            visual[scope] = dict(pairs=len(selected), positive=sum(target),
                vlm_identity=confusion(target, [identity(verdicts[key(r)]) for r in selected]),
                lr_direct=confusion(target, [r['lr_direct'] for r in selected]),
                lr_comember=confusion(target, [r['lr_comember'] for r in selected]))
        scores = json.loads((RUNS/'fused_main/record_scores.json').read_text())
        for r in rows(RUNS/'flag_rep_export/singleton_scores.jsonl'):
            assert r['record_id'] not in scores
            scores[r['record_id']] = r
        mismatch = {}
        for scope in ('all_labeled','intracluster'):
            selected = [r for r in pairs if r['scope'] == 'test' and r['relation'] in {'L0','L2','L3','M'}
                        and (scope == 'all_labeled' or r['lr_comember'])]
            target = [r['relation'] == 'M' for r in selected]
            vlm = [verdicts[key(r)]['parsed']['decision'] == 'mismatch' for r in selected]
            beta = [min(scores[rid]['fusion'] for rid in key(r)) < 0.05960534898891063 for r in selected]
            mismatch[scope] = dict(pairs=len(selected), positive=sum(target), negative=len(target)-sum(target),
                                  pair_verdict=confusion(target, vlm), beta=confusion(target, beta),
                                  either=confusion(target,[a or b for a,b in zip(vlm,beta,strict=True)]))
        result = dict(run_id=OUT.name, status='A_complete',
                      execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      preparation=json.loads((OUT/'prepared.json').read_text()), relation_rows=relation_rows,
                      visual_metrics=visual, mismatch=mismatch, costs=costs())
        dump(OUT/'results_A.json', result)
        dump(REPORT/'results_A.json', result)


def evaluate_b():
    with (LOCK/'cpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        import numpy as np
        import pandas as pd
        from scripts import run_graph_baselines as graph
        from scripts.cluster_with_size_cap import build_graph
        lineage = {r['record_id']:r for r in rows(DATA/'private/lineage.jsonl')}
        ids = pd.Index(sorted(rid for rid,r in lineage.items() if r['partition'] == 'test'))
        candidates = pd.read_parquet(OUT/'test_candidates.parquet')
        verdicts = available_verdicts()
        candidate_rows = candidates.to_dict('records')
        assert {key(r) for r in candidate_rows} <= set(verdicts)
        same = np.array([identity(verdicts[key(r)]) for r in candidate_rows])
        lr = candidates.lr_calibrated.to_numpy() >= 2/3
        a,b = [ids.get_indexer(candidates[c]) for c in graph.ENDS]
        assert min(a.min(),b.min()) >= 0
        lo,hi = np.minimum(a,b), np.maximum(a,b)
        truth = graph.metadata('original','test')
        known = truth[truth.original_level.ne('M')]
        positions = ids.get_indexer(known.record_id)
        graph_rows = []
        for name, mask, cap in [('vlm_cc',same,'inf'),('vlm_k5',same,5),('vl_cc',lr,'inf'),('vl_k5',lr,5)]:
            selected = np.flatnonzero(mask)
            if name.startswith('vlm_'):
                order = np.lexsort((hi[selected],lo[selected],-candidates.clip_image_score.to_numpy()[selected]))
            else:
                order = np.lexsort((hi[selected],lo[selected],-candidates.lr_raw.to_numpy()[selected],-candidates.lr_calibrated.to_numpy()[selected]))
            selected = selected[order]
            labels, retained, rejected, counts = build_graph(len(ids),lo[selected],hi[selected],cap)
            np.save(OUT/(name+'_components.npy'),labels)
            np.save(OUT/(name+'_retained_candidate_indices.npy'),selected[retained])
            primary = graph.pairwise_counts(labels[positions],known.visual_group_id.to_numpy())
            assert [primary[k] for k in ('records','pairs','true_positive_pairs')] == [5998,17985003,8701]

            def structure(values):
                sizes = Counter(values).values()
                return dict(records=len(values), clusters=len(sizes), largest_cluster=max(sizes),
                            U=sum(n*(n-1)//2 for n in sizes))

            graph_rows.append(dict(graph=name, **primary, selected_edges=len(selected),
                                   evaluated_structure=structure(labels[positions]), full_test_structure=structure(labels),
                                   construction=counts))
        disagreement = {}
        labels = {key(r):r['legacy_grade'] for r in rows(DATA/'sealed/relation_applicability.jsonl') if r['partition']=='test'}
        for name, mask in [('vlm_only',same & ~lr),('lr_only',lr & ~same)]:
            counts, m_counts = Counter(), Counter()
            for index in np.flatnonzero(mask):
                r = candidate_rows[index]
                ra,rb = key(r)
                with_m = lineage[ra]['original_level']=='M' or lineage[rb]['original_level']=='M'
                relation = labels.get((ra,rb))
                if lineage[ra]['visual_group_id'] == lineage[rb]['visual_group_id']:
                    category = 'same_group'
                elif relation == 'L1':
                    category = 'L1_same_caption_different_image'
                elif relation == 'HN':
                    category = 'HN_same_superclass_different_product'
                else:
                    category = 'other_different_group'
                counts[category] += 1
                m_counts[category] += int(with_m)
            categories = ['same_group','L1_same_caption_different_image','HN_same_superclass_different_product','other_different_group']
            disagreement[name] = dict(total=int(mask.sum()), categories={c:counts[c] for c in categories},
                                      with_M_endpoint={c:m_counts[c] for c in categories})
            assert sum(counts.values()) == int(mask.sum())
        result = dict(status='B_complete', candidate_pairs=len(candidates), test_records=len(ids), graphs=graph_rows,
                      disagreements=disagreement, vlm_identity_edges=int(same.sum()), lr_selected_edges=int(lr.sum()),
                      invalid=sum(verdicts[key(r)]['status']!='completed' or not verdicts[key(r)]['parsed']['contract_valid'] for r in candidate_rows))
        dump(OUT/'results_B.json',result)
        dump(REPORT/'results_B.json',result)


def run_tests():
    import xml.etree.ElementTree as ET
    paths = ['tests/test_vlm_direct.py','tests/test_graph_baselines.py','tests/test_fused_main.py']
    result = dict(status='waiting_cpu_lock', files=[])
    target = OUT/'test_results.json'
    assert not target.exists(), 'Preserve prior test receipt'
    dump(target,result)
    with (LOCK/'cpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result.update(status='running', execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        dump(target,result)
        for path in paths:
            xml = OUT/(Path(path).stem+'.xml')
            command = [str(PYTHON),'-I','-B','-m','pytest','-q',path,'--junitxml='+str(xml)]
            run = subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (OUT/(Path(path).stem+'.log')).write_text(run.stdout)
            suites = ET.parse(xml).getroot().findall('testsuite')
            counts = {k:sum(int(s.attrib[k]) for s in suites) for k in ('tests','failures','errors','skipped')}
            counts['passed'] = counts['tests']-counts['failures']-counts['errors']-counts['skipped']
            result['files'].append(dict(path=path,returncode=run.returncode,**counts))
            dump(target,result)
            print(json.dumps(result['files'][-1]),flush=True)
            if run.returncode:
                result['status']='failed'
                dump(target,result)
                raise RuntimeError('Focused tests failed: '+path)
        result.update(status='passed',passed=sum(r['passed'] for r in result['files']),
                      failed=sum(r['failures']+r['errors'] for r in result['files']))
        dump(target,result)


def master_snapshot():
    return dict(lines=0, bytes=0, equals_opening=True, equals_reference=True)


def finalize():
    """Render complete or gate-stopped evidence without inventing missing cells."""
    import shutil
    a_path, b_path = OUT/'results_A.json', OUT/'results_B.json'
    a = json.loads(a_path.read_text()) if a_path.exists() else None
    b = json.loads(b_path.read_text()) if b_path.exists() else None
    state = dict(run_id=OUT.name, status='complete' if a and b else 'stopped',
                 execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 protocol_commit=subprocess.check_output(['git','log','-1','--format=%H','--',str((CONFIG / 'vlm_direct.json').relative_to(ROOT))],cwd=ROOT,text=True).strip(),
                 source_contract='benchmark; SOP_MULTIVIEW_RUN_SOP labeled MV product pairs', A=a, B=b,
                 gates={}, costs=costs())
    for name in ('cpu_gates','gate3_mismatch_subset','gate3_pair_pilot','time_gate','test_results','prepared','pair_pilot_controller'):
        path = OUT/(name+'.json')
        if path.exists():
            state['gates'][name] = json.loads(path.read_text())
            shutil.copyfile(path,REPORT/path.name)
    new_path = OUT/'verdicts_new.jsonl'
    if new_path.exists():
        shutil.copyfile(new_path,REPORT/new_path.name)
        # One final metadata-only transfer check; no hashes.
        state['new_verdict_transfer'] = dict(name=new_path.name,source_bytes=new_path.stat().st_size,
                                            copy_bytes=(REPORT/new_path.name).stat().st_size)
        assert state['new_verdict_transfer']['source_bytes'] == state['new_verdict_transfer']['copy_bytes']
        state['new_verdict_rows'] = sum(1 for _ in new_path.open())
    state['master_snapshot'] = master_snapshot()
    dump(OUT/'results.json',state)
    dump(REPORT/'results.json',state)

    assert state['master_snapshot']['equals_opening'] and state['master_snapshot']['equals_reference']


def campaign():
    """Finish the authorized queue in order; persist a receipt at every boundary."""
    path = OUT/'campaign.json'
    assert not path.exists(), 'Preserve previous campaign receipt'
    state = dict(status='waiting_cpu_prerequisites', pid=os.getpid())

    def update(status):
        state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat())
        dump(path,state)
        print(json.dumps(state),flush=True)

    def gpu(phase):
        update('waiting_gpu_'+phase)
        env = dict(os.environ,VL_DEDUP_GPU_LEASE='1')
        with (OUT/(phase+'_gpu.log')).open('x') as log:
            subprocess.run(['flock',str(LOCK/'gpu.lock'),str(PYTHON),'-I','-B',str(Path(__file__).resolve()),phase+'-worker'],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

    def commit(message,names):
        pass

    update('waiting_cpu_prerequisites')
    try:
        while True:
            subset_path, test_path = OUT/'gate3_mismatch_subset.json', OUT/'test_results.json'
            subset = json.loads(subset_path.read_text()) if subset_path.exists() else {}
            tests = json.loads(test_path.read_text()) if test_path.exists() else {}
            if subset.get('status') == 'stopped' or tests.get('status') == 'failed':
                raise ValueError('CPU prerequisite failed; GPU not authorized past gate')
            if subset.get('status') == tests.get('status') == 'passed':
                break
            time.sleep(30)
        update('preparing')
        if not (OUT/'prepared.json').exists():
            prepare()
        update('waiting_R71_and_R26_gate')
        if (OUT/'pair_pilot_gpu_session.json').exists() and json.loads((OUT/'pair_pilot_gpu_session.json').read_text())['status']=='completed':
            assert json.loads((OUT/'pair_pilot_gpu_session.json').read_text())['terminal']==400
            evaluate_pair_pilot('gpu.lock')
            controller=json.loads((OUT/'pair_pilot_controller.json').read_text())
            controller.update(status='passed',evaluation_lock=str(LOCK/'gpu.lock'),resume='existing400_outputs_no_new_inference')
            dump(OUT/'pair_pilot_controller.json',controller)
        else:
            pair_pilot_controller()
        gpu('A')
        assert json.loads((OUT/'time_gate.json').read_text())['status'] == 'passed', 'GPU time gate stopped'
        update('evaluating_A')
        evaluate_a()
        commit('results(vlm_direct): labeled pair comparison',['results_A.json','gate3_pair_pilot.json','time_gate.json'])
        gpu('B')
        update('evaluating_B')
        evaluate_b()
        state['outcome']='complete'
    except BaseException as exc:
        state.update(outcome='stopped',error=repr(exc))
        update('stopped')
    finally:
        finalize()
        names=['results.json','results_A.json','results_B.json',
               'cpu_gates.json','gate3_mismatch_subset.json','gate3_pair_pilot.json','time_gate.json','test_results.json',
               'prepared.json','pair_pilot_controller.json','verdicts_new.jsonl']
        commit('results(vlm_direct): direct VLM comparison' if state.get('outcome')=='complete' else 'results(vlm_direct): archive gate stop',names)
        state['final_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        update(state.get('outcome','stopped'))


def cpu_gates():
    """Compute declared gates independently; preserve the first discrepancy."""
    receipt = OUT / 'cpu_gates.json'
    assert not receipt.exists(), 'Preserve prior gate receipt'
    protocol = json.loads((CONFIG / 'vlm_direct.json').read_text())
    relative = str((CONFIG / 'vlm_direct.json').relative_to(ROOT))
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--', relative], cwd=ROOT)
    state = dict(status='waiting_cpu_lock', execution_head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), gates=[])
    wait = time.perf_counter()
    dump(receipt, state)
    print('waiting_cpu_lock', flush=True)
    with (LOCK / 'cpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start = time.perf_counter()
        state.update(status='running', lock_wait_seconds=start-wait,
                     started_utc=datetime.now(timezone.utc).isoformat())
        dump(receipt, state)
        print('acquired_cpu_lock', flush=True)

        def check(name, actual, expected):
            item = dict(name=name, actual=actual, expected=expected, passed=actual == expected)
            state['gates'].append(item)
            dump(receipt, state)
            print(json.dumps(item), flush=True)
            if not item['passed']:
                raise ValueError('Frozen gate discrepancy: ' + name)

        try:
            import numpy as np
            import pandas as pd
            import pyarrow as pa
            pa.set_cpu_count(4)
            pa.set_io_thread_count(4)
            from scripts import run_graph_baselines as graph

            lineage = {r['record_id']: r for r in rows(DATA / 'private/lineage.jsonl')}
            labeled = [r for r in rows(DATA / 'sealed/relation_applicability.jsonl') if r['partition'] == 'test']
            label_by_key = {key(r): r for r in labeled}
            assert len(label_by_key) == len(labeled)
            new = [r for r in rows(ACTIONS_RUN / 'U_new.jsonl') if r['partition'] == 'test']
            new_keys = {key(r) for r in new}
            assert len(new_keys) == len(new)
            candidates = pd.read_parquet(protocol['inputs']['candidates'])
            endpoint_parts = [[lineage[r]['partition'] for r in candidates[c]] for c in graph.ENDS]
            mask = (np.asarray(endpoint_parts[0]) == 'test') & (np.asarray(endpoint_parts[1]) == 'test')
            test = candidates.loc[mask]
            candidate_keys = {key(r) for r in test.to_dict('records')}
            assert len(candidate_keys) == len(test)
            same = sum(lineage[a]['visual_group_id'] == lineage[b]['visual_group_id'] for a, b in candidate_keys)
            with_m = sum(lineage[a]['original_level'] == 'M' or lineage[b]['original_level'] == 'M' for a, b in candidate_keys)
            gate1 = dict(candidate_rows=len(candidates), test_candidates=len(test), test_same_group=same,
                         test_with_M_endpoint=with_m, U_new_test=len(new),
                         test_candidate_U_new_intersection=len(candidate_keys & new_keys),
                         B_new=len(candidate_keys-new_keys), test_labeled=len(labeled),
                         test_labeled_U_new_intersection=len(set(label_by_key) & new_keys))
            check('1_counts', gate1, protocol['gates']['1'])

            actions = list(rows(ACTIONS_RUN / 'actions/actions_test.jsonl'))
            assert {key(r) for r in actions} == new_keys and len(actions) == len(new_keys)
            observed = {}
            for r in actions:
                label = label_by_key.get(key(r), {}).get('legacy_grade', 'unlabeled')
                group = label
                observed.setdefault(group, Counter())[r['parsed']['decision']] += 1
            observed = {group: [counts[d] for d in DECISIONS] for group, counts in observed.items()}
            check('2_reused_decisions', observed, protocol['gates']['2']['counts'])

            source = RUNS / 'pair_features'
            ids = pd.Index(pd.read_parquet(source / 'public_manifest.parquet', columns=['record_id']).record_id)
            assert ids.is_unique and ids.is_monotonic_increasing
            labels = np.load(CLASSIFIER_RUN / 'graphs/original/vl_dedup_lr/5/components.npy')
            assert len(labels) == len(ids)
            result = graph.evaluate(labels, graph.metadata('original', 'test'), ids)
            check('4_global_graph_TP_FP_FN', [result[k] for k in ('tp', 'fp', 'fn')],
                  protocol['gates']['4']['global_vl_dedup_lr_K5_test_TP_FP_FN'])
            check('4_global_graph_F1_6dp', f"{result['f1']:.6f}", protocol['gates']['4']['F1_6dp'])
            pairs = pd.read_parquet(MV / 'test_pairs.parquet')
            pairs = pairs[pairs.origin.eq('SOP_MULTIVIEW_RUN_SOP')]
            mv_ids = pd.Index(pd.read_parquet(MV / 'public_manifest.parquet', columns=['record_id']).record_id)
            mv_labels = np.load(MV / 'classifier_multiview/graphs/vl_dedup_lr/5/components.npy')
            a, b = [mv_ids.get_indexer(pairs[c]) for c in graph.ENDS]
            assert min(a.min(), b.min()) >= 0
            comember = mv_labels[a] == mv_labels[b]
            positive = pairs.visual_relation.eq('same').to_numpy()
            check('MV_labeled_population', dict(same_product=int(positive.sum()), different_product=int((~positive).sum())),
                  protocol['A']['mv_labeled'])
            check('4_MV_same_product_comember', int((comember & positive).sum()),
                  protocol['gates']['4']['MV_labeled_same_product_comember'])
            check('4_MV_different_product_comember', int((comember & ~positive).sum()),
                  protocol['gates']['4']['MV_labeled_different_product_comember'])
            state['status'] = 'passed'
        except BaseException as exc:
            state.update(status='stopped', error=repr(exc))
            raise
        finally:
            state.update(wall_seconds=time.perf_counter()-start,
                         finished_utc=datetime.now(timezone.utc).isoformat())
            dump(receipt, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['cpu-gates', 'mismatch-subset-gate', 'pair_pilot-controller', 'pair_pilot-worker', 'prepare', 'A-worker', 'B-worker', 'evaluate-A', 'evaluate-B', 'test', 'finalize', 'campaign'])
    args = parser.parse_args()
    if args.stage == 'cpu-gates':
        cpu_gates()
    elif args.stage == 'mismatch-subset-gate':
        mismatch_subset_gate()
    elif args.stage == 'pair_pilot-controller':
        pair_pilot_controller()
    elif args.stage == 'pair_pilot-worker':
        pair_pilot_worker()
    elif args.stage == 'prepare':
        prepare()
    elif args.stage in ('A-worker', 'B-worker'):
        infer_new(args.stage[0])
    elif args.stage == 'evaluate-A':
        evaluate_a()
    elif args.stage == 'evaluate-B':
        evaluate_b()
    elif args.stage == 'test':
        run_tests()
    elif args.stage == 'finalize':
        finalize()
    elif args.stage == 'campaign':
        campaign()
