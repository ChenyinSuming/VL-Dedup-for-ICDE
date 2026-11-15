"""COVERAGE_EXPORT_RUN frozen coverage-checked exports and retained-set audits."""
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
from scripts.experiments.support import export_cluster
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import mimetypes
import os
from pathlib import Path
import random
from fractions import Fraction
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import retained_set_helpers as base
from scripts import coverage_audit
from scripts.ai_audit import AUDITOR_MODELS
from scripts.retained_set_helpers import LABEL_MAP
MODELS = [AUDITOR_MODELS[1], AUDITOR_MODELS[2]]
from scripts.experiments import run_coverage_check as coverage_check
from scripts import run_combined_actions as combined_actions

REPORT = ROOT / 'outputs/reports/coverage_export'
RUN = (RESOURCES / 'runs/coverage_export')
STATES = ('retained', 'pending', 'deleted')
ARMS = ('keep_one', 'b', 'b_cov')
read, rows, dump = coverage_check.read, coverage_check.rows, coverage_check.dump


def write_rows(path, values):
    Path(path).write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values))


def protocol():
    return read(CONFIG / 'coverage_export.json')


def coverage_export(original, scores, check, iterative=True):
    """check(source_id, complete retained_ids) returns bool/null; only true deletes."""
    result = {s: list(original[s]) for s in STATES}
    result['deleted'] = []
    for rid in sorted(original['deleted'], key=lambda r: (-scores[r], r)):
        reference = result['retained'] if iterative else original['retained']
        if check(rid, sorted(reference)) is True:
            result['deleted'].append(rid)
        else:
            result['retained'].append(rid)
    return {s: sorted(result[s]) for s in STATES}


def counts(exports):
    return {s: sum(len(e[s]) for e in exports) for s in STATES}


def identity(source, retained):
    return source, tuple(sorted(retained))


def prepare():
    p = protocol()
    dump(RUN / 'protocol.json', p)
    inputs = p['inputs']
    parts, groups, clip = combined_actions.inputs()
    fusion = {k: v['fusion'] for k, v in read(Path(inputs['combined_actions']) / 'record_scores.json').items()}
    corpus, gate = [], {'passed': False, 'checks': {}}
    dump(RUN / 'gates.json', gate)
    for part, expected in p['gates']['combined_actions'].items():
        decisions = defaultdict(dict)
        for row in rows(combined_actions.REFERENCE / f'actions_{part}.jsonl'):
            decisions[row['component']][combined_actions.key(row)] = row['parsed']['decision']
        archived = {e['component']: e for e in rows(Path(inputs['combined_actions']) / f'exports_{part}.jsonl')}
        for component, ids in groups.items():
            if parts[ids[0]] != part:
                continue
            rep = min(ids, key=lambda r: (-clip[r], r))
            actions = {pair: combined_actions.pair_action(pair, fusion, value)
                       for pair, value in decisions[component].items()}
            arm = export_cluster(ids, rep, actions)['b']
            assert all(arm[s] == archived[component][s] for s in STATES)
            corpus.append(dict(archived[component], **{'b': arm}))
        got = counts([e['b'] for e in corpus if e['partition'] == part])
        gate['checks']['combined_actions_' + part] = got
        dump(RUN / 'gates.json', gate)
        assert list(got.values()) == expected, (part, got)
    b34 = ROOT / inputs['retained_audit_report']
    old = rows(b34 / 'exports.jsonl')
    by_ids = {tuple(sorted(e['record_ids'])): e for e in corpus}
    for c in old:
        assert by_ids[tuple(sorted(c['record_ids']))]['b'] == c['strategies']['b']
    old_counts = {a: counts([c['strategies'][a] for c in old]) for a in ('a', 'b')}
    audits = [{a['cell_id']: a for a in rows(b34 / f'audit_{i}.jsonl')} for i in (1, 2)]
    lost = {}
    for arm in ('a', 'b'):
        votes = [[index[c['cell_id']]['audit']['label'] == 'lost' for index in audits]
                 for c in rows(b34 / 'audit_cells.jsonl') if c['strategy'] == arm]
        lost[arm] = [sum(all(v) for v in votes), sum(any(v) for v in votes)]
    gate['checks']['retained_audit'] = dict(counts=old_counts, lost=lost)
    dump(RUN / 'gates.json', gate)
    assert list(old_counts['a'].values()) == p['gates']['retained_audit']['keep_one']
    assert list(old_counts['b'].values()) == p['gates']['retained_audit']['b']
    assert lost['a'] == p['gates']['retained_audit']['lost_keep_one'] and lost['b'] == p['gates']['retained_audit']['lost_b']
    natural = Path(inputs['noisy_pool'])
    nr = {r['record_id']: r for r in rows(natural / 'a_records.jsonl')}
    ne = rows(natural / 'a_exports.jsonl')
    nf = read(natural / 'a_fusion.json')
    pairs, verdicts = rows(natural / 'a_pairs.jsonl'), rows(natural / 'a_pair_verdicts.jsonl')
    assert len(pairs) == len(verdicts)
    decisions = {}
    for pair, verdict in zip(pairs, verdicts, strict=True):
        assert pair['pair_id'] == verdict['pair_id']
        decisions[combined_actions.key(pair)] = verdict['parsed']['decision']
    ngroups = defaultdict(list)
    for rid, record in nr.items():
        ngroups[record['component']].append(rid)
    for e in ne:
        ids = sorted(ngroups[e['component']])
        actions = {pair: combined_actions.pair_action(pair, nf, decisions[pair]) for pair in itertools.combinations(ids, 2)}
        rebuilt = export_cluster(ids, e['representative'], actions)['b']
        assert rebuilt == {s: e[s] for s in STATES}
    coverage = rows(natural / 'coverage_by_caption.jsonl')
    votes = [list(c['votes'].values()) for c in coverage]
    nc, cv = counts(ne), [sum(all(v is True for v in vs) for vs in votes), sum(any(v is True for v in vs) for vs in votes)]
    gate['checks']['noisy_pool'] = dict(counts=nc, covered=cv)
    dump(RUN / 'gates.json', gate)
    assert list(nc.values()) == p['gates']['noisy_pool']['b'] and cv == p['gates']['noisy_pool']['covered']
    predictions = rows(Path(inputs['coverage_check']) / 'predictions.jsonl')
    extra = {g: sum(v['group'] == g and (not v['valid'] or v['covered'] is not True) for v in predictions)
             for g in ('noisy_pool_a', 'retained_audit_b')}
    gate['checks']['coverage_check_cov_one_pass_rescued'] = extra
    dump(RUN / 'gates.json', gate)
    assert extra == p['gates']['coverage_check_cov_one_pass_rescued']
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(p['sampling']['new_seed']))
    selected = []
    old_sets = {tuple(sorted(c['record_ids'])) for c in old}
    for lo, hi, target in p['sampling']['strata_order']:
        pool = sorted((e for e in corpus if e['partition'] == 'test' and lo <= len(e['record_ids']) <= hi
                       and tuple(sorted(e['record_ids'])) not in old_sets), key=lambda e: min(e['record_ids']))
        assert len(pool) >= target
        for i in rng.choice(len(pool), size=target, replace=False):
            e = pool[int(i)]
            selected.append(dict(cluster_id=min(e['record_ids']), record_ids=sorted(e['record_ids']),
                                 component=e['component'], group='new150'))
    for c in old:
        e = by_ids[tuple(sorted(c['record_ids']))]
        selected.append(dict(cluster_id=c['cluster_id'], record_ids=sorted(c['record_ids']),
                             component=e['component'], group='original50'))
    selected.sort(key=lambda c: c['cluster_id'])
    assert len(selected) == 200 and len({c['cluster_id'] for c in selected}) == 200
    selected_ids = {rid for c in selected for rid in c['record_ids']}
    data = Path(inputs['dataset'])
    records = {r['record_id']: r for r in rows(data / 'public_records.jsonl')}
    for rid in selected_ids:
        for image in records[rid]['images']:
            image['resolved_image_path'] = image['image_path']
            assert Path(image['image_path']).is_file()
    order = sorted(selected_ids)
    random.Random(20261104).shuffle(order)
    frozen = dict(records=[records[r] for r in sorted(selected_ids)],
                  opaque_ids={rid: f'R{i+1:04d}' for i, rid in enumerate(order)})
    jobs = []
    for e in corpus:
        jobs.append(dict(key=f"benchmark:{e['partition']}:{e['component']}", domain='benchmark',
                         component=e['component'], partition=e['partition'], record_ids=e['record_ids'], b=e['b']))
    for e in ne:
        ids = sorted(ngroups[e['component']])
        jobs.append(dict(key=f"natural:{e['component']}", domain='natural', component=e['component'],
                         record_ids=ids, b={s: e[s] for s in STATES}))
    all_records = {rid: dict(caption=r['text'], clipscore=clip[rid]) for rid, r in records.items()}
    assert not (set(all_records) & set(nr)), 'Record namespace collision'
    all_records.update(nr)
    write_rows(RUN / 'jobs.jsonl', jobs)
    write_rows(RUN / 'sample_clusters.jsonl', selected)
    dump(RUN / 'audit_records.json', frozen)
    dump(RUN / 'records.json', all_records)
    gate['passed'] = True
    dump(RUN / 'gates.json', gate)
    print(json.dumps(gate), flush=True)


def text_key(rid, retained, records):
    return json.dumps(dict(deleted_caption=records[rid]['caption'],
                           retained_captions=[records[r]['caption'] for r in retained]), ensure_ascii=False)


def infer():
    assert os.environ.get('VL_DEDUP_GPU_LEASE') == '1'
    assert read(RUN / 'gates.json')['passed']
    assert not (RUN / 'exports.jsonl').exists()
    started = time.perf_counter()
    p, records, jobs = protocol(), read(RUN / 'records.json'), rows(RUN / 'jobs.jsonl')
    snapshot = subprocess.check_output(['nvidia-smi'], text=True)
    (RUN / 'nvidia_smi_before_load.txt').write_text(snapshot)
    print(snapshot, flush=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True)
    assert not processes.strip(), 'GPU occupied before load'
    cache = {}
    old = Path(p['inputs']['coverage_check'])
    inputs = {(i['group'], i['unit_id']): i for i in rows(old / 'inputs.jsonl')}
    for prediction in rows(old / 'predictions.jsonl'):
        inp = inputs[prediction['group'], prediction['unit_id']]
        # Preserve exact COVERAGE_CHECK_RUN retained ordering, not a set approximation.
        k = json.dumps(dict(deleted_caption=inp['deleted_caption'], retained_captions=[r['caption'] for r in inp['retained']]), ensure_ascii=False)
        cache[k] = dict(prediction, origin='coverage_check', origin_unit=[prediction['group'], prediction['unit_id']])
    assert not (RUN / 'predictions.jsonl').exists()
    sys.path.insert(0, p['model']['dependency_site'])
    os.environ['PYTHONPATH'] = p['model']['dependency_site']
    os.chdir(RUN)
    os.environ['VLLM_RPC_BASE_PATH'] = '.'
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    cfg = p['model']
    tokenizer = AutoTokenizer.from_pretrained(cfg['path'], local_files_only=True)
    kwargs = {k: v for k, v in cfg.items() if k not in ('name', 'path', 'backend', 'dependency_site')}
    llm = LLM(model=cfg['path'], **kwargs)
    sampling = SamplingParams(temperature=0, top_p=1.0, max_tokens=384, n=1)
    session = dict(status='running', model_requests=0, invalid=0, abstentions=0, cache_uses=0, checks=0)
    def evaluate(requests):
        missing = {}
        for job, rid, retained in requests:
            k = text_key(rid, retained, records)
            if k not in cache:
                missing.setdefault(k, (job, rid, retained))
        items = list(missing.items())
        for offset in range(0, len(items), 16):
            batch = items[offset:offset+16]
            rendered = []
            with (RUN / 'prompts.jsonl').open('a') as stream:
                for k, (job, rid, retained) in batch:
                    content = p['prompt_prefix'] + k
                    prompt = tokenizer.apply_chat_template([dict(role='user', content=content)], tokenize=False, add_generation_prompt=True)
                    tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                    assert tokens + 384 <= cfg['max_model_len']
                    rendered.append(prompt)
                    stream.write(json.dumps(dict(job=job, source_id=rid, retained_ids=retained, content=content, rendered=prompt, prompt_tokens=tokens), ensure_ascii=False)+'\n')
            session['model_requests'] += len(batch)
            dump(RUN / 'gpu_session.json', session)
            outputs = llm.generate(rendered, sampling, use_tqdm=False)
            assert len(outputs) == len(batch)
            with (RUN / 'predictions.jsonl').open('a') as stream:
                for (k, (job, rid, retained)), out in zip(batch, outputs, strict=True):
                    v = out.outputs[0]
                    parsed = coverage_check.parse_output(v.text, v.finish_reason)
                    prediction = dict(job=job, source_id=rid, retained_ids=retained, raw=v.text,
                                      finish_reason=v.finish_reason, completion_tokens=len(v.token_ids), **parsed, origin='coverage_export')
                    cache[k] = prediction
                    stream.write(json.dumps(prediction, ensure_ascii=False)+'\n')
                    session['invalid'] += not parsed['valid']
                    session['abstentions'] += parsed['valid'] and parsed['covered'] is None
            session['wall_seconds'] = time.perf_counter()-started
            dump(RUN / 'gpu_session.json', session)
            print(json.dumps(session), flush=True)
        answers = []
        with (RUN / 'checks.jsonl').open('a') as stream:
            for job, rid, retained in requests:
                v = cache[text_key(rid, retained, records)]
                answer = v['covered'] if v['valid'] else None
                stream.write(json.dumps(dict(job=job, source_id=rid, retained_ids=retained,
                                             covered=answer, valid=v['valid'], origin=v['origin'],
                                             origin_unit=v.get('origin_unit')), ensure_ascii=False)+'\n')
                answers.append(answer)
                session['checks'] += 1
                session['cache_uses'] += v['origin'] == 'coverage_check'
        return answers
    # Count-only noniterative arm also supplies cached first-pass inputs.
    requests = [(j['key']+':b_cov_one_pass', rid, sorted(j['b']['retained'])) for j in jobs
                for rid in sorted(j['b']['deleted'], key=lambda r: (-records[r]['clipscore'], r))]
    answers = evaluate(requests)
    noniter = Counter()
    for (key, rid, retained), answer in zip(requests, answers, strict=True):
        if answer is not True:
            noniter[key.removesuffix(':b_cov_one_pass')] += 1
    states = {j['key']: dict(retained=list(j['b']['retained']), pending=list(j['b']['pending']), deleted=[]) for j in jobs}
    orders = {j['key']: sorted(j['b']['deleted'], key=lambda r: (-records[r]['clipscore'], r)) for j in jobs}
    for rank in range(max(map(len, orders.values()))):
        requests = [(j['key']+':b_cov', orders[j['key']][rank], sorted(states[j['key']]['retained']))
                    for j in jobs if rank < len(orders[j['key']])]
        answers = evaluate(requests)
        for (key, rid, retained), answer in zip(requests, answers, strict=True):
            states[key.removesuffix(':b_cov')]['deleted' if answer is True else 'retained'].append(rid)
    exports = []
    for j in jobs:
        def checked(rid, retained):
            value = cache[text_key(rid, retained, records)]
            return value['covered'] if value['valid'] else None
        result = coverage_export(j['b'], {r: records[r]['clipscore'] for r in j['record_ids']}, checked)
        assert result == {s: sorted(states[j['key']][s]) for s in STATES}
        assert result['pending'] == sorted(j['b']['pending'])
        assert set(j['b']['retained']) <= set(result['retained'])
        assert set(result['deleted']) <= set(j['b']['deleted'])
        assert sum(map(len, result.values())) == len(j['record_ids'])
        exports.append(dict(j, b_cov=result, b_cov_one_pass_rescued=noniter[j['key']]))
    write_rows(RUN / 'exports.jsonl', exports)
    session.update(status='complete', wall_seconds=time.perf_counter()-started)
    dump(RUN / 'gpu_session.json', session)
    print(json.dumps(session), flush=True)


def prepare_audits():
    p = protocol()
    exports = rows(RUN / 'exports.jsonl')
    benchmark = {e['component']: e for e in exports if e['domain'] == 'benchmark'}
    sample = rows(RUN / 'sample_clusters.jsonl')
    frozen = read(RUN / 'audit_records.json')
    records = read(RUN / 'records.json')
    cells, kept = [], {}
    old_report = ROOT / p['inputs']['retained_audit_report']
    old_cells = rows(old_report / 'audit_cells.jsonl')
    old_audits = [{r['cell_id']: r for r in rows(old_report / f'audit_{i}.jsonl')} for i in (1, 2)]
    reusable = {}
    for c in old_cells:
        reusable[identity(c['source_id'], c['retained_ids'])] = [a[c['cell_id']] for a in old_audits]
    canonical, jobs, reuse = {}, [], []
    for cluster in sample:
        e = benchmark[cluster['component']]
        # The highest CLIPScore representative is frozen in the baseline job.
        rep = min(e['record_ids'], key=lambda rid: (-records[rid]['clipscore'], rid))
        strategies = dict(keep_one=dict(retained=[rep], pending=[], deleted=sorted(set(e['record_ids'])-{rep})), b=e['b'], b_cov=e['b_cov'])
        cluster['strategies'] = strategies
        for arm, result in strategies.items():
            arm_id = cluster['cluster_id']+':'+arm
            kept[arm_id] = result['retained']
            for rid in result['deleted']:
                cell = dict(cell_id=f"COVERAGE_EXPORT_RUN-{arm_id}-{rid}", source_id=rid, retained_ids=result['retained'],
                            arm=arm_id, strategy=arm, cluster_id=cluster['cluster_id'], group=cluster['group'], size=len(cluster['record_ids']))
                key = identity(rid, result['retained'])
                if key in canonical:
                    cell['reuse_of'] = canonical[key]
                elif key in reusable:
                    canonical[key] = cell['cell_id']
                    cell['reuse_of'] = None
                    reuse.append(dict(cell_id=cell['cell_id'], audits=reusable[key], origin='retained_audit_exact_input'))
                else:
                    canonical[key] = cell['cell_id']
                    cell['reuse_of'] = None
                    jobs.extend(dict(kind='benchmark', auditor=i, cell=cell, job_id=cell['cell_id']+f'-{i}') for i in (1, 2))
                cells.append(cell)
    frozen['retained_ids'] = kept
    dump(RUN / 'audit_frozen.json', frozen)
    write_rows(RUN / 'sample_exports.jsonl', sample)
    write_rows(RUN / 'audit_cells.jsonl', cells)
    write_rows(RUN / 'audit_reuse.jsonl', reuse)
    natural = Path(p['inputs']['noisy_pool'])
    nr = {r['record_id']: r for r in rows(natural / 'a_records.jsonl')}
    old_units = {u['deleted_id']: u for units in read(natural / 'a_audit_units.json').values() for u in units}
    old_votes = {c['deleted_id']: c for c in rows(natural / 'coverage_by_caption.jsonl')}
    n_units, n_reuse = [], []
    by_image = defaultdict(list)
    for e in exports:
        if e['domain'] != 'natural':
            continue
        for rid in e['b_cov']['deleted']:
            assert nr[rid]['sampled_image']
            unit = dict(image_id=nr[rid]['image_id'], deleted_id=rid, deleted_caption=nr[rid]['caption'],
                        retained=[dict(record_id=r, caption=nr[r]['caption']) for r in e['b_cov']['retained']])
            n_units.append(unit)
            old = old_units[rid]
            if unit['deleted_caption'] == old['deleted_caption'] and unit['retained'] == old['retained']:
                n_reuse.append(dict(deleted_id=rid, votes=old_votes[rid]['votes'], origin='noisy_pool_exact_input'))
            else:
                by_image[str(unit['image_id'])].append({k: v for k, v in unit.items() if k != 'image_id'})
    for image_id, units in sorted(by_image.items()):
        for i in (1, 2):
            jobs.append(dict(kind='natural', auditor=i, units=units, job_id=f'natural-{image_id}-{i}', image_id=image_id))
    write_rows(RUN / 'natural_units.jsonl', n_units)
    write_rows(RUN / 'natural_reuse.jsonl', n_reuse)
    write_rows(RUN / 'audit_jobs.jsonl', jobs)
    budget = dict(new_jobs=len(jobs), benchmark_jobs=sum(j['kind']=='benchmark' for j in jobs),
                  natural_jobs=sum(j['kind']=='natural' for j in jobs), benchmark_reused_units=len(reuse),
                  natural_reused_units=len(n_reuse), max_attempts=3000)
    dump(RUN / 'audit_budget.json', budget)
    assert len(jobs) <= 3000, budget
    print(json.dumps(budget), flush=True)


def audit():
    import requests
    secret = os.environ['AUDIT_API_SECRET']
    endpoint = os.environ['AUDIT_API_BASE_URL'].rstrip('/')
    url = endpoint if endpoint.endswith('/chat/completions') else endpoint+'/chat/completions'
    frozen, jobs = read(RUN / 'audit_frozen.json'), rows(RUN / 'audit_jobs.jsonl')
    h1 = coverage_audit
    assert h1.GUIDE == protocol()['audit']['benchmark_prompt']
    n_input = read(Path(protocol()['inputs']['noisy_pool']) / 'a_audit_2/105529_0_input.json')['messages'][0]['content']
    prefix = n_input.split('Input:\n', 1)[0]+'Input:\n'
    folder = RUN / 'audits'
    folder.mkdir(exist_ok=True)
    lock, stopped = threading.Lock(), threading.Event()
    attempts = [len(list(folder.glob('*/attempt_*.json')))]
    started = time.perf_counter()
    def one(job):
        target = folder / job['job_id']
        target.mkdir(exist_ok=True)
        terminal = target / 'terminal.json'
        if terminal.exists():
            return read(terminal)
        if job['kind'] == 'benchmark':
            messages, provenance = h1.make_messages(frozen, [job['cell']], job['cell']['arm'])
            paths = iter(provenance['image_index'].values())
            for item in messages[1]['content']:
                if item['type'] == 'image_url':
                    mime = mimetypes.guess_type(next(paths))[0] or 'image/jpeg'
                    item['image_url']['url'] = item['image_url']['url'].replace('data:image/jpeg;', f'data:{mime};', 1)
            dump(target / 'input.json', provenance)
        else:
            messages = [dict(role='user', content=prefix+json.dumps(job['units'], ensure_ascii=False))]
            dump(target / 'input.json', dict(messages=messages))
        payload = dict(model=MODELS[job['auditor']-1], messages=messages, max_completion_tokens=18000)
        for attempt in range(1, 4):
            receipt = target / f'attempt_{attempt}.json'
            if receipt.exists():
                meta = read(receipt)
                if meta['status'] == 'complete':
                    dump(terminal, meta['result'])
                    return meta['result']
                continue
            with lock:
                if stopped.is_set():
                    return None
                if attempts[0] >= 3000:
                    stopped.set()
                    dump(RUN / 'audit_stop.json', dict(reason='global3000_attempt_cap', attempts=attempts[0]))
                    return None
                attempts[0] += 1
                meta = dict(job_id=job['job_id'], attempt=attempt, admitted_index=attempts[0], status='admitted')
                dump(receipt, meta)
                dump(RUN / 'audit_progress.json', dict(attempts=attempts[0], wall_seconds=time.perf_counter()-started))
            begin = time.perf_counter()
            try:
                response = requests.post(url, headers={'Authorization': 'Bearer '+secret}, json=payload, timeout=(30, 900))
                raw = response.text.replace(secret, '[REDACTED]')
                (target / f'attempt_{attempt}_raw.txt').write_text(raw)
                meta['http_status'] = response.status_code
                if response.status_code in (401, 402, 403, 404):
                    stopped.set()
                    dump(RUN / 'audit_stop.json', dict(reason='http_'+str(response.status_code)))
                response.raise_for_status()
                obj = json.loads(raw)
                meta.update(usage=obj.get('usage'), model_returned=obj.get('model'), finish_reason=obj['choices'][0].get('finish_reason'))
                if job['kind'] == 'benchmark':
                    _, parsed = h1.parse_response(raw, [job['cell']], frozen, job['cell']['arm'])
                    answer = parsed[0]['audit']
                    assert len({w['record_id'] for w in answer['witnesses']}) == len(answer['witnesses'])
                    answer['label'] = LABEL_MAP[answer['label']]
                    result = dict(job_id=job['job_id'], kind='benchmark', auditor=job['auditor'], cell_id=job['cell']['cell_id'], audit=answer)
                else:
                    content = obj['choices'][0]['message']['content'].strip()
                    if content.startswith('```'):
                        content = content.split('\n', 1)[1].rsplit('```', 1)[0]
                    labels = json.loads(content)['labels']
                    assert len(labels) == len(job['units'])
                    assert {v['deleted_id'] for v in labels} == {u['deleted_id'] for u in job['units']}
                    assert all((v['covered'] is None or type(v['covered']) is bool) and isinstance(v['reason'], str) and v['reason'].strip() for v in labels)
                    result = dict(job_id=job['job_id'], kind='natural', auditor=job['auditor'], labels=labels)
                meta.update(status='complete', result=result)
            except Exception as error:
                meta.update(status='failed', error_type=type(error).__name__)
            meta['wall_seconds'] = time.perf_counter()-begin
            dump(receipt, meta)
            if meta['status'] == 'complete':
                dump(terminal, result)
                return result
        stopped.set()
        dump(RUN / 'audit_stop.json', dict(reason='three_attempt_failure', job_id=job['job_id']))
        return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(one, jobs))
    dump(RUN / 'audit_session.json', dict(attempts=attempts[0], jobs=len(jobs), complete=sum(v is not None for v in result), wall_seconds=time.perf_counter()-started))
    assert all(v is not None for v in result), 'Audit stopped; see audit_stop.json and preserved receipts'


def aggregate_votes(votes):
    values = list(votes)
    return dict(deleted=len(values), covered_strict=sum(all(v is True for v in vs) for vs in values),
                covered_loose=sum(any(v is True for v in vs) for vs in values),
                uncovered_strict=sum(all(v is False for v in vs) for vs in values),
                uncovered_loose=sum(any(v is False for v in vs) for vs in values),
                any_null=sum(any(v is None for v in vs) for vs in values))


def summarize():
    from scripts.experiments.support import normalize_text
    p = protocol()
    exports, sample = rows(RUN / 'exports.jsonl'), rows(RUN / 'sample_exports.jsonl')
    records = read(RUN / 'records.json')
    checks = rows(RUN / 'checks.jsonl')
    current_references = {c['source_id']: c['retained_ids'] for c in checks if c['job'].endswith(':b_cov')}
    lineage = {r['record_id']: r for r in rows(Path(p['inputs']['dataset']) / 'private/lineage.jsonl')}
    cells = rows(RUN / 'audit_cells.jsonl')
    audits = {}
    for reuse in rows(RUN / 'audit_reuse.jsonl'):
        audits[reuse['cell_id']] = [dict(a['audit'], origin='retained_audit_exact_input', original_cell_id=a['cell_id']) for a in reuse['audits']]
    for path in (RUN / 'audits').glob('*/terminal.json'):
        value = read(path)
        if value['kind'] == 'benchmark':
            audits.setdefault(value['cell_id'], [None, None])[value['auditor']-1] = value['audit']
    for c in cells:
        if c['reuse_of']:
            audits[c['cell_id']] = audits[c['reuse_of']]
        assert len(audits[c['cell_id']]) == 2 and all(audits[c['cell_id']])
        for answer in audits[c['cell_id']]:
            assert len({w['record_id'] for w in answer['witnesses']}) == len(answer['witnesses'])
    joined = [dict(c, votes=[a['label'] for a in audits[c['cell_id']]], audits=audits[c['cell_id']]) for c in cells]
    write_rows(RUN / 'benchmark_audits.jsonl', joined)
    table = []
    for group in ('original50', 'new150', 'total200'):
        for stratum in ('all', '2-4', '5'):
            selected = [c for c in sample if (group == 'total200' or c['group'] == group)
                        and (stratum == 'all' or (2 <= len(c['record_ids']) <= 4 if stratum == '2-4' else len(c['record_ids']) == 5))]
            cluster_ids = {c['cluster_id'] for c in selected}
            reference = [c for c in joined if c['cluster_id'] in cluster_ids and c['strategy'] == 'keep_one']
            ref_strict = sum(all(v == 'lost' for v in c['votes']) for c in reference)
            ref_loose = sum(any(v == 'lost' for v in c['votes']) for c in reference)
            for arm in ARMS:
                chosen = [c for c in joined if c['cluster_id'] in cluster_ids and c['strategy'] == arm]
                numbers = counts([c['strategies'][arm] for c in selected])
                d = numbers['deleted']
                exact = [Fraction(d * n, len(reference)) if reference else Fraction(0) for n in (ref_strict, ref_loose)]
                fate = {}
                for mode, predicate in [('strict', all), ('loose', any)]:
                    lost_sources = [(c['cluster_id'], c['source_id']) for c in reference if predicate(v == 'lost' for v in c['votes'])]
                    lookup = {c['cluster_id']: c for c in selected}
                    fate[mode] = {s: sum(rid in lookup[cid]['strategies'][arm][s] for cid, rid in lost_sources) for s in STATES}
                    assert sum(fate[mode].values()) == len(lost_sources)
                table.append(dict(group=group, stratum=stratum, clusters=len(selected), records=sum(len(c['record_ids']) for c in selected),
                                  strategy=arm, **numbers, lost_strict=sum(all(v == 'lost' for v in c['votes']) for c in chosen),
                                  lost_loose=sum(any(v == 'lost' for v in c['votes']) for c in chosen),
                                  random_lost_strict=float(exact[0]), random_lost_loose=float(exact[1]),
                                  random_exact=[str(x) for x in exact], keep_one_lost_fate=fate,
                                  any_uncertain=sum('uncertain' in c['votes'] for c in chosen)))
    full, rescues, identical = {}, [], []
    for part in ('train', 'validation', 'test'):
        selected = [e for e in exports if e['domain'] == 'benchmark' and e['partition'] == part]
        transitions = Counter()
        for e in selected:
            prior = {rid: s for s in STATES for rid in e['b'][s]}
            for s in STATES:
                for rid in e['b_cov'][s]:
                    transitions[prior[rid]+'->'+s] += 1
            for rid in sorted(set(e['b_cov']['retained'])-set(e['b']['retained'])):
                equal = [r for r in e['b']['retained'] if normalize_text(records[r]['caption']) == normalize_text(records[rid]['caption'])]
                current_equal = [r for r in current_references[rid] if normalize_text(records[r]['caption']) == normalize_text(records[rid]['caption'])]
                other_equal = [r for r in e['b_cov']['retained'] if r != rid and normalize_text(records[r]['caption']) == normalize_text(records[rid]['caption'])]
                row = dict(domain='benchmark', partition=part, component=e['component'], source_id=rid,
                           level=lineage[rid]['original_level'], source_dataset=lineage[rid]['source_dataset'],
                           normalized_equal_b_retained=equal, normalized_equal_other_final_retained=other_equal,
                           normalized_equal_current_retained=current_equal,
                           b_retained=e['b']['retained'], retained_ids=e['b_cov']['retained'])
                rescues.append(row)
                if equal or other_equal:
                    identical.append(row)
        part_rescues = [r for r in rescues if r['partition'] == part]
        full[part] = dict(b=counts([e['b'] for e in selected]), b_cov=counts([e['b_cov'] for e in selected]),
                          transitions=dict(transitions), rescued=len(part_rescues), rescued_levels=dict(Counter(r['level'] for r in part_rescues)),
                          rescued_sources=dict(Counter(r['source_dataset'] for r in part_rescues)),
                          normalized_equal_b_retained=sum(bool(r['normalized_equal_b_retained']) for r in part_rescues),
                          normalized_equal_current_retained=sum(bool(r['normalized_equal_current_retained']) for r in part_rescues),
                          normalized_equal_other_final_retained=sum(bool(r['normalized_equal_other_final_retained']) for r in part_rescues),
                          b_cov_one_pass_rescued=sum(e['b_cov_one_pass_rescued'] for e in selected))
        full[part]['b_cov_one_pass_counts'] = {s: full[part]['b'][s] + (full[part]['b_cov_one_pass_rescued'] if s == 'retained' else
            -full[part]['b_cov_one_pass_rescued'] if s == 'deleted' else 0) for s in STATES}
    write_rows(RUN / 'rescued_benchmark.jsonl', rescues)
    write_rows(RUN / 'rescued_normalized_equal.jsonl', identical)
    n_run = Path(p['inputs']['noisy_pool'])
    nr = {r['record_id']: r for r in rows(n_run / 'a_records.jsonl')}
    n_exports = [e for e in exports if e['domain'] == 'natural']
    n_votes = {u['deleted_id']: u['votes'] for u in rows(RUN / 'natural_reuse.jsonl')}
    n_reused = set(n_votes)
    n_reasons = defaultdict(dict)
    for path in (n_run / 'a_audit_2').glob('*_terminal.json'):
        value = read(path)
        for label in value['labels']:
            if label['deleted_id'] in n_reused:
                n_reasons[label['deleted_id']][value['model']] = label['reason']
    for path in (RUN / 'audits').glob('*/terminal.json'):
        value = read(path)
        if value['kind'] == 'natural':
            model = MODELS[value['auditor']-1]
            for label in value['labels']:
                n_votes.setdefault(label['deleted_id'], {})[model] = label['covered']
                n_reasons[label['deleted_id']][model] = label['reason']
    units = rows(RUN / 'natural_units.jsonl')
    assert set(n_votes) == {u['deleted_id'] for u in units}
    assert all(set(v) == set(MODELS) for v in n_votes.values())
    write_rows(RUN / 'natural_coverage.jsonl', [dict(u, votes=n_votes[u['deleted_id']], reasons=n_reasons[u['deleted_id']],
               origin='noisy_pool_exact_input' if u['deleted_id'] in n_reused else 'coverage_export_new_audit') for u in units])
    natural = {}
    for arm in ('b', 'b_cov'):
        by_image = defaultdict(Counter)
        for e in n_exports:
            for s in STATES:
                for rid in e[arm][s]:
                    if nr[rid]['sampled_image']:
                        by_image[nr[rid]['image_id']][s] += 1
        assert len(by_image) == 500
        votes = [c['votes'].values() for c in rows(n_run / 'coverage_by_caption.jsonl')] if arm == 'b' else [v.values() for v in n_votes.values()]
        natural[arm] = dict(**counts([e[arm] for e in n_exports]), images=500, clusters=505,
                            retained_per_image_distribution=dict(sorted(Counter(v['retained'] for v in by_image.values()).items())),
                            coverage=aggregate_votes(votes))
    natural['b_cov_one_pass_rescued'] = sum(e['b_cov_one_pass_rescued'] for e in n_exports)
    natural['b_cov_one_pass_counts'] = {s: natural['b'][s] + (natural['b_cov_one_pass_rescued'] if s == 'retained' else -natural['b_cov_one_pass_rescued'] if s == 'deleted' else 0) for s in STATES}
    natural['rescued'] = natural['b_cov']['retained']-natural['b']['retained']
    for e in n_exports:
        for rid in sorted(set(e['b_cov']['retained'])-set(e['b']['retained'])):
            rescues.append(dict(domain='natural', source_id=rid, retained_ids=e['b_cov']['retained'], b_retained=e['b']['retained']))
    bad = [dict(domain='benchmark', source_id=c['source_id'], retained_ids=c['retained_ids'], votes=c['votes'], audits=c['audits'])
           for c in joined if c['strategy'] == 'b_cov' and any(v == 'lost' for v in c['votes'])]
    bad.extend(dict(domain='natural', source_id=u['deleted_id'], retained_ids=[r['record_id'] for r in u['retained']],
                    votes=n_votes[u['deleted_id']], reasons=n_reasons[u['deleted_id']]) for u in units if any(v is False for v in n_votes[u['deleted_id']].values()))
    write_rows(RUN / 'rescued_records.jsonl', rescues)
    write_rows(RUN / 'auditor_lost_records.jsonl', bad)
    receipts = [read(path) for path in (RUN / 'audits').glob('*/attempt_*.json')]
    assert len(receipts) <= 3000
    predictions = rows(RUN / 'predictions.jsonl')
    assert len(predictions) == read(RUN / 'gpu_session.json')['model_requests']
    by_component = {e['component']: e for e in exports if e['domain'] == 'benchmark'}
    sample_cov_one_pass = {}
    for group in ('original50', 'new150', 'total200'):
        chosen = [by_component[c['component']] for c in sample if group == 'total200' or c['group'] == group]
        baseline = counts([e['b'] for e in chosen])
        rescued = sum(e['b_cov_one_pass_rescued'] for e in chosen)
        sample_cov_one_pass[group] = dict(rescued=rescued, counts={s: baseline[s]+(rescued if s == 'retained' else -rescued if s == 'deleted' else 0) for s in STATES})
    assert sample_cov_one_pass['original50']['rescued'] == 17
    assert natural['b_cov_one_pass_rescued'] == 422
    check_counts = {}
    for domain in ('benchmark', 'natural'):
        for arm in ('b_cov_one_pass', 'b_cov'):
            chosen = [c for c in checks if c['job'].startswith(domain+':') and c['job'].endswith(':'+arm)]
            new = [c for c in predictions if c['job'].startswith(domain+':') and c['job'].endswith(':'+arm)]
            check_counts[domain+':'+arm] = dict(checks=len(chosen), invalid=sum(not c['valid'] for c in chosen),
                null=sum(c['valid'] and c['covered'] is None for c in chosen), coverage_check_reused=sum(c['origin']=='coverage_check' for c in chosen),
                new_model_requests=len(new), new_invalid=sum(not c['valid'] for c in new))
    result = dict(run_id=p['run_id'], evidence_status='archived-complete',
                  gates=read(RUN / 'gates.json'), commits=dict(protocol=subprocess.check_output(['git','log','-1','--format=%H','--',str(CONFIG / 'coverage_export.json')],cwd=ROOT,text=True).strip(),
                  implementation=subprocess.check_output(['git','log','-1','--format=%H','--',str(ROOT/'tests/test_coverage_export.py')],cwd=ROOT,text=True).strip()),
                  full_corpus=full, benchmark_audit=table, benchmark_cov_one_pass=sample_cov_one_pass, natural=natural, check_counts=check_counts,
                  gpu=read(RUN/'gpu_session.json'), audit=read(RUN/'audit_session.json'), audit_budget=read(RUN/'audit_budget.json'),
                  audit_attempts=len(receipts), audit_failed_attempts=sum(r['status']!='complete' for r in receipts),
                  normalized_caption_rule='scripts.strict_release_qc.normalize_text: NFKC, lowercase, punctuation to spaces, whitespace collapse',
                  interpretation='Strict/loose are dual-AI aggregation ranges, not human labels. Same graphs, edges and pair actions. No threshold selection.',
                  resource_directory=str(RUN))
    dump(REPORT / 'results.json', result)
    print(json.dumps(dict(audit_attempts=len(receipts), full_corpus=full, natural=natural)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'infer', 'prepare_audits', 'audit', 'summarize'])
    phase = parser.parse_args().phase
    started = time.perf_counter()
    try:
        globals()[phase]()
    except Exception as error:
        if phase == 'infer':
            path = RUN / 'gpu_session.json'
            session = read(path) if path.exists() else {}
            session.update(status='failed', error_type=type(error).__name__, wall_seconds=time.perf_counter()-started)
            dump(path, session)
        raise


if __name__ == '__main__':
    main()
