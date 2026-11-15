import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
import pandas as pd
from scripts.caption_action import reference as ref
ROOT = REPO_ROOT
RES = RESOURCES
OUT = RES / 'runs/caption_action'
REPORT = ROOT / str(REPORTS / 'caption_action')
OBS = RES / 'runs/pair_features'
RUN_ID = OUT.name

def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if Path(path).exists() else []

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def write_rows(path, rows):
    Path(path).write_text(''.join((json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n' for x in rows)))

def append(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')

def select_sample(frame, reused, excluded, seed=20_261_026):
    rng = random.Random(seed)
    selected = []
    for grade in ['L2', 'L3']:
        group = frame[frame.fine_grade.eq(grade)].sort_values(['record_id_a', 'record_id_b', 'pair_id'])
        forced = group[group.pair_id.isin(reused)]
        candidates = group[~group.pair_id.isin(excluded)]
        take = 200 - len(forced)
        if take < 0 or len(candidates) < take:
            raise ValueError('Insufficient stratum or too many reused pairs')
        picked = candidates.iloc[sorted(rng.sample(range(len(candidates)), take))]
        for row in pd.concat([forced, picked]).sort_values(['record_id_a', 'record_id_b', 'pair_id']).to_dict('records'):
            selected.append(dict(pair_id=row['pair_id'], record_id_a=row['record_id_a'], record_id_b=row['record_id_b'], grade=grade, partition='test', same_cluster=True, reused=row['pair_id'] in reused))
    return selected

def metrics(sample, predictions, labels):
    ids = [r['pair_id'] for r in sample]
    observed = [k for k in ids if predictions.get(k) is not None and labels.get(k) not in [None, 'uncertain']]
    tp = sum((predictions[k] and labels[k] == 'complementary' for k in observed))
    fp = sum((predictions[k] and labels[k] != 'complementary' for k in observed))
    fn = sum((not predictions[k] and labels[k] == 'complementary' for k in observed))
    redundant = sum((labels[k] == 'redundant' for k in observed))
    redundant_ka = sum((predictions[k] and labels[k] == 'redundant' for k in observed))
    unknown_pos = sum((predictions.get(k) is True and labels.get(k) in [None, 'uncertain'] for k in ids))
    unknown_neg = sum((predictions.get(k) is False and labels.get(k) in [None, 'uncertain'] for k in ids))
    missing_pred = sum((predictions.get(k) is None for k in ids))
    div = lambda a, b: a / b if b else None
    bounds = None if missing_pred else dict(precision_lower=div(tp, tp + fp + unknown_pos), precision_upper=div(tp + unknown_pos, tp + fp + unknown_pos), recall_lower=div(tp, tp + fn + unknown_neg), recall_upper=div(tp + unknown_pos, tp + fn + unknown_pos))
    return dict(total=len(ids), observed=len(observed), missing_prediction=missing_pred, unresolved_reference=sum((labels.get(k) in [None, 'uncertain'] for k in ids)), reference_counts=dict(Counter((labels.get(k) or 'unresolved' for k in ids))), tp=tp, fp=fp, fn=fn, tn=len(observed) - tp - fp - fn, precision=div(tp, tp + fp), recall=div(tp, tp + fn), predicted_keep_augment=sum((predictions.get(k) is True for k in ids)), known_complementary=tp + fn, known_redundant=redundant, action_nonempty_confirmed_count=tp, action_nonempty_confirmed_rate=div(tp, len(ids)), action_nonempty_unresolved_predictions=unknown_pos, redundant_false_keep_augment=redundant_ka, redundant_false_positive_rate=div(redundant_ka, redundant), redundant_fraction_of_known_keep_augment=div(redundant_ka, tp + fp), full_sample_bounds=bounds)

def classify(precision, recall):
    if precision is None:
        return 'not_evaluable'
    if precision < 0.6:
        return 'no_material_difference_from_band_by_preregistered_rule'
    if recall is None:
        return 'not_evaluable'
    if precision >= 0.75 and recall >= 0.5:
        return 'caption_layer_positive_on_observed_AI_reference'
    return 'inconclusive'

def full_sample_decision(item):
    bounds = item['full_sample_bounds']
    if bounds is None:
        return 'not_evaluable_missing_predictions'
    if bounds['precision_upper'] is not None and bounds['precision_upper'] < 0.6:
        return 'negative_rule_certified_by_reference_bounds'
    if bounds['precision_lower'] is not None and bounds['recall_lower'] is not None and (bounds['precision_lower'] >= 0.75) and (bounds['recall_lower'] >= 0.5):
        return 'positive_rule_certified_by_reference_bounds'
    return 'pending_reference_adjudication' if item['unresolved_reference'] else classify(item['precision'], item['recall'])

def reference_issues(vote, row):
    issues = ref.evidence_issues(vote, row)
    if vote.get('label') in ref.LABELS:
        if not isinstance(vote.get('explanation'), str) or not vote['explanation'].strip():
            issues.append('missing_required_explanation')
        if vote.get('confidence') not in ['high', 'medium', 'low']:
            issues.append('invalid_guide_confidence')
    if vote.get('label') == 'one_unsupported' and (not isinstance(vote.get('unsupported_claims'), list) or not vote['unsupported_claims']):
        issues.append('one_unsupported_requires_claim_evidence')
    return issues

def summarize():
    protocol = json.loads((OUT / 'settings.json').read_text())
    sample = protocol['sample']
    votes = {i: {x['pair_id']: x for x in read_rows(OUT / f'audit_{i}.jsonl')} for i in [1, 2]}
    (agreements, labels) = ([], {})
    per_auditor = {i: {} for i in [1, 2]}
    for row in sample:
        rid = row['pair_id']
        pair = [votes[i].get(rid, {}) for i in [1, 2]]
        issues = [reference_issues(v.get('vote') or {}, row) for v in pair]
        values = [v.get('label') if v.get('status') == 'completed' and (not issue) else None for (v, issue) in zip(pair, issues)]
        consensus = values[0] if values[0] == values[1] and values[0] not in [None, 'uncertain'] else None
        subtype = consensus == 'redundant' and pair[0]['vote']['redundancy_kind'] != pair[1]['vote']['redundancy_kind']
        if subtype:
            pass
        labels[rid] = consensus
        for (i, v) in enumerate(values, 1):
            per_auditor[i][rid] = v
        item = dict(pair_id=rid, auditor_labels=[v.get('label') for v in pair], auditor_statuses=[v.get('status', 'not_run') for v in pair], ai_consensus_label=consensus, evidence_issues=issues, redundancy_subtype_disagreement=subtype)
        agreements.append(item)
    vlm_rows = read_rows(OUT / 'vlm_verdicts.jsonl')
    vlms = {r['pair_id']: r for r in vlm_rows}
    decisions = {k: r['parsed']['decision'] if r['status'] == 'completed' and r['parsed']['contract_valid'] else None for (k, r) in vlms.items()}
    predictions = {k: v == 'keep_augment' if v is not None else None for (k, v) in decisions.items()}
    scopes = {'all': sample, 'L2': [r for r in sample if r['grade'] == 'L2'], 'L3': [r for r in sample if r['grade'] == 'L3'], 'new_only': [r for r in sample if not r['reused']], 'reused_only': [r for r in sample if r['reused']]}
    scored = {name: metrics(rows, predictions, labels) for (name, rows) in scopes.items()}
    main = scored['all']
    either = {k: v in ['keep_augment', 'mismatch'] if v is not None else None for (k, v) in decisions.items()}
    results = dict(task='caption_action', run_id=RUN_ID, population=protocol['population'], metrics=scored, per_auditor={str(i): {name: metrics(rows, predictions, per_auditor[i]) for (name, rows) in scopes.items()} for i in [1, 2]}, either_caption_action=metrics(sample, either, labels), decision_counts=dict(Counter(decisions.values())), decision=classify(main['precision'], main['recall']), full400_decision=full_sample_decision(main))
    for destination in [OUT, REPORT]:
        write_rows(destination / 'content_labels.jsonl', agreements)
        write_json(destination / 'results.json', results)
    write_rows(REPORT / 'vlm_verdicts.jsonl', [{k: v for (k, v) in r.items() if k != 'rendered_prompt'} for r in vlm_rows])
    write_rows(REPORT / 'audit_votes.jsonl', [v for i in [1, 2] for v in votes[i].values()])

def freeze():
    protocol = json.loads((REPORT / 'settings.json').read_text())
    selected = protocol['sample']
    manifest = pd.read_parquet(OBS / 'public_manifest.parquet').set_index('record_id')
    from scripts import stage_e
    jobs = []
    for row in selected:
        for side in ['a', 'b']:
            record = manifest.loc[row['record_id_' + side]]
            (row['caption_' + side], row['image_' + side]) = (record.text, record.image_path)
        prompt_row = dict(edge_id=row['pair_id'], record_id_a=row['record_id_a'], record_id_b=row['record_id_b'], image_path_a_context=row['image_a'], image_path_b_context=row['image_b'], text_a_context=row['caption_a'], text_b_context=row['caption_b'], identical_pixels=stage_e.identical_pixels(row['image_a'], row['image_b']))
        prompt = stage_e.build_prompt_record(pd.Series(prompt_row), RES / 'datasets/benchmark', ROOT)
        jobs.append(dict(pair_id=row['pair_id'], prompt_record=prompt, messages=stage_e.build_qwen_messages(prompt)))
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'settings.json', protocol)
    write_rows(OUT / 'sample.jsonl', selected)
    write_rows(OUT / 'jobs.jsonl', jobs)

def audit():
    """Ask both AI auditors for a caption-action label on each newly sampled pair.

    Pairs marked reused already carry the labels of an earlier audit in audit_1.jsonl and audit_2.jsonl.
    """
    from scripts import ai_audit
    from scripts.stage_e import extract_json_object
    sample = [r for r in read_rows(OUT / 'sample.jsonl') if not r['reused']]
    for auditor in [1, 2]:
        for row in sample:
            content = [dict(type='text', text='Caption A: ' + row['caption_a'] + '\nCaption B: ' + row['caption_b'])]
            content += [ai_audit.image_part(path) for path in dict.fromkeys([row['image_a'], row['image_b']])]
            messages = [dict(role='system', content=ref.GUIDE), dict(role='user', content=content)]
            vote = extract_json_object(ai_audit.content(ai_audit.chat(auditor, messages, 2200)))
            status = 'completed' if ref.valid_vote(vote) else 'invalid_output'
            append(OUT / f'audit_{auditor}.jsonl', dict(auditor=auditor, model=ai_audit.AUDITOR_MODELS[auditor], pair_id=row['pair_id'], vote=vote, label=(vote or {}).get('label'), status=status))

def infer():
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    jobs = read_rows(OUT / 'jobs.jsonl')
    cfg = json.loads((OUT / 'settings.json').read_text())['inference']
    processor = AutoProcessor.from_pretrained(cfg['model'], local_files_only=True, min_pixels=3136, max_pixels=802816)
    llm = LLM(model=cfg['model'], dtype='float16', max_model_len=8192, max_num_seqs=4, max_num_batched_tokens=4096, enforce_eager=True, enable_chunked_prefill=True, limit_mm_per_prompt={'image': 2}, mm_processor_kwargs={'min_pixels': 3136, 'max_pixels': 802816}, gpu_memory_utilization=0.92, disable_log_stats=True)
    settings = SamplingParams(temperature=0, max_tokens=384)
    results = []
    for job in jobs:
        start = time.monotonic()
        images = [Image.open(job['prompt_record'][key]).convert('RGB') for key in ['image_path_a', 'image_path_b']]
        try:
            prompt = processor.apply_chat_template(job['messages'], tokenize=False, add_generation_prompt=True)
            output = llm.generate([dict(prompt=prompt, multi_modal_data={'image': images})], settings, use_tqdm=False)[0]
            raw = output.outputs[0]
            results.append(dict(pair_id=job['pair_id'], status='completed', raw_return=raw.text, finish_reason=raw.finish_reason, prompt_tokens=len(output.prompt_token_ids), output_tokens=len(raw.token_ids), rendered_prompt=prompt, parsed=ref.strict_parse(raw.text), elapsed_seconds=time.monotonic() - start))
        finally:
            for image in images:
                image.close()
    write_rows(OUT / 'vlm_verdicts.jsonl', results)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['freeze', 'audit', 'infer', 'summarize'])
    a = p.parse_args()
    globals()[a.action]()
