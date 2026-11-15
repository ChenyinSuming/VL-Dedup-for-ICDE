import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import json
from pathlib import Path
import re
try:
    from scripts.prepare_captions import rows, write_rows, dump, pair_mask
except ModuleNotFoundError:
    from prepare_captions import rows, write_rows, dump, pair_mask
BLOCKED = ('class_id', 'super_class_id', 'split=', 'near duplicate view', 'stanford online products', 'listing metadata', 'vlb50k_', 'source_image_id', 'record_id')
MATERIALS = {'wood', 'wooden', 'metal', 'metallic', 'brass', 'chrome', 'ceramic', 'leather', 'fabric', 'plastic', 'vintage'}

def qc(response):
    text = (response.get('response') or '').strip()
    hard = []
    flags = []
    if response.get('status') != 'generated':
        hard.append(response.get('status', 'no_output'))
    if not text:
        hard.append('empty')
    lower = text.lower()
    if any((term in lower for term in BLOCKED)):
        hard.append('construction_metadata_text')
    if any((x in lower for x in ('cannot describe', 'unable to', 'as an ai', 'i cannot'))):
        hard.append('refusal')
    words = re.findall('[A-Za-z]+', lower)
    if len(words) < 5:
        hard.append('insufficient_description')
    if len(words) > 40:
        flags.append('long_description')
    if MATERIALS.intersection(words):
        flags.append('material_or_age_not_verified')
    if any((x in lower for x in ('brand', 'labeled', 'labelled', 'text:', 'written', 'logo'))) or chr(34) in text:
        flags.append('printed_text_or_brand_claim')
    if any((x in words for x in ('one', 'two', 'three', 'four', 'five', 'six', 'digital', 'led', 'purifier'))):
        flags.append('fine_detail_or_function_not_verified')
    return {'text': None if hard else text, 'status': 'missing_after_qc' if hard else 'generated_unverified', 'hard_failures': hard, 'quality_flags': flags, 'support_gold_mask': False}

def finalize(root):
    target = root / 'public_records.jsonl'
    contract = json.loads((root / 'contract.json').read_text())
    generation = json.loads((root / 'generation/full/config.json').read_text())
    if generation['model'] != contract['generation_model']:
        raise ValueError('Generation model differs from revision contract')
    actual_requests = rows(root / 'generation/full/requests.jsonl')
    expected_requests = rows(root / 'generation/requests.jsonl')
    if actual_requests != expected_requests:
        raise ValueError('Generation requests differ from revision contract')
    records = rows(root / 'private/staged_records.jsonl')
    lineage = rows(root / 'private/lineage.jsonl')
    responses = rows(root / 'generation/full/responses.jsonl')
    lookup = {r['id']: r for r in responses}
    pending = {r['record_id'] for r in records if r['text'] is None}
    if len(lookup) != len(responses) or set(lookup) != pending:
        raise ValueError('Generation must have exactly one terminal outcome per SOP record')
    quality = []
    for r in records:
        if r['record_id'] in pending:
            q = qc(lookup[r['record_id']])
            r['text'] = q['text']
            quality.append({'record_id': r['record_id'], **q, 'model': generation['model'], 'source_response': 'generation/full/responses.jsonl', 'action_if_missing': 'retain; skip text-dependent decisions'})
    write_rows(root / 'private/caption_quality.jsonl', quality)
    source = Path(contract['source_root'])
    original = {r['record_id']: r for r in rows(source / 'data/manifests/manifest.jsonl')}
    mapping = {r['original_record_id']: r['record_id'] for r in lineage}
    labels = [pair_mask(p, original) | {'record_id_a': mapping[p['record_id_a']], 'record_id_b': mapping[p['record_id_b']], 'partition': p['partition']} for p in rows(source / 'data/manifests/pairs_eval.jsonl')]
    write_rows(root / 'sealed/relation_applicability.jsonl', labels)
    status = Counter((r['status'] for r in quality))
    receipt = {'records': len(records), 'coco_records': len(records) - len(pending), 'generated_SOP_records': len(pending), 'caption_statuses': dict(status), 'quality_flag_counts': dict(Counter((f for r in quality for f in r['quality_flags']))), 'relation_rows': len(labels), 'coarse_relation_applicable': sum((r['graded_relation_mask'] for r in labels)), 'fine_grade_applicable': sum((r['fine_grade_mask'] for r in labels)), 'fine_grade_unresolved_after_revision': sum((not r['fine_grade_mask'] for r in labels)), 'exact_record_rule_positives': sum((r['exact_record_rule_positive'] for r in labels))}
    contract['generated_caption_missing'] = status['missing_after_qc']
    contract['fine_grade_policy'] = 'affected L0 masked; exact-record positives rule-derived separately; coarse visual relations retained'
    temporary = root / 'public_records.jsonl.tmp'
    write_rows(temporary, records)
    temporary.replace(target)
    dump(root / 'contract.json', contract)
    print(json.dumps(receipt), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    finalize(p.parse_args().root)
