import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
PROMPT = 'Write exactly one short English sentence of 15-30 words describing this image. State only the main object type, visible colors, shape, parts and arrangement. Omit every brand, model, word printed on the object, material, quality, age, compatibility, intended use and hidden property. Do not infer object identity. Do not use quotations, headings, bullet points or introductory phrases. Output only the sentence.'
MARKER = ' near duplicate view'

def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

def write_rows(path, values):
    with Path(path).open('w') as f:
        for row in values:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def clean_text(row):
    if row['source_dataset'] == 'stanford_online_products':
        return (None, 'image_only_generated_caption_pending')
    text = row['text']
    if row['label_tags'] == 'synthetic_near_copy' and text.endswith(MARKER):
        return (text[:-len(MARKER)], 'remove_construction_suffix')
    return (text, 'preserve_original_text')

def pair_mask(pair, source_by_id):
    sop = any((source_by_id[pair[k]]['source_dataset'] == 'stanford_online_products' for k in ('record_id_a', 'record_id_b')))
    visual = {'L0': 'same', 'L1': 'different', 'L2': 'same', 'L3': 'same', 'M': None, 'HN': 'different'}[pair['grade']]
    (a, b) = (source_by_id[pair[k]] for k in ('record_id_a', 'record_id_b'))
    ((ta, ca), (tb, cb)) = (clean_text(a), clean_text(b))
    changed = 'remove_construction_suffix' in (ca, cb)
    fine_valid = not (changed and pair['grade'] == 'L0')
    exact = ta is not None and ta == tb and (a['image_path'] == b['image_path'])
    return {'pair_id': pair['pair_id'], 'use_scope': 'historical_construction_analysis_only', 'endpoint_text_revisions': {'A': ca, 'B': cb}, 'semantic_reference_status': 'independent_reference_unavailable', 'grade': pair['grade'], 'relation': pair['relation'], 'visual_relation': visual, 'visual_relation_mask': pair['grade'] != 'M', 'graded_relation_mask': not sop or pair['grade'] == 'HN', 'fine_grade': pair['grade'] if fine_valid else None, 'fine_grade_mask': fine_valid, 'exact_record_rule_positive': exact, 'text_support_mask': False, 'claim_coverage_mask': False, 'record_deletion_gold_mask': False, 'label_origin': 'existing_controlled_relation_annotation', 'reason': 'SOP visual hard negative remains; generated caption support unknown' if sop else 'coarse relation retained; affected L0 grade masked; no claim or semantic deletion gold implied'}

def prepare(source, output, model, seed=23):
    data = rows(source / 'data/manifests/manifest.jsonl')
    pairs = rows(source / 'data/manifests/pairs_eval.jsonl')
    if len(data) != 50000 or len(pairs) != 66788:
        raise ValueError('Expected canonical controlled 50K inputs')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('private', 'sealed', 'images', 'generation'):
        (output / name).mkdir()
    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)
    remap = {r['record_id']: f'record_{i:06d}' for (i, r) in enumerate(shuffled)}
    paths = sorted({r['image_path'] for r in data})
    random.Random(seed).shuffle(paths)
    image_ids = {p: f'image_{i:06d}' for (i, p) in enumerate(paths)}
    for (p, image_id) in image_ids.items():
        original = (source / p).resolve()
        (output / 'images' / (image_id + original.suffix)).symlink_to(original)
    (records, lineage, requests, buckets) = ([], [], [], defaultdict(list))
    for row in data:
        (rid, iid) = (remap[row['record_id']], image_ids[row['image_path']])
        image = output / 'images' / (iid + Path(row['image_path']).suffix)
        (text, action) = clean_text(row)
        records.append({'record_id': rid, 'images': [{'image_id': iid, 'image_path': str(image.absolute()), 'role': None}], 'text': text, 'association_scope': 'single_image'})
        lineage.append({'record_id': rid, 'original_record_id': row['record_id'], 'original_image_path': row['image_path'], 'source_dataset': row['source_dataset'], 'partition': row['partition'], 'source_image_id': row['source_image_id'], 'visual_group_id': row['visual_group_id'], 'original_text': row['text'], 'original_level': row['level'], 'text_revision': action, 'caption_origin': 'image_only_model_generated' if text is None else 'controlled_coco_text', 'caption_gold_mask': False, 'generation_model': str(model) if text is None else None})
        if text is None:
            request = {'id': rid, 'messages': [{'role': 'user', 'content': [{'type': 'image', 'url': str(image.absolute())}, {'type': 'text', 'text': PROMPT}]}]}
            requests.append(request)
            category = re.search('super_class_id=([0-9]+)', row['text']).group(1)
            buckets[category].append(request)
    if len(requests) != 14000:
        raise ValueError('SOP generation count differs from contract')
    write_rows(output / 'private/staged_records.jsonl', sorted(records, key=lambda r: r['record_id']))
    write_rows(output / 'private/lineage.jsonl', lineage)
    write_rows(output / 'generation/requests.jsonl', sorted(requests, key=lambda r: r['id']))
    indexed = {r['record_id']: r for r in data}
    masks = [pair_mask(p, indexed) | {'record_id_a': remap[p['record_id_a']], 'record_id_b': remap[p['record_id_b']], 'partition': p['partition']} for p in pairs]
    write_rows(output / 'sealed/relation_applicability.jsonl', masks)
    dump(output / 'contract.json', {'revision_id': output.name, 'base_identity': 'vl_dedupbench_50k_release', 'records': 50000, 'pairs': 66788, 'source_root': str(source), 'generation_model': str(model), 'generation_prompt': PROMPT, 'seed': seed, 'generation_prompt_version': 'visible_sentence', 'generation_model_inputs': ['image_pixels', 'fixed_prompt'], 'decision_inputs': ['image_pixels', 'revised_text', 'computed_observable_features'], 'private_only': ['source_dataset', 'source_image_id', 'original_level', 'partition', 'visual_group_id', 'label_origin'], 'generated_caption_quality': 'programmatic_QC_and_limited_visual_audit; not_human_gold', 'relation_policy': 'preserve_controlled_relation_where_valid; generated_caption_claim_support_unknown'})
    receipt = {'records': len(records), 'unique_image_paths': len(paths), 'SOP_generation_requests': len(requests), 'text_revisions': dict(Counter((r['text_revision'] for r in lineage))), 'SOP_pair_grades': dict(Counter((m['grade'] for m in masks if 'SOP' in m['reason'])))}
    print(json.dumps(receipt), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    a = p.parse_args()
    prepare(a.source, a.output, a.model)
