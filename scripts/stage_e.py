import json
import math
import re
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
PROMPT = 'stage_e_pair_review'
DECISIONS = ('duplicate', 'non_duplicate', 'keep_augment', 'mismatch', 'uncertain')
TRISTATE = ('yes', 'no', 'uncertain')
PAIR_TYPES = ('equivalent_captions', 'complementary_captions', 'conflicting_records', 'different_entities', 'uncertain')
BLOCKERS = ('none', 'visual_difference', 'caption_unsupported', 'caption_conflict', 'different_entity', 'insufficient_visual_evidence', 'uncertain')
DECISION_OF_PAIR_TYPE = {'equivalent_captions': 'duplicate', 'complementary_captions': 'keep_augment', 'conflicting_records': 'mismatch', 'different_entities': 'non_duplicate'}
ROLE = 'Role: reviewer only. Your answer can block an unsafe merge, keep both captions, or confirm a duplicate; it does not add new merges. '
INTRO = 'You are reviewing two image-text training records for a vision-language deduplication pipeline. The first image is Record A. The second image is Record B. Use each image with its own caption, then compare A and B.'
SCHEMA = 'Return one compact JSON object only with keys `pair_type`, `same_visual_entity`, `caption_a_supported`, `caption_b_supported`, `dedup_blocker`, `extra_caption_value`, `confidence`, `visual_evidence_a`, `visual_evidence_b`, `visual_difference`, `caption_note`, and `reason`. Do not include a free `decision` key. `pair_type` must be equivalent_captions, complementary_captions, conflicting_records, different_entities, or uncertain. The three support keys and `extra_caption_value` must use yes, no, or uncertain. `dedup_blocker` must be none, visual_difference, caption_unsupported, caption_conflict, different_entity, insufficient_visual_evidence, or uncertain. '
RULES = 'Inspect pixels first, then classify the relation using one coarse `pair_type`. `pair_type=equivalent_captions` only when the images show the same visual entity, both captions are supported, and the captions preserve the same visible facts with no concrete information loss. `pair_type=complementary_captions` when the same visual entity has supported captions but either caption adds a concrete visible object, count, attribute, action, relation, setting, condition, location, viewpoint, scene role, relative position, distance, or motion detail that would be lost after merge. `pair_type=conflicting_records` when a caption is unsupported by its own image or the records conflict on a visible object, count, material, action, direction, terrain, location, state, or other dedup-relevant fact. `pair_type=different_entities` when the images do not show the same image or the same underlying visual entity. `pair_type=uncertain` when visual evidence is insufficient. Notes and structured keys must agree: if `visual_difference`, `caption_note`, or `reason` mentions additional details, different focus, extra context, or information loss, do not use `equivalent_captions`; use `complementary_captions` unless the details conflict, in which case use `conflicting_records`. `equivalent_captions` is required when the captions describe the same visible facts, even if they use different wording, word order, specificity, or fluency. If your notes say no significant difference, minor variation, same scene, same action, same essential information, or no additional details, do not use `complementary_captions`; use `equivalent_captions` with `extra_caption_value=no`. When the records share a core object, event or scene, do not use `conflicting_records` merely for crop, angle, focus, stage, proximity, or one caption naming a visible detail omitted by the other. If both captions can be true and one would lose concrete visible information after merge, use `complementary_captions`. Reserve `conflicting_records` for unsupported captions or mutually exclusive facts such as different material, count, terrain, action direction, object type, state, or location. Choose `complementary_captions` only with `same_visual_entity=yes`, `dedup_blocker=none`, and `extra_caption_value=yes`. Do not pair `complementary_captions` with `visual_difference` or `extra_caption_value=no`; that is internally inconsistent. When both records show the same image or the same event, angle, crop, focus, cooking stage, person/object presence, street context, vehicle proximity, close-up versus wide view, or one caption naming a visible object omitted by the other are additive caption utility, not `conflicting_records`. Use `equivalent_captions` only when removing one caption would lose no visible object, action, attribute, relation, setting, stage, position, or context. If the captions mention different supported visible facts and both can be true, preserve them with `complementary_captions`; if they cannot both be true, use `conflicting_records`.'

def clean_text(value: object, max_chars: int=600) -> str:
    if value is None:
        return ''
    if isinstance(value, dict):
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, (list, tuple, set)):
        raw = '; '.join((str(item) for item in value))
    else:
        try:
            if pd.isna(value):
                return ''
        except (TypeError, ValueError):
            pass
        raw = str(value)
    return ' '.join(raw.split())[:max_chars]

def caption_jaccard(text_a: object, text_b: object) -> float:
    a = set(re.findall('[a-z0-9]+', str(text_a or '').lower()))
    b = set(re.findall('[a-z0-9]+', str(text_b or '').lower()))
    return float(len(a & b) / len(a | b)) if a | b else 0.0

def identical_pixels(path_a: object, path_b: object) -> bool:
    (a, b) = (Path(str(path_a)), Path(str(path_b)))
    return a == b or a.read_bytes() == b.read_bytes()

def bool_value(row, column: str) -> bool:
    value = row.get(column)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {'1', 'true', 'yes'}

def numeric_value(row, columns: Iterable[str]) -> float:
    for column in columns:
        value = row.get(column)
        if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == '':
            continue
        parsed = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
        return float(parsed) if pd.notna(parsed) else np.nan
    return np.nan

def format_optional_float(value: object, digits: int=4) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 'unknown'
    return 'unknown' if math.isnan(parsed) else f'{parsed:.{digits}f}'

def hint_values(row) -> dict[str, object]:
    same_pixels = bool_value(row, 'identical_pixels')
    overlap = caption_jaccard(row.get('text_a_context'), row.get('text_b_context'))
    clip_min = numeric_value(row, ['clipscore_min'])
    channels = {c.strip() for c in str(row.get('channels') or '').split(';') if c.strip()}
    text_exact = 'text_exact' in channels or bool_value(row, 'text_exact_match')
    exact_text_low_clip = text_exact and (not same_pixels) and pd.notna(clip_min) and (float(clip_min) < 0.06)
    different_captions = same_pixels and overlap < 0.85
    return {'identical_pixels': same_pixels, 'caption_overlap': overlap, 'clipscore_min': clip_min, 'clipscore_abs_diff': numeric_value(row, ['clipscore_abs_diff']), 'text_exact': text_exact, 'likely_conflict': not same_pixels and overlap < 0.12 or exact_text_low_clip, 'identical_image_different_captions': different_captions, 'exact_text_low_clipscore': exact_text_low_clip, 'identical_image_low_caption_overlap': different_captions and overlap < 0.3}

def hint_text(row) -> str:
    h = hint_values(row)
    flags = [name for (name, on) in [('exact_text_low_clipscore', h['exact_text_low_clipscore']), ('identical_image_low_caption_overlap', h['identical_image_low_caption_overlap']), ('likely_caption_conflict', h['likely_conflict']), ('identical_image_different_captions', h['identical_image_different_captions'])] if on]
    return f"Observable features: identical_image_pixels={str(h['identical_pixels']).lower()}; caption_overlap={format_optional_float(h['caption_overlap'], digits=3)}; clipscore_min={format_optional_float(h['clipscore_min'])}; clipscore_abs_diff={format_optional_float(h['clipscore_abs_diff'])}; exact_text_match={str(h['text_exact']).lower()}; flags={';'.join(flags) or 'none'}."

def build_prompt_record(row, image_root: Path, fallback_root: Path | None=None) -> dict[str, object]:

    def resolve(raw):
        path = Path(str(raw))
        if path.is_absolute():
            return path
        rooted = image_root / path
        return rooted if rooted.exists() or fallback_root is None else fallback_root / path
    (image_a, image_b) = (resolve(row['image_path_a_context']), resolve(row['image_path_b_context']))
    return {'edge_id': row['edge_id'], 'record_id_a': row['record_id_a'], 'record_id_b': row['record_id_b'], 'image_path_a': str(image_a), 'image_path_b': str(image_b), 'caption_a': clean_text(row.get('text_a_context')), 'caption_b': clean_text(row.get('text_b_context')), 'hints': hint_text(row), 'prompt_version': PROMPT}

def build_text_prompt(record: dict[str, object]) -> str:
    return f"{ROLE}{INTRO}\n\nRecord A caption: {record['caption_a']}\nRecord B caption: {record['caption_b']}\n\n{clean_text(record['hints'], max_chars=400)}\n\n{SCHEMA}{RULES} Keep text fields under 12 words each."

def build_qwen_messages(record: dict[str, object]) -> list[dict[str, object]]:
    return [{'role': 'user', 'content': [{'type': 'image', 'image': str(record['image_path_a'])}, {'type': 'image', 'image': str(record['image_path_b'])}, {'type': 'text', 'text': build_text_prompt(record)}]}]

def repair_json_like_object(candidate: str) -> str:
    words = 'yes|no|none|uncertain|certain|maybe|unsure|high|medium|low|visual_difference|caption_unsupported|caption_conflict|different_entity|insufficient_visual_evidence|' + '|'.join(PAIR_TYPES[:-1])
    repaired = re.sub(f':\\s*({words})\\s*([,}}])', lambda m: f': "{m.group(1).lower()}"{m.group(2)}', candidate.replace('\\_', '_'), flags=re.IGNORECASE)
    return re.sub(':\\s*(True|False)\\s*([,}])', lambda m: f': {m.group(1).lower()}{m.group(2)}', repaired)

def extract_json_object(text: object) -> dict[str, object] | None:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return None
    raw = str(text).strip().replace('\\_', '_')
    candidates = []
    fence = re.search('```(?:json)?\\s*(\\{.*?\\})\\s*```', raw, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    (start, end) = (raw.find('{'), raw.rfind('}'))
    if 0 <= start < end:
        candidates.append(raw[start:end + 1])
    candidates.append(raw)
    for candidate in candidates:
        for attempt in dict.fromkeys([candidate, repair_json_like_object(candidate)]):
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
            break
    return None

def normalize(value: object, valid: Iterable[str], aliases: dict[str, str]) -> str:
    raw = clean_text(value).lower().replace('-', '_').replace(' ', '_')
    raw = aliases.get(raw, raw)
    return raw if raw in valid else 'uncertain'

def tristate(value: object) -> str:
    return normalize(value, TRISTATE, {'true': 'yes', 'false': 'no', 'same': 'yes', 'different': 'no', 'present': 'yes', 'absent': 'no', 'unknown': 'uncertain', 'unsure': 'uncertain', 'maybe': 'uncertain'})

def blocker(value: object) -> str:
    return normalize(value, BLOCKERS, {'': 'uncertain', 'no': 'none', 'no_blocker': 'none', 'same': 'none', 'visual': 'visual_difference', 'different_visual': 'visual_difference', 'unsupported_caption': 'caption_unsupported', 'unsupported': 'caption_unsupported', 'conflict': 'caption_conflict', 'non_duplicate': 'different_entity', 'insufficient_evidence': 'insufficient_visual_evidence', 'unknown': 'uncertain', 'unsure': 'uncertain'})

def pair_type(value: object) -> str:
    return normalize(value, PAIR_TYPES, {'equivalent': 'equivalent_captions', 'same': 'equivalent_captions', 'duplicate': 'equivalent_captions', 'complementary': 'complementary_captions', 'conflict': 'conflicting_records', 'conflicting': 'conflicting_records', 'mismatch': 'conflicting_records', 'different': 'different_entities', 'different_entity': 'different_entities', 'unknown': 'uncertain', 'unsure': 'uncertain'})

def parse_response(text: object) -> dict[str, object]:
    parsed = extract_json_object(text)
    if parsed is None:
        return {'parse_success': False, 'decision': 'uncertain', 'pair_type': 'uncertain', 'same_visual_entity': 'uncertain', 'caption_a_supported': 'uncertain', 'caption_b_supported': 'uncertain', 'dedup_blocker': 'uncertain', 'extra_caption_value': 'uncertain', 'confidence': 0.0, 'visual_evidence_a': '', 'visual_evidence_b': '', 'reason': ''}
    kind = pair_type(parsed.get('pair_type'))
    same = tristate(parsed.get('visual_entity_match'))
    same = tristate(parsed.get('same_visual_entity')) if same == 'uncertain' else same
    (support_a, support_b) = (tristate(parsed.get('caption_a_supported')), tristate(parsed.get('caption_b_supported')))
    (block, extra) = (blocker(parsed.get('dedup_blocker')), tristate(parsed.get('extra_caption_value')))
    if kind in ('equivalent_captions', 'complementary_captions'):
        block = 'none' if block == 'uncertain' else block
        if extra == 'uncertain':
            extra = 'no' if kind == 'equivalent_captions' else 'yes'
        (same, support_a, support_b) = ['yes' if v == 'uncertain' else v for v in (same, support_a, support_b)]
    elif kind == 'conflicting_records':
        block = 'caption_conflict' if block == 'uncertain' else block
    elif kind == 'different_entities':
        block = 'different_entity' if block == 'uncertain' else block
        same = 'no' if same == 'uncertain' else same
    try:
        confidence = float(parsed.get('confidence', 0.0))
    except (TypeError, ValueError):
        confidence = {'certain': 1.0, 'high': 0.9, 'medium': 0.6, 'low': 0.3}.get(str(parsed.get('confidence', '')).strip().lower(), 0.0)
    return {'parse_success': True, 'decision': DECISION_OF_PAIR_TYPE.get(kind, 'uncertain'), 'pair_type': kind, 'same_visual_entity': same, 'caption_a_supported': support_a, 'caption_b_supported': support_b, 'dedup_blocker': block, 'extra_caption_value': extra, 'confidence': float(np.clip(confidence, 0.0, 1.0)), 'visual_evidence_a': clean_text(parsed.get('visual_evidence_a'), max_chars=240), 'visual_evidence_b': clean_text(parsed.get('visual_evidence_b'), max_chars=240), 'reason': clean_text(parsed.get('reason', ''), max_chars=240)}

def has_visual_evidence(row) -> bool:
    texts = [clean_text(row.get('visual_evidence_a')).lower(), clean_text(row.get('visual_evidence_b')).lower()]
    generic = ['caption', 'marketplace listing', 'minor caption', 'same product', 'same product type', 'same visual entity', 'consistent captions', 'minor overlap']
    return all(texts) and (not any((p in t for t in texts for p in generic)))

def checked_decision(row) -> dict[str, object]:
    raw = row.get('decision') if row.get('decision') in DECISIONS else 'uncertain'
    out = lambda reason, final: {'vlm_raw_decision': raw, 'decision_check_applied': bool(reason), 'decision_check_reason': reason, 'decision': final}
    if raw in ('duplicate', 'keep_augment'):
        expected_extra = 'no' if raw == 'duplicate' else 'yes'
        expected_type = 'equivalent_captions' if raw == 'duplicate' else 'complementary_captions'
        complete = tristate(row.get('same_visual_entity')) == 'yes' and tristate(row.get('caption_a_supported')) == 'yes' and (tristate(row.get('caption_b_supported')) == 'yes') and (blocker(row.get('dedup_blocker')) == 'none') and (tristate(row.get('extra_caption_value')) == expected_extra) and (pair_type(row.get('pair_type')) == expected_type)
        if not complete:
            return out(f'{raw}_support_incomplete', 'uncertain')
        if not has_visual_evidence(row):
            return out('missing_visual_evidence', 'uncertain')
    if raw != 'duplicate':
        return out('', raw)
    hints = hint_values(row)
    if hints['exact_text_low_clipscore']:
        return out('exact_text_low_clipscore', 'mismatch')
    if hints['identical_image_low_caption_overlap']:
        return out('identical_image_low_caption_overlap', 'keep_augment')
    return out('', raw)
