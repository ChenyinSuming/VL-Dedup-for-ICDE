GUIDE = "Independently inspect the supplied full image(s) and caption A and caption B. Do NOT assume either caption is true. Use visible evidence, not world-knowledge guesses. Do not infer relations from writing style.\nJudge whether BOTH captions are supported by the image and EACH contains at least one concrete supported fact absent from the other caption. Different wording alone is not new information. Count objects, attributes, actions, locations and relations only when visible and actually asserted by that caption. Do not invent detail from the image that the caption does not say. A more general phrase entails its ordinary specific instance only in the specific-to-general direction.\nChoose exactly one label:\ncomplementary: both captions are supported AND A has a supported fact absent from B AND B has a supported fact absent from A. Quote both facts and explain their visible support.\nredundant: both captions are supported but they are equivalent OR one caption's information is contained in the other. A one-way extension is NOT mutual complementarity; set redundancy_kind=one_way_subsumption. Otherwise redundancy_kind=equivalent.\none_unsupported: one caption makes an important unsupported or contradicted claim while the other is supported. Identify which side and the exact claim. Minor uncertain detail alone should produce uncertain rather than a confident unsupported judgment.\nimage_text_mismatch: neither caption matches the image/item, or the supplied images depict different items such that they cannot support one shared item. Explain.\nuncertain: image detail, caption ambiguity or incomplete evidence prevents a confident decision. Do not force a label.\nReturn ONLY one JSON object with keys: label, caption_a_supported (yes/no/uncertain), caption_b_supported (yes/no/uncertain), a_unique_supported_facts (list of objects with caption_quote and visual_evidence), b_unique_supported_facts (same), redundancy_kind (equivalent/one_way_subsumption/not_applicable), unsupported_claims (list), explanation (brief Chinese explanation), confidence (high/medium/low)."
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import re, unicodedata
from pathlib import Path
import numpy as np
LABELS = ['complementary', 'redundant', 'one_unsupported', 'image_text_mismatch', 'uncertain']

def valid_vote(v):
    if v.get('label') not in LABELS:
        return False
    required = ['caption_a_supported', 'caption_b_supported', 'a_unique_supported_facts', 'b_unique_supported_facts', 'redundancy_kind', 'unsupported_claims', 'explanation', 'confidence']
    if any((k not in v for k in required)):
        return False
    if v['caption_a_supported'] not in ['yes', 'no', 'uncertain'] or v['caption_b_supported'] not in ['yes', 'no', 'uncertain']:
        return False
    if v['label'] in ['complementary', 'redundant'] and (v['caption_a_supported'] != 'yes' or v['caption_b_supported'] != 'yes'):
        return False
    if v['label'] == 'complementary':
        return all((isinstance(v[k], list) and v[k] and all((isinstance(f, dict) and f.get('caption_quote') and f.get('visual_evidence') for f in v[k])) for k in ['a_unique_supported_facts', 'b_unique_supported_facts']))
    return True

def evidence_issues(vote, row):
    support = [vote.get('caption_a_supported'), vote.get('caption_b_supported')]
    if vote.get('label') == 'redundant' and vote.get('redundancy_kind') not in ['equivalent', 'one_way_subsumption']:
        return ['redundant_requires_equivalent_or_one_way_subsumption']
    if vote.get('label') == 'one_unsupported' and sorted((str(x) for x in support)) != ['no', 'yes']:
        return ['one_unsupported_requires_one_yes_and_one_no']
    if vote.get('label') == 'image_text_mismatch' and row['image_a'] == row['image_b'] and (support != ['no', 'no']):
        return ['same_image_mismatch_requires_both_captions_unsupported']
    if vote.get('label') != 'complementary':
        return []

    def normalized(text):
        return re.sub('\\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip(' "\'""‘’')
    issues = []
    for side in ['a', 'b']:
        for fact in vote.get(f'{side}_unique_supported_facts', []):
            quote = fact.get('caption_quote')
            if not isinstance(quote, str) or not normalized(quote) or normalized(quote) not in normalized(row[f'caption_{side}']):
                issues.append(f'{side}_unique_fact_quote_not_in_caption')
    return sorted(set(issues))
REQUIRED = {'pair_type', 'same_visual_entity', 'caption_a_supported', 'caption_b_supported', 'dedup_blocker', 'extra_caption_value', 'confidence', 'visual_evidence_a', 'visual_evidence_b', 'visual_difference', 'caption_note', 'reason'}

def strict_parse(raw):
    from scripts import stage_e
    obj = stage_e.extract_json_object(raw)
    errors = []
    if not isinstance(obj, dict) or not REQUIRED.issubset(obj):
        errors.append('missing_schema_fields_or_json')
    else:
        tri = {'yes', 'no', 'uncertain'}
        for key in ('same_visual_entity', 'caption_a_supported', 'caption_b_supported', 'extra_caption_value'):
            if not isinstance(obj[key], str) or obj[key] not in tri:
                errors.append('invalid_' + key)
        pair = obj['pair_type']
        if not isinstance(pair, str) or pair not in set(stage_e.PAIR_TYPES):
            errors.append('invalid_pair_type')
        if not isinstance(obj['dedup_blocker'], str) or obj['dedup_blocker'] not in {'none', 'visual_difference', 'caption_unsupported', 'caption_conflict', 'different_entity', 'insufficient_visual_evidence', 'uncertain'}:
            errors.append('invalid_dedup_blocker')
        confidence = obj['confidence']
        if isinstance(confidence, str):
            valid_confidence = confidence.strip().lower() in {'certain', 'high', 'medium', 'low', 'uncertain'}
            if not valid_confidence:
                try:
                    numeric = float(confidence)
                    valid_confidence = np.isfinite(numeric) and 0 <= numeric <= 1
                except ValueError:
                    pass
        else:
            valid_confidence = isinstance(confidence, (int, float)) and (not isinstance(confidence, bool)) and np.isfinite(confidence) and (0 <= confidence <= 1)
        if not valid_confidence:
            errors.append('invalid_confidence')
        for key in ('visual_evidence_a', 'visual_evidence_b', 'visual_difference', 'caption_note', 'reason'):
            if not isinstance(obj[key], str):
                errors.append('invalid_' + key)
        if 'decision' in obj:
            errors.append('forbidden_free_decision')
        if errors:
            return {'parsed': None, 'contract_valid': False, 'contract_errors': errors, 'decision': 'uncertain'}
        (same, a, b, blocker, extra) = (obj[k] for k in ('same_visual_entity', 'caption_a_supported', 'caption_b_supported', 'dedup_blocker', 'extra_caption_value'))
        if pair in {'equivalent_captions', 'complementary_captions'}:
            expected_extra = 'no' if pair == 'equivalent_captions' else 'yes'
            if (same, a, b, blocker, extra) != ('yes', 'yes', 'yes', 'none', expected_extra):
                errors.append('inconsistent_positive_fields')
        if pair == 'different_entities' and (same == 'yes' or blocker == 'none'):
            errors.append('inconsistent_negative_fields')
        if pair == 'conflicting_records' and a == b == 'yes' and (blocker == 'none'):
            errors.append('inconsistent_mismatch_fields')
    parsed = stage_e.parse_response(raw)
    guard = stage_e.checked_decision(parsed)
    if guard['decision_check_applied']:
        errors.append(guard['decision_check_reason'])
    return {'parsed': parsed, 'contract_valid': not errors, 'contract_errors': errors, 'decision': guard['decision'] if not errors else 'uncertain'}
