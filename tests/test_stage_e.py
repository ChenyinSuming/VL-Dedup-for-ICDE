import json
from scripts import stage_e

def row(**extra):
    return dict(edge_id='e1', record_id_a='record_000001', record_id_b='record_000002', image_path_a_context='a.jpg', image_path_b_context='b.jpg', text_a_context='A dog runs on grass.', text_b_context='A brown dog runs past a red ball on grass.', **extra)

def answer(pair_type, **extra):
    base = dict(pair_type=pair_type, same_visual_entity='yes', caption_a_supported='yes', caption_b_supported='yes', dedup_blocker='none', extra_caption_value='no' if pair_type == 'equivalent_captions' else 'yes', confidence='high', visual_evidence_a='brown dog mid-stride', visual_evidence_b='red ball near dog', visual_difference='', caption_note='', reason='')
    return json.dumps({**base, **extra})

def test_prompt_shows_only_captions_and_observable_features(tmp_path):
    record = stage_e.build_prompt_record(row(identical_pixels=True), tmp_path)
    text = stage_e.build_text_prompt(record)
    assert 'record_000001' not in text and 'record_000002' not in text
    for word in ['L2', 'L3', 'gold', 'partition', 'visual_group', 'level', 'relation=']:
        assert word not in text
    assert 'identical_image_pixels=true' in text
    assert [c['type'] for c in stage_e.build_qwen_messages(record)[0]['content']] == ['image', 'image', 'text']

def test_identical_pixels_compares_bytes(tmp_path):
    (tmp_path / 'a.jpg').write_bytes(b'x')
    (tmp_path / 'b.jpg').write_bytes(b'x')
    (tmp_path / 'c.jpg').write_bytes(b'y')
    assert stage_e.identical_pixels(tmp_path / 'a.jpg', tmp_path / 'b.jpg')
    assert not stage_e.identical_pixels(tmp_path / 'a.jpg', tmp_path / 'c.jpg')

def test_decision_follows_pair_type():
    for (kind, decision) in stage_e.DECISION_OF_PAIR_TYPE.items():
        assert stage_e.parse_response(answer(kind))['decision'] == decision
    assert stage_e.parse_response('not json')['decision'] == 'uncertain'

def test_inconsistent_or_generic_answer_becomes_uncertain():
    parsed = stage_e.parse_response(answer('equivalent_captions', extra_caption_value='yes'))
    assert stage_e.checked_decision(parsed)['decision'] == 'uncertain'
    parsed = stage_e.parse_response(answer('equivalent_captions', visual_evidence_a='same product'))
    assert stage_e.checked_decision(parsed)['decision'] == 'uncertain'

def test_duplicate_on_identical_pixels_with_different_captions_keeps_both():
    parsed = stage_e.parse_response(answer('equivalent_captions'))
    fields = dict(text_a_context='A dog.', text_b_context='Two cats sleep on a sofa.')
    assert stage_e.checked_decision({**parsed, **fields, 'identical_pixels': True})['decision'] == 'keep_augment'
    assert stage_e.checked_decision({**parsed, **fields, 'identical_pixels': False})['decision'] == 'duplicate'
