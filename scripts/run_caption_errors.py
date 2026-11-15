import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import sys
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
import numpy as np
RES = RESOURCES
RUN = RES / 'runs/caption_errors'
REPORT = ROOT / str(REPORTS / 'caption_errors')
caption_actions = RES / 'runs/caption_actions'
DATA = RES / 'datasets/benchmark'
CACHE = RES / 'runs/pair_features/features/clip'
T = 0.14858555048704147
CLASSES = ('color', 'number', 'object')
DECISIONS = ('duplicate', 'non_duplicate', 'keep_augment', 'mismatch', 'uncertain')
COLORS = 'red blue green yellow white black brown orange pink purple gray grey'.split()
NUMBERS = 'two three four five six seven eight nine ten'.split()
OBJECTS = dict(animal='dog cat horse cow sheep elephant bear zebra giraffe bird'.split(), vehicle='car bus train truck motorcycle bicycle airplane boat'.split(), food='pizza cake sandwich banana apple donut'.split())
PLURALS = {word: word + 's' for words in OBJECTS.values() for word in words}
PLURALS.update(sheep='sheep', bus='buses', sandwich='sandwiches')
FORMS = {form: (word, group, plural) for (group, words) in OBJECTS.items() for word in words for (plural, form) in ((True, PLURALS[word]), (False, word))}
VOCAB = dict(color=COLORS, number=NUMBERS + list(map(str, range(2, 11))), object=sorted(FORMS))
PATTERNS = {kind: re.compile('\\b(?:' + '|'.join(sorted(words, key=len, reverse=True)) + ')\\b', re.I) for (kind, words) in VOCAB.items()}

def stamp():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    return json.loads(Path(path).read_text())

def dump(path, value, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)

def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')

def partition_rows(path, partition):
    pattern = re.compile('"partition"\\s*:\\s*"' + re.escape(partition) + '"')
    with Path(path).open() as stream:
        for line in stream:
            if pattern.search(line):
                row = json.loads(line)
                assert row['partition'] == partition
                yield row

def partition_mapping(path):
    record_pattern = re.compile('"record_id"\\s*:\\s*"([^"]+)"')
    partition_pattern = re.compile('"partition"\\s*:\\s*"([^"]+)"')
    result = {}
    with Path(path).open() as stream:
        for line in stream:
            rid = record_pattern.search(line).group(1)
            partition = partition_pattern.search(line).group(1)
            assert rid not in result
            result[rid] = partition
    return result

def case_like(source, replacement):
    if source.isupper():
        return replacement.upper()
    if source.istitle():
        return replacement.title()
    if source.islower() or source.isdigit():
        return replacement
    return ''.join((c.upper() if source[min(i, len(source) - 1)].isupper() else c.lower() for (i, c) in enumerate(replacement)))

def replacement_choices(kind, word, caption='', start=0):
    word = word.lower()
    if kind == 'color':
        source = 'gray' if word == 'grey' else word
        return [c for c in COLORS if c not in (source, 'grey')]
    if kind == 'number':
        value = int(word) if word.isdigit() else NUMBERS.index(word) + 2
        return [str(n) if word.isdigit() else NUMBERS[n - 2] for n in range(2, 11) if n != value]
    (lemma, group, plural) = FORMS[word]
    if lemma == 'sheep':
        before = caption[:start].lower()
        if re.search('\\b(a|an|one|this|that|lone|single)(?:\\s+small)?\\s+$', before):
            plural = False
        elif re.search('\\b(two|three|four|five|six|seven|eight|nine|ten|[2-9]|10|many|several|these|those|some)\\s+$', before) or re.search('\\b(herd|flock|group)\\s+of\\s+$', before):
            plural = True
        else:
            raise ValueError('Selected sheep has ambiguous grammatical number')
    return [PLURALS[w] if plural else w for (g, words) in OBJECTS.items() if g != group for w in words]

def edit_caption(caption, kind, rng):
    matches = list(PATTERNS[kind].finditer(caption))
    if len(matches) != 1:
        raise ValueError('Caption must contain exactly one class-word occurrence')
    match = matches[0]
    choices = replacement_choices(kind, match.group(), caption, match.start())
    replacement = case_like(match.group(), str(rng.choice(choices)))
    new = caption[:match.start()] + replacement + caption[match.end():]
    assert new != caption and new[:match.start()] == caption[:match.start()]
    assert new[match.start() + len(replacement):] == caption[match.end():]
    return (new, dict(initial_word=match.group(), new_word=replacement, start=match.start(), end=match.end()))

def sample_sources(records, lineage, kind, rng, count=100):
    candidates = sorted((r for r in lineage if len(list(PATTERNS[kind].finditer(records[r['record_id']]['text']))) == 1), key=lambda r: r['record_id'])
    groups = {r['visual_group_id'] for r in candidates}
    counts = dict(candidates=len(candidates), visual_groups=len(groups), required=count)
    if len(groups) < count:
        raise ValueError(f'{kind}: insufficient eligible visual groups: {counts}')
    (selected, seen) = ([], set())
    for index in rng.permutation(len(candidates)):
        source = candidates[index]
        if source['visual_group_id'] not in seen:
            selected.append(source)
            seen.add(source['visual_group_id'])
            if len(selected) == count:
                break
    assert len(selected) == len(seen) == count
    return (selected, counts)

def construct(partition):
    if partition == 'test':
        pass
    records = {r['record_id']: r for r in rows(DATA / 'public_records.jsonl')}
    lineage = [r for r in partition_rows(DATA / 'private/lineage.jsonl', partition) if r['source_dataset'] == 'coco2017' and r['original_level'] != 'M']
    rng = np.random.Generator(np.random.PCG64({'validation': 20_261_042, 'test': 20_261_043}[partition]))
    (counts, result) = ({}, [])
    for kind in CLASSES:
        (selected, counts[kind]) = sample_sources(records, lineage, kind, rng)
        for source in selected:
            rid = source['record_id']
            record = records[rid]
            (new, edit) = edit_caption(record['text'], kind, rng)
            pid = f'caption_errors__{partition}__{kind}__{rid}'
            result.append(dict(pair_id=pid, partition=partition, error_class=kind, source_record_id=rid, new_record_id=pid + '__edited', visual_group_id=source['visual_group_id'], image=record['images'][0]['image_path'], source_caption=record['text'], new_caption=new, edit=edit))
    write_rows(RUN / f'{partition}_synthetic.jsonl', result)
    dump(RUN / f'{partition}_construction.json', dict(eligibility=counts, pairs=len(result)))

def metrics(tp, fp, fn, tn):
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, P=tp / (tp + fp) if tp + fp else None, R=tp / (tp + fn) if tp + fn else None, F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)

def rule_counts(population):
    result = {}
    for rule in ('R_s', 'R_sv', 'VLM_only'):
        tp = fp = fn = tn = 0
        for row in population:
            pred = row['score'] < T if rule == 'R_s' else row['decision'] == 'mismatch' if rule == 'VLM_only' else row['score'] < T or row['decision'] == 'mismatch'
            positive = row['group'] == 'M'
            tp += int(pred and positive)
            fp += int(pred and (not positive))
            fn += int(not pred and positive)
            tn += int(not pred and (not positive))
        result[rule] = metrics(tp, fp, fn, tn)
        assert tp + fp + fn + tn == len(population)
    extra = [r for r in population if r['score'] >= T and r['decision'] == 'mismatch']
    result['extra'] = dict(TP=sum((r['group'] == 'M' for r in extra)), FP={g: sum((r['group'] == g for r in extra)) for g in ('L0', 'L2', 'L3')})
    assert result['R_sv']['TP'] == result['R_s']['TP'] + result['extra']['TP']
    assert result['R_sv']['FP'] == result['R_s']['FP'] + sum(result['extra']['FP'].values())
    return result

def score_mapping():
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    mapping = pq.read_table(CACHE / 'metadata.parquet', columns=['record_id', 'row_index']).to_pydict()
    assert len(mapping['record_id']) == len(set(mapping['record_id'])) == 50000
    assert sorted(mapping['row_index']) == list(range(50000))
    values = np.load(CACHE / 'clipscore.npy', mmap_mode='r')
    positions = dict(zip(mapping['record_id'], mapping['row_index'], strict=True))
    return (positions, {rid: float(values[i]) for (rid, i) in positions.items()})

def encode_captions(partition):
    import torch
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    cfg = read_json(REPORT / 'settings.json')['synthetic']['clip']
    (positions, scores) = score_mapping()
    image = np.load(CACHE / 'image_emb.npy', mmap_mode='r')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], local_files_only=True)
    model = CLIPTextModelWithProjection.from_pretrained(cfg['model'], local_files_only=True).to('cuda').eval()

    def encode(texts, source_ids):
        output = []
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                batch = tokenizer(texts[start:start + 32], padding=True, truncation=True, max_length=77, return_tensors='pt').to('cuda')
                vector = torch.nn.functional.normalize(model(**batch).text_embeds.float(), dim=1).cpu().numpy()
                cached_image = image[[positions[rid] for rid in source_ids[start:start + 32]]]
                output.extend(np.sum(cached_image * vector, axis=1).tolist())
        return output
    synthetic = list(rows(RUN / f'{partition}_synthetic.jsonl'))
    computed = encode([r['new_caption'] for r in synthetic], [r['source_record_id'] for r in synthetic])
    output = [dict(pair_id=r['pair_id'], new_clipscore=float(v), source_clipscore=scores[r['source_record_id']], pair_score=min(float(v), scores[r['source_record_id']])) for (r, v) in zip(synthetic, computed, strict=True)]
    assert len(output) == 300 and all((np.isfinite(r['pair_score']) for r in output))
    write_rows(RUN / f'{partition}_synthetic_scores.jsonl', output)
    torch.cuda.synchronize()

def synthetic_summary(partition, threshold=T):
    samples = list(rows(RUN / f'{partition}_synthetic.jsonl'))
    scores = {r['pair_id']: r for r in rows(RUN / f'{partition}_synthetic_scores.jsonl')}
    predictions = {r['pair_id']: r for r in rows(RUN / f'{partition}_synthetic_verdicts.jsonl')}
    assert set(scores) == set(predictions) == {r['pair_id'] for r in samples}
    assert len(samples) == 300
    result = {}
    for kind in CLASSES:
        selected = [r for r in samples if r['error_class'] == kind]
        count = len(selected)
        assert count == len({r['visual_group_id'] for r in selected}) == 100
        low = mismatch = union = extra = 0
        distribution = Counter()
        for row in selected:
            pid = row['pair_id']
            decision = predictions[pid]['parsed']['decision']
            distribution[decision] += 1
            score_low = scores[pid]['pair_score'] < threshold
            low += int(score_low)
            mismatch += int(decision == 'mismatch')
            union += int(score_low or decision == 'mismatch')
            extra += int(not score_low and decision == 'mismatch')
        assert union == low + extra and sum(distribution.values()) == count
        result[kind] = dict(pairs=count, score_below_threshold=low, vlm_decisions={d: distribution[d] for d in DECISIONS}, vlm_mismatch=mismatch, R_s=dict(detected=low, rate=low / count), R_sv=dict(detected=union, rate=union / count), extra_detected=extra)
    return result

def largest_remainder(population, total=60):
    groups = ('L0', 'L2', 'L3')
    denominator = sum((population.get(g, 0) for g in groups))
    if denominator < total:
        raise ValueError('Natural audit population has fewer pairs than required sample')
    (allocated, remainders) = ({}, {})
    for group in groups:
        (allocated[group], remainders[group]) = divmod(total * population.get(group, 0), denominator)
    order = sorted(groups, key=lambda g: (-remainders[g], groups.index(g)))
    for group in order[:total - sum(allocated.values())]:
        allocated[group] += 1
    assert sum(allocated.values()) == total
    return allocated

def evaluate(partition):
    threshold = float(read_json(caption_actions / 'threshold_frozen.json')['t_new'])
    summary = synthetic_summary(partition, threshold)
    result = dict(partition=partition, threshold=threshold, synthetic=summary, scores_path=str(RUN / f'{partition}_synthetic_scores.jsonl'), verdicts_path=str(RUN / f'{partition}_synthetic_verdicts.jsonl'))
    dump(REPORT / f'synthetic_results_{partition}.json', result)
    return result
AUDIT_LABELS = ('correct', 'local_error', 'whole_mismatch', 'unsure')
ERROR_TYPES = ('color', 'count', 'object', 'action', 'location', 'text', 'other')
AUDIT_GUIDE = 'Judge each caption independently against its own image. Do not infer image support from the other caption.\ncorrect: no statement contradicts the image. A short or incomplete caption can be correct.\nlocal_error: one or more specific statements contradict the image, such as color, count, object, action, location or text.\nwhole_mismatch: the caption overall describes a different image.\nunsure: the image does not allow a reliable judgment. An unverified detail alone is not proof of contradiction.\nFor local_error quote the exact contradicted words from that caption and list types from color/count/object/action/location/text/other.\nFor each caption provide a short image-grounded explanation. Use only these labels. Return exactly one JSON object:\n{"blind_id":"N...", "a":{"label":"...","wrong_words":[],"error_types":[],"reason":"..."},\n"b":{"label":"...","wrong_words":[],"error_types":[],"reason":"..."}}.\nFor correct and unsure use empty wrong_words and error_types. For whole_mismatch use empty error_types and explain the overall mismatch in reason.'

def natural_sample():
    (all_pairs, authority) = caption_action_population()
    public = {r['record_id']: r for r in rows(DATA / 'public_records.jsonl')}
    cfg = read_json(REPORT / 'settings.json')['natural']
    (selected, sizes) = ([], {})
    audit_sizes = {g: sum((r['group'] == g and r['decision'] == 'mismatch' for r in all_pairs)) for g in ('L0', 'L2', 'L3')}
    allocation = largest_remainder(audit_sizes)
    for arm in ('audit', 'control'):
        rng = np.random.Generator(np.random.PCG64(cfg['seeds'][arm]))
        sizes[arm] = {}
        for (group, count) in allocation.items():
            candidates = sorted((r for r in all_pairs if r['group'] == group and (r['decision'] == 'mismatch') == (arm == 'audit')), key=lambda r: r['pair_id'])
            sizes[arm][group] = len(candidates)
            if len(candidates) < count:
                raise ValueError('Natural audit stratum shortage: ' + str((arm, group, len(candidates), count)))
            selected.extend((dict(candidates[int(i)], arm=arm) for i in rng.choice(len(candidates), count, replace=False)))
    assert len(selected) == len({r['pair_id'] for r in selected}) == 120
    rng = np.random.Generator(np.random.PCG64(cfg['seeds']['blind_shuffle']))
    (blinded, private) = ([], [])
    media = RUN / 'natural' / 'media'
    media.mkdir(parents=True, exist_ok=True)
    for (ordinal, index) in enumerate(rng.permutation(len(selected)), 1):
        original = selected[int(index)]
        blind_id = f'N{ordinal:03d}'
        row = dict(blind_id=blind_id)
        for side in ('a', 'b'):
            record = public[original['record_id_' + side]]
            source = Path(record['images'][0]['image_path'])
            filename = blind_id + '_' + side + source.suffix.lower()
            shutil.copyfile(source, media / filename)
            row[side] = dict(caption=record['text'], image='media/' + filename)
        blinded.append(row)
        private.append(dict(blind_id=blind_id, **original))
    write_rows(RUN / 'natural' / 'blind_pairs.jsonl', blinded)
    write_rows(RUN / 'natural' / 'private_key.jsonl', private)
    sample_summary = dict(arms=sizes, sampled_per_arm=allocation, audit_population_pairs=sum(audit_sizes.values()), total=120, seeds=cfg['seeds'], sampled_utc=stamp(), authority=authority)
    dump(RUN / 'natural' / 'sample_summary.json', sample_summary)
    dump(REPORT / 'natural_sample_summary.json', sample_summary)

def parse_audit(raw, row):
    from scripts.stage_e import extract_json_object
    envelope = json.loads(raw)
    content = envelope['choices'][0]['message']['content']
    answer = extract_json_object(content)
    if not isinstance(answer, dict) or answer.get('blind_id') != row['blind_id']:
        raise ValueError('Audit blind identity mismatch')
    for side in ('a', 'b'):
        item = answer.get(side)
        if not isinstance(item, dict) or item.get('label') not in AUDIT_LABELS:
            raise ValueError('Invalid audit label')
        for field in ('wrong_words', 'error_types'):
            if not isinstance(item.get(field), list) or not all((isinstance(v, str) for v in item[field])):
                raise ValueError('Invalid audit ' + field)
        if not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError('Missing image-grounded reason')
        if not set(item['error_types']) <= set(ERROR_TYPES):
            raise ValueError('Unknown error type')
        if item['label'] == 'local_error':
            if not item['wrong_words'] or not item['error_types']:
                raise ValueError('Local error requires exact wrong words and types')
            if not all((v and v in row[side]['caption'] for v in item['wrong_words'])):
                raise ValueError('Wrong words are not exact stored caption text')
        elif item['error_types']:
            raise ValueError('Error types apply only to local_error')
        if item['label'] in ('correct', 'unsure') and item['wrong_words']:
            raise ValueError('Unexpected wrong words')
    return (answer, envelope)

def audit_messages(row):
    content = [dict(type='text', text='blind_id: ' + row['blind_id'])]
    for side in ('a', 'b'):
        source = RUN / 'natural' / row[side]['image']
        mime = mimetypes.guess_type(source.name)[0] or 'image/jpeg'
        data = base64.b64encode(source.read_bytes()).decode('ascii')
        content.extend([dict(type='text', text=f'Caption {side}: ' + row[side]['caption']), dict(type='image_url', image_url=dict(url=f'data:{mime};base64,{data}'))])
    return [dict(role='system', content=AUDIT_GUIDE), dict(role='user', content=content)]

def audit():
    """Ask both AI auditors to label each caption of the 120 sampled pairs."""
    from scripts import ai_audit
    limit = read_json(REPORT / 'settings.json')['natural']['api']['max_completion_tokens']
    for row in rows(RUN / 'natural' / 'blind_pairs.jsonl'):
        for auditor in (1, 2):
            terminal = RUN / 'natural' / 'requests' / f'auditor_{auditor}' / row['blind_id'] / 'terminal.json'
            if terminal.exists():
                continue
            response = ai_audit.chat(auditor, audit_messages(row), limit, token_field='max_completion_tokens')
            (answer, _) = parse_audit(json.dumps(response), row)
            dump(terminal, dict(status='completed', blind_id=row['blind_id'], auditor=auditor, model=ai_audit.AUDITOR_MODELS[auditor], answer=answer))

def natural_summary():
    sample = list(rows(RUN / 'natural' / 'blind_pairs.jsonl'))
    sample_summary = read_json(RUN / 'natural' / 'sample_summary.json')
    private = {r['blind_id']: r for r in rows(RUN / 'natural' / 'private_key.jsonl')}
    (summaries, unresolved, accepted) = ({}, [], [])
    for arm in ('audit', 'control'):
        summaries[arm] = dict(sampled=60, ai_consensus_pairs=0, ai_error_pairs=0, human_pending=0, human_reviewed=0, human_error_pairs=None, strata={}, caption_labels=Counter(), local_error_types_by_auditor={'1': Counter(), '2': Counter()}, pair_error_types_by_auditor={'1': Counter(), '2': Counter()})
    for row in sample:
        bid = row['blind_id']
        (arm, group) = (private[bid]['arm'], private[bid]['group'])
        summary = summaries[arm]
        stratum = summary['strata'].setdefault(group, dict(sampled=0, ai_consensus_pairs=0, ai_error_pairs=0, human_pending=0))
        stratum['sampled'] += 1
        answers = [read_json(RUN / 'natural' / 'requests' / f'auditor_{i}' / bid / 'terminal.json') for i in (1, 2)]
        agree = all((a['status'] == 'completed' for a in answers))
        if agree:
            agree = all((answers[0]['answer'][s]['label'] == answers[1]['answer'][s]['label'] != 'unsure' for s in ('a', 'b')))
        if not agree:
            unresolved.append(row)
            summary['human_pending'] += 1
            stratum['human_pending'] += 1
            continue
        error = any((answers[0]['answer'][s]['label'] in ('local_error', 'whole_mismatch') for s in ('a', 'b')))
        summary['ai_consensus_pairs'] += 1
        summary['ai_error_pairs'] += int(error)
        stratum['ai_consensus_pairs'] += 1
        stratum['ai_error_pairs'] += int(error)
        for side in ('a', 'b'):
            summary['caption_labels'][answers[0]['answer'][side]['label']] += 1
        for (i, answer) in enumerate(answers, 1):
            summary['local_error_types_by_auditor'][str(i)].update((t for side in ('a', 'b') for t in set(answer['answer'][side]['error_types'])))
            summary['pair_error_types_by_auditor'][str(i)].update({t for side in ('a', 'b') for t in answer['answer'][side]['error_types']})
        accepted.append(dict(blind_id=bid, pair_error=error, answers=answers))
    for summary in summaries.values():
        n = summary['ai_consensus_pairs']
        summary['ai_conditional_error_rate'] = summary['ai_error_pairs'] / n if n else None
        summary['full60_error_rate_bounds'] = [summary['ai_error_pairs'] / 60, (summary['ai_error_pairs'] + summary['human_pending']) / 60]
        assert n + summary['human_pending'] == 60
    (a, b) = (summaries['audit'], summaries['control'])
    difference = a['ai_conditional_error_rate'] - b['ai_conditional_error_rate'] if a['ai_consensus_pairs'] and b['ai_consensus_pairs'] else None
    result = dict(run_id=RUN.name, groups=summaries, ai_conditional_difference=difference, full_sample_difference=None, full_sample_difference_bounds=[a['full60_error_rate_bounds'][0] - b['full60_error_rate_bounds'][1], a['full60_error_rate_bounds'][1] - b['full60_error_rate_bounds'][0]], audit_population_pairs=sample_summary['audit_population_pairs'], population_by_group=sample_summary['arms'], sample_allocation=sample_summary['sampled_per_arm'], estimated_true_errors_in_population=None, blind_key=str(RUN / 'natural' / 'private_key.jsonl'), completed_utc=stamp())
    write_rows(RUN / 'natural' / 'ai_accepted.jsonl', accepted)
    dump(REPORT / 'natural_ai_results.json', result)

def caption_action_population():
    threshold_data = read_json(caption_actions / 'threshold_frozen.json')
    threshold = float(threshold_data.get('t_new', threshold_data['threshold']))
    source = caption_actions / 'actions/actions_test.jsonl'
    actions = list(rows(source))
    assert len({r['pair_id'] for r in actions}) == len(actions)
    parts = partition_mapping(DATA / 'private/lineage.jsonl')
    (_, scores) = score_mapping()
    references = {}
    for r in partition_rows(DATA / 'sealed/relation_applicability.jsonl', 'test'):
        identity = tuple(sorted((r['record_id_a'], r['record_id_b'])))
        assert identity not in references
        references[identity] = r['grade']
    population = []
    same = 0
    for r in actions:
        (a, b) = (r['record_id_a'], r['record_id_b'])
        assert parts[a] == parts[b] == r['partition'] == 'test'
        identity = tuple(sorted((a, b)))
        grade = references.get(identity)
        if grade not in ('L0', 'L2', 'L3'):
            continue
        same += 1
        score = min(scores[a], scores[b])
        if score < threshold:
            continue
        decision = r['parsed']['decision']
        assert decision in DECISIONS
        if r['status'] != 'completed' or not r['parsed']['contract_valid']:
            assert decision == 'uncertain'
        population.append(dict(pair_id=r['pair_id'], record_id_a=a, record_id_b=b, group=grade, score=score, decision=decision, source_file=r['source_file']))
    return (population, dict(judgment_source=str(source), U_test=len(actions), same_pairs=same, threshold=threshold, eligible_same_score_above=len(population)))

def infer_synthetic(partition):
    from scripts import run_stage_e_clusters as stage_e_clusters
    synthetic = list(rows(RUN / f'{partition}_synthetic.jsonl'))
    (records, pairs) = ({}, [])
    for row in synthetic:
        (a, b) = (row['new_record_id'], row['source_record_id'])
        records[a] = dict(record_id=a, text=row['new_caption'], images=[dict(image_path=row['image'])])
        records[b] = dict(record_id=b, text=row['source_caption'], images=[dict(image_path=row['image'])])
        pairs.append(dict(pair_id=row['pair_id'], record_id_a=a, record_id_b=b))
    config = read_json(REPORT / 'settings.json')['synthetic']['vlm']
    stage_e_clusters.infer_session(pairs, records, {'inference': config['inference']}, dict(max_num_seqs=16, submission_batch_size=16), partition, RUN / f'{partition}_synthetic_verdicts.jsonl')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['construct', 'encode', 'infer', 'evaluate', 'natural_sample', 'audit', 'natural_summary'])
    p.add_argument('--partition', choices=['validation', 'test'], default='validation')
    a = p.parse_args()
    if a.action == 'construct':
        construct(a.partition)
    elif a.action == 'encode':
        encode_captions(a.partition)
    elif a.action == 'infer':
        infer_synthetic(a.partition)
    elif a.action == 'evaluate':
        evaluate(a.partition)
    else:
        globals()[a.action]()
