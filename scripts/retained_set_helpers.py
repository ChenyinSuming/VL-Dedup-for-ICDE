import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import itertools
import json
from pathlib import Path
import random
ROOT = REPO_ROOT
REPORT = ROOT / str(REPORTS / 'retained_set')
RUN = RESOURCES / 'runs/retained_set'
AXES = ['text_coverage', 'image_coverage', 'association_coverage', 'text_only_in_retained_image']
LABEL_MAP = {'single': 'covered_single', 'joint': 'covered_joint', 'lost': 'lost', 'uncertain': 'uncertain'}

def utc():
    return datetime.now(timezone.utc).isoformat()

def readl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def dump(path, value, frozen=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if frozen and path.exists():
        if path.read_text() != text:
            raise ValueError(f'Frozen artifact differs: {path}')
        return
    path.write_text(text)

def writel(path, rows, frozen=False):
    text = ''.join((json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
    path = Path(path)
    if frozen and path.exists():
        if path.read_text() != text:
            raise ValueError(f'Frozen artifact differs: {path}')
        return
    path.write_text(text)

def protocol():
    return json.loads((REPORT / 'settings.json').read_text())

def coverage_module():
    spec = importlib.util.spec_from_file_location('coverage_audit', REPO_ROOT / 'scripts/coverage_audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sample_clusters(row_ids, components, partitions, scores):
    import numpy as np
    groups = defaultdict(list)
    assert len(row_ids) == len(components) == len(set(row_ids))
    for (rid, label) in zip(row_ids, components):
        groups[int(label)].append(rid)
    population = [sorted(g) for g in groups.values() if len(g) >= 2 and all((partitions[r] == 'test' for r in g))]
    rng = np.random.Generator(np.random.PCG64(20_261_029))
    (sample, strata) = ([], [])
    for (lo, hi, target) in [(2, 4, 10), (5, 5, 25), (6, 8, 15)]:
        pool = sorted((g for g in population if lo <= len(g) <= hi), key=lambda g: g[0])
        selected = sorted(rng.choice(len(pool), size=min(target, len(pool)), replace=False))
        for index in selected:
            ids = pool[index]
            assert all((np.isfinite(scores[r]) for r in ids))
            representative = min(ids, key=lambda r: (-scores[r], r))
            sample.append({'cluster_id': ids[0], 'record_ids': ids, 'size': len(ids), 'stratum': f'{lo}-{hi}', 'representative_id': representative, 'clip_scores': {r: float(scores[r]) for r in ids}})
        strata.append({'min_size': lo, 'max_size': hi, 'requested': target, 'available': len(pool), 'sampled': len(selected)})
    return (sample, strata)

def export_cluster(ids, representative, judgments):
    ids = set(ids)
    assert representative in ids
    expected = set(itertools.combinations(sorted(ids), 2))
    assert set(judgments) == expected, 'Missing or extraneous cluster pair judgments'
    kept = {representative}
    while True:
        additions = {r for r in ids - kept if any((judgments[tuple(sorted((r, k)))] == 'keep_augment' for k in kept))}
        if not additions:
            break
        kept.update(additions)
    held = {r for r in ids - kept if any((judgments[tuple(sorted((r, other)))] == 'mismatch' for other in ids - {r}))}
    result = {'a': {'retained': [representative], 'held': [], 'deleted': sorted(ids - {representative})}, 'b': {'retained': sorted(kept), 'held': sorted(held), 'deleted': sorted(ids - kept - held)}}
    for arm in result.values():
        parts = [set(arm[k]) for k in ['retained', 'held', 'deleted']]
        assert set.union(*parts) == ids and sum(map(len, parts)) == len(ids)
    assert set(result['b']['deleted']) <= set(result['a']['deleted'])
    return result

def audit_cells(exports):
    (cells, canonical) = ([], {})
    for cluster in exports:
        for strategy in ['a', 'b']:
            arm = f"{cluster['cluster_id']}:{strategy}"
            retained = cluster['strategies'][strategy]['retained']
            for source in cluster['strategies'][strategy]['deleted']:
                key = (source, tuple(sorted(retained)))
                cid = f"cell-{cluster['cluster_id']}-{strategy}-{source}"
                reuse = canonical.get(key)
                canonical.setdefault(key, cid)
                cells.append({'cell_id': cid, 'cluster_id': cluster['cluster_id'], 'strategy': strategy, 'arm': arm, 'source_id': source, 'retained_ids': retained, 'reuse_of': reuse, 'evidence_route': 'dual_ai'})
    return cells

def prepare():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    p = protocol()['inputs']
    RUN.mkdir(parents=True, exist_ok=True)
    ids = pq.read_table(p['component_row_ids'], columns=['record_id']).column(0).to_pylist()
    labels = np.load(p['components'], mmap_mode='r')
    partitions = {}
    with open(p['lineage']) as handle:
        for line in handle:
            row = json.loads(line)
            partitions[row['record_id']] = row['partition']
    score_array = np.load(p['clip_scores'], mmap_mode='r')
    meta = pq.read_table(p['clip_metadata'], columns=['record_id', 'row_index']).to_pylist()
    assert len(meta) == len({r['record_id'] for r in meta})
    scores = {r['record_id']: float(score_array[r['row_index']]) for r in meta}
    (sample, strata) = sample_clusters(ids, labels, partitions, scores)
    selected = {rid for c in sample for rid in c['record_ids']}
    records = []
    with open(p['public_records']) as handle:
        for line in handle:
            row = json.loads(line)
            if row['record_id'] in selected:
                for image in row['images']:
                    image['resolved_image_path'] = image['image_path']
                    assert Path(image['image_path']).is_file()
                records.append(row)
    records.sort(key=lambda r: r['record_id'])
    assert {r['record_id'] for r in records} == selected
    order = sorted(selected)
    random.Random(20_261_029).shuffle(order)
    frozen = {'records': records, 'opaque_ids': {rid: f'R{i + 1:03d}' for (i, rid) in enumerate(order)}}
    for destination in [RUN, REPORT]:
        writel(destination / 'sample_clusters.jsonl', sample)
        dump(destination / 'sample_summary.json', {'strata': strata, 'clusters': len(sample), 'records': len(selected), 'selected_cluster_pairs': sum((c['size'] * (c['size'] - 1) // 2 for c in sample))})
    dump(RUN / 'sample_records.json', frozen)
    (RUN / 'prompt.txt').write_text(coverage_module().GUIDE)

def export(verdict_rows):
    sample = readl(RUN / 'sample_clusters.jsonl')
    relevant = {pair for c in sample for pair in itertools.combinations(c['record_ids'], 2)}
    judgments = {}
    for row in verdict_rows:
        pair = tuple(sorted(row['record_ids']))
        if pair not in relevant:
            continue
        action = row['final_judgment']
        assert isinstance(action, str) and action
        if pair in judgments:
            assert judgments[pair] == action, f'Conflicting final actions: {pair}'
        judgments[pair] = action
    assert set(judgments) == relevant, f'Selected cluster pairs missing: {len(relevant - set(judgments))}'
    exports = []
    for c in sample:
        subset = {pair: judgments[pair] for pair in itertools.combinations(c['record_ids'], 2)}
        exports.append({**c, 'strategies': export_cluster(c['record_ids'], c['representative_id'], subset)})
    cells = audit_cells(exports)
    frozen = json.loads((RUN / 'sample_records.json').read_text())
    frozen['retained_ids'] = {f"{c['cluster_id']}:{s}": c['strategies'][s]['retained'] for c in exports for s in ['a', 'b']}
    frozen['cells'] = cells
    dump(RUN / 'frozen_inputs.json', frozen)
    for destination in [RUN, REPORT]:
        writel(destination / 'exports.jsonl', exports)
        writel(destination / 'audit_cells.jsonl', cells)
        for s in ['a', 'b']:
            for state in ['retained', 'held', 'deleted']:
                writel(destination / f'{s}_{state}.jsonl', [{'cluster_id': c['cluster_id'], 'record_id': rid} for c in exports for rid in c['strategies'][s][state]])
        dump(destination / 'export_summary.json', {'clusters': len(exports), 'records': sum((c['size'] for c in exports)), 'strategies': {s: {k: sum((len(c['strategies'][s][k]) for c in exports)) for k in ['retained', 'held', 'deleted']} for s in ['a', 'b']}, 'deleted_cells': len(cells), 'unique_audit_queries': sum((c['reuse_of'] is None for c in cells)), 'reused_cells': sum((c['reuse_of'] is not None for c in cells)), 'selected_pair_judgments': dict(Counter(judgments.values()))})

def one_audit(auditor, cell, frozen):
    """Ask one AI auditor whether the retained set covers the information of one deleted record."""
    import json as _json
    import mimetypes
    from scripts import ai_audit
    coverage = coverage_module()
    (messages, provenance) = coverage.make_messages(frozen, [cell], cell['arm'])
    image_paths = iter(provenance['image_index'].values())
    for item in messages[1]['content']:
        if item['type'] == 'image_url':
            mime = mimetypes.guess_type(next(image_paths))[0] or 'image/jpeg'
            item['image_url']['url'] = item['image_url']['url'].replace('data:image/jpeg;', f'data:{mime};', 1)
    response = ai_audit.chat(auditor, messages, 18000, token_field='max_completion_tokens')
    (_, parsed) = coverage.parse_response(_json.dumps(response), [cell], frozen, cell['arm'])
    answer = parsed[0]['audit']
    answer['label'] = LABEL_MAP[answer['label']]
    return {**cell, 'auditor': auditor, 'model': ai_audit.AUDITOR_MODELS[auditor], 'status': 'completed', 'audit': answer}

def substantive(a):
    return (a['label'], *(a[k] for k in AXES), tuple(sorted(set(a['coverage_types']))), tuple(sorted(set(a['loss_types']))), tuple(sorted((w['record_id'] for w in a['witnesses']))))

def compare(cell, first, second):
    reasons = []
    valid = [r for r in [first, second] if r['status'] == 'completed']
    same = len(valid) == 2 and substantive(first['audit']) == substantive(second['audit'])
    if len(valid) != 2:
        reasons.append('terminal_failure_or_no_output')
    elif not same:
        reasons.append('substantive_disagreement')
    if any((r['audit']['label'] == 'uncertain' or any((r['audit'][k] == 'uncertain' for k in AXES)) for r in valid)):
        reasons.append('any_uncertain_label_axis_or_cross_modal_flag')
    return {**cell, 'audit_1': first, 'audit_2': second, 'structured_agreement': same, 'needs_adjudication': bool(reasons), 'coverage_label': first['audit']['label'] if not reasons else None}

def project_stage_e_clusters():
    source = RESOURCES / 'runs/stage_e_clusters'
    pairs = {r['pair_id']: r for r in readl(source / 'U_test.jsonl')}
    selected = {pair for c in readl(RUN / 'sample_clusters.jsonl') for pair in itertools.combinations(c['record_ids'], 2)}
    projection = []
    for row in readl(source / 'phase1b_verdicts.jsonl'):
        endpoints = pairs[row['pair_id']]
        pair = tuple(sorted((endpoints['record_id_a'], endpoints['record_id_b'])))
        if pair not in selected:
            continue
        parsed = row.get('parsed') or {}
        decision = parsed.get('decision', 'uncertain') if row['status'] == 'completed' else 'uncertain'
        assert decision in {'duplicate', 'keep_augment', 'mismatch', 'non_duplicate', 'uncertain'}
        projection.append(dict(record_ids=list(pair), final_judgment=decision, pair_id=row['pair_id']))
    assert {tuple(r['record_ids']) for r in projection} == selected
    writel(RUN / 'stage_e_clusters_selected_projection.jsonl', projection)
    return projection

def summarize():
    frozen = json.loads((RUN / 'frozen_inputs.json').read_text())
    audits = [{r['cell_id']: r for r in readl(RUN / f'audit_{i}.jsonl')} for i in [1, 2]]
    expected = {c['cell_id'] for c in frozen['cells']}
    assert all((set(a) == expected for a in audits))
    results = [compare(c, audits[0][c['cell_id']], audits[1][c['cell_id']]) for c in frozen['cells']]
    summary = json.loads((RUN / 'export_summary.json').read_text())
    summary.update(ai_agreement_cells=sum((not r['needs_adjudication'] for r in results)), by_strategy={s: dict(ai_agreement=sum((r['strategy'] == s and (not r['needs_adjudication']) for r in results)), unresolved=sum((r['strategy'] == s and r['needs_adjudication'] for r in results))) for s in ['a', 'b']})
    writel(RUN / 'agreement.jsonl', results)
    dump(REPORT / 'results.json', summary)
    return summary
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'export', 'summarize'])
    a = p.parse_args()
    if a.action == 'export':
        export(project_stage_e_clusters())
    else:
        globals()[a.action]()
