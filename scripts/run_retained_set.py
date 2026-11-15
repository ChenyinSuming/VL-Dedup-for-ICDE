import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import REPO_ROOT, RESOURCES, REPORTS
import argparse
from collections import Counter, defaultdict
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
ROOT = REPO_ROOT
PARENT = ROOT / str(REPORTS / 'retained_set')
PROTOCOL = PARENT / 'settings.json'
REPORT = PARENT / 'final'
REFERENCE_RUN = RESOURCES / 'runs/retained_set'
RUN = REFERENCE_RUN / 'retained_set'
spec = importlib.util.spec_from_file_location('retained_set_helpers', ROOT / 'scripts/retained_set_helpers.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
(readl, dump, writel) = (base.readl, base.dump, base.writel)

def protocol():
    return json.loads(PROTOCOL.read_text())

def identity(cell):
    return (cell['source_id'], tuple(sorted(cell['retained_ids'])))

def sample_clusters(groups, partitions, scores, initial_sample, seed=20_261_046, strata=((2, 4, 10), (5, 5, 40))):
    import numpy as np
    preselected_sets = {frozenset(c['record_ids']) for c in initial_sample}
    eligible = [sorted(g) for g in groups if len(g) >= 2 and all((partitions[r] == 'test' for r in g))]
    rng = np.random.Generator(np.random.PCG64(seed))
    (selected, summary) = ([], [])
    for (lo, hi, target) in strata:
        pool = sorted((g for g in eligible if lo <= len(g) <= hi), key=lambda g: g[0])
        mandatory = [g for g in pool if frozenset(g) in preselected_sets]
        remaining = [g for g in pool if frozenset(g) not in preselected_sets]
        need = min(max(0, target - len(mandatory)), len(remaining))
        indices = rng.choice(len(remaining), size=need, replace=False)
        chosen = sorted(mandatory + [remaining[int(i)] for i in indices], key=lambda g: g[0])
        for ids in chosen:
            assert all((math.isfinite(scores[r]) for r in ids))
            selected.append({'cluster_id': ids[0], 'record_ids': ids, 'size': len(ids), 'stratum': f'{lo}-{hi}', 'preselected': frozenset(ids) in preselected_sets, 'representative_id': min(ids, key=lambda r: (-scores[r], r)), 'clip_scores': {r: scores[r] for r in ids}})
        summary.append({'min_size': lo, 'max_size': hi, 'target': target, 'available': len(pool), 'preselected': len(mandatory), 'additional_sampled': need, 'sampled': len(chosen)})
    return (selected, summary)

def classify_pair(pair, scores, threshold, decision):
    if min((scores[r] for r in pair)) < threshold:
        return 'score_mismatch'
    if decision == 'keep_augment':
        return 'keep_augment'
    if decision == 'mismatch':
        return 'vlm_mismatch_review'
    return 'none'

def export_cluster(ids, representative, scores, threshold, decisions):
    expected = set(itertools.combinations(sorted(ids), 2))
    assert set(decisions) == expected, 'Selected pair missing or duplicated'
    actions = {p: classify_pair(p, scores, threshold, d) for (p, d) in decisions.items()}
    effective = {p: 'mismatch' if a in ['score_mismatch', 'vlm_mismatch_review'] else a for (p, a) in actions.items()}
    return (base.export_cluster(ids, representative, effective), actions)

def assign_cells(exports, reference_cells):
    reference_index = defaultdict(list)
    for cell in reference_cells:
        reference_index[identity(cell)].append(cell)
    (used_ids, canonical, result) = (set(), {}, [])
    for cluster in sorted(exports, key=lambda c: c['cluster_id']):
        for strategy in ['a', 'b']:
            retained = cluster['strategies'][strategy]['retained']
            for source in sorted(cluster['strategies'][strategy]['deleted']):
                cell = {'cluster_id': cluster['cluster_id'], 'strategy': strategy, 'arm': cluster['cluster_id'] + ':' + strategy, 'source_id': source, 'retained_ids': retained, 'evidence_route': 'dual_ai'}
                key = identity(cell)
                candidates = sorted(reference_index[key], key=lambda c: (c['strategy'] != strategy, c['reuse_of'] is not None, c['cell_id']))
                available = [c for c in candidates if c['cell_id'] not in used_ids]
                cid = available[0]['cell_id'] if available else f"coverage-{cluster['cluster_id']}-{strategy}-{source}"
                assert cid not in used_ids
                used_ids.add(cid)
                reuse = canonical.get(key)
                canonical.setdefault(key, cid)
                cell.update(cell_id=cid, reuse_of=reuse, unit_id=canonical[key])
                result.append(cell)
    assert all((c['reuse_of'] is None or c['reuse_of'] in used_ids for c in result))
    return result

def load_sampling_inputs():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    p = protocol()['inputs']
    ids = pq.read_table(p['manifest'], columns=['record_id']).column(0).to_pylist()
    labels = np.load(p['components'], mmap_mode='r')
    assert ids == sorted(set(ids)) and labels.shape == (len(ids),) and (len(ids) == 50000)
    parts = {}
    with open(p['lineage']) as handle:
        for line in handle:
            row = json.loads(line)
            parts[row['record_id']] = row['partition']
    values = np.load(p['clip_scores'], mmap_mode='r')
    meta = pq.read_table(p['clip_metadata'], columns=['record_id', 'row_index']).to_pylist()
    assert len(meta) == len(values) == len(parts) == len(ids)
    assert sorted((r['row_index'] for r in meta)) == list(range(len(ids)))
    scores = {r['record_id']: float(values[r['row_index']]) for r in meta}
    assert set(ids) == set(parts) == set(scores)
    groups = defaultdict(list)
    for (rid, label) in zip(ids, labels):
        groups[int(label)].append(rid)
    assert max(map(len, groups.values())) <= 5, 'Named K=5 graph has component larger than5'
    return (list(groups.values()), parts, scores)

def prepare():
    RUN.mkdir(exist_ok=True)
    REPORT.mkdir(exist_ok=True)
    dump(RUN / 'settings.json', protocol())
    (groups, parts, scores) = load_sampling_inputs()
    p = protocol()['inputs']
    candidates = readl(p['initial_sample'])
    for c in candidates:
        assert all((scores[r] == value for (r, value) in c['clip_scores'].items())), 'Original score cache changed'
    (sample, strata) = sample_clusters(groups, parts, scores, candidates)
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
    random.Random(20_261_046).shuffle(order)
    summary = {'run_id': RUN.name, 'strata': strata, 'clusters': len(sample), 'records': len(selected), 'preselected': sum((c['preselected'] for c in sample)), 'selected_cluster_pairs': sum((c['size'] * (c['size'] - 1) // 2 for c in sample))}
    for target in [RUN, REPORT]:
        writel(target / 'sample_clusters.jsonl', sample)
        dump(target / 'sample_summary.json', summary)
    dump(RUN / 'sample_records.json', {'records': records, 'opaque_ids': {rid: f'R{i + 1:03d}' for (i, rid) in enumerate(order)}})

def export():
    p = protocol()['inputs']
    threshold = float(json.loads((REPORTS / 'caption_actions/threshold_frozen.json').read_text())['t_new'])
    path = RESOURCES / 'runs/caption_actions/actions/actions_test.jsonl'
    rows = readl(path)
    sample = readl(RUN / 'sample_clusters.jsonl')
    expected = {pair for c in sample for pair in itertools.combinations(c['record_ids'], 2)}
    (selected, seen) = ({}, set())
    for row in rows:
        pair = tuple(sorted((row['record_id_a'], row['record_id_b'])))
        assert pair not in seen, 'Duplicate test action pair'
        seen.add(pair)
        if pair in expected:
            assert row['partition'] == 'test'
            decision = row['parsed']['decision']
            assert decision in {'duplicate', 'keep_augment', 'mismatch', 'non_duplicate', 'uncertain'}
            if row['status'] != 'completed' or not row['parsed']['contract_valid']:
                assert decision == 'uncertain'
            selected[pair] = row
    assert set(selected) == expected, f'Selected action pairs missing: {len(expected - set(selected))}'
    (exports, actions) = ([], [])
    for c in sample:
        decisions = {pair: selected[pair]['parsed']['decision'] for pair in itertools.combinations(c['record_ids'], 2)}
        (strategies, typed) = export_cluster(c['record_ids'], c['representative_id'], c['clip_scores'], threshold, decisions)
        exports.append({**c, 'strategies': strategies})
        for (pair, action) in typed.items():
            row = selected[pair]
            actions.append({'cluster_id': c['cluster_id'], 'record_ids': list(pair), 'score': min((c['clip_scores'][r] for r in pair)), 'effective_action': action, 'vlm_decision_used': row['parsed']['decision'] if action != 'score_mismatch' else None, 'source_file': row['source_file'], 'contract_valid': row['parsed']['contract_valid'], 'status': row['status']})
    cells = assign_cells(exports, readl(p['reference_cells']))
    frozen = json.loads((RUN / 'sample_records.json').read_text())
    frozen.update(cells=cells, retained_ids={f"{c['cluster_id']}:{s}": c['strategies'][s]['retained'] for c in exports for s in ['a', 'b']})
    dump(RUN / 'frozen_inputs.json', frozen)
    summary = {'run_id': RUN.name, 'clusters': len(exports), 'records': sum((c['size'] for c in exports)), 'strategies': {s: {k: sum((len(c['strategies'][s][k]) for c in exports)) for k in ['retained', 'held', 'deleted']} for s in ['a', 'b']}, 't_new': threshold, 'deleted_cells': len(cells), 'unique_audit_queries': sum((c['reuse_of'] is None for c in cells)), 'reused_cells': sum((c['reuse_of'] is not None for c in cells)), 'typed_pair_actions': dict(Counter((r['effective_action'] for r in actions)))}
    for target in [RUN, REPORT]:
        writel(target / 'exports.jsonl', exports)
        writel(target / 'effective_pair_actions.jsonl', actions)
        writel(target / 'audit_cells.jsonl', cells)
        dump(target / 'export_summary.json', summary)
        for strategy in ['a', 'b']:
            for state in ['retained', 'held', 'deleted']:
                writel(target / f'{strategy}_{state}.jsonl', [{'cluster_id': c['cluster_id'], 'record_id': rid} for c in exports for rid in c['strategies'][strategy][state]])

def audit():
    frozen = json.loads((RUN / 'frozen_inputs.json').read_text())
    for auditor in [1, 2]:
        results = {c['cell_id']: base.one_audit(auditor, c, frozen) for c in frozen['cells'] if c['reuse_of'] is None}
        rows = [{**results[c['reuse_of'] or c['cell_id']], **c} for c in frozen['cells']]
        writel(RUN / f'audit_{auditor}.jsonl', rows)

def summarize():
    frozen = json.loads((RUN / 'frozen_inputs.json').read_text())
    audits = [{r['cell_id']: r for r in readl(RUN / f'audit_{i}.jsonl')} for i in [1, 2]]
    expected = {c['cell_id'] for c in frozen['cells']}
    assert all((set(a) == expected for a in audits))
    results = [base.compare(c, audits[0][c['cell_id']], audits[1][c['cell_id']]) for c in frozen['cells']]
    summary = json.loads((RUN / 'export_summary.json').read_text())
    summary['by_strategy'] = {}
    for strategy in ['a', 'b']:
        part = [r for r in results if r['strategy'] == strategy]
        accepted = Counter((r['coverage_label'] for r in part if not r['needs_adjudication']))
        summary['by_strategy'][strategy] = dict(ai_agreement=sum(accepted.values()), covered_single=accepted['covered_single'], covered_joint=accepted['covered_joint'], covered=accepted['covered_single'] + accepted['covered_joint'], lost=accepted['lost'], uncertain=accepted['uncertain'], unresolved=sum((r['needs_adjudication'] for r in part)))
    writel(RUN / 'agreement.jsonl', results)
    dump(REPORT / 'results.json', summary)
    return summary
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'export', 'audit', 'summarize'])
    a = p.parse_args()
    globals()[a.action]()
