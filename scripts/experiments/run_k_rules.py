"""CPU-only rules for choosing K on saved edges.

Run each phase under the shared CPU flock. Validation is committed before test.
"""
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_graph_baselines as graph
from scripts import run_coco_scale as coco
from scripts.cluster_with_size_cap import build_graph
import numpy as np
import pandas as pd
import pyarrow as pa
from threadpoolctl import threadpool_limits

OUT = graph.RUNS / 'k_rules'
REPORT = ROOT / 'outputs/reports/k_rules'
PROTOCOL = json.loads((CONFIG / 'k_rules.json').read_text())
SETTINGS = PROTOCOL['settings']
GRID = PROTOCOL['K_rules']['paper_grid']
DENSE = PROTOCOL['K_rules']['dense_grid']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def read(path):
    return json.loads(path.read_text())


def size_statistics(sizes):
    """Nearest ranks on non-singleton entities, with integer arithmetic."""
    counts = Counter(int(s) for s in sizes if s >= 2)
    entities, records = sum(counts.values()), sum(s*n for s, n in counts.items())
    if not entities:
        return dict(entities_ge2=0, records_ge2=0, max=None, p95_entity=None, p95_record=None)

    def percentile(weighted):
        total = records if weighted else entities
        rank, cumulative = (95*total+99)//100, 0
        for size, count in sorted(counts.items()):
            cumulative += count*size if weighted else count
            if cumulative >= rank:
                return size

    return dict(entities_ge2=entities, records_ge2=records, max=max(counts),
                p95_entity=percentile(False), p95_record=percentile(True))


def load_setting(setting):
    surface, model = setting['surface'], setting['model']
    if setting['setting'] <= 8:
        ids, a, b, _, _ = graph.inputs(surface, model)
        frame = None
    else:
        source = graph.RUNS / ('sop_scale' if surface == 'sop_scale'
                               else 'coco_scale')
        frame = pd.read_parquet(source / 'corpus.parquet')
        candidate_path = source / ('stage_a_amendment2/candidates.parquet' if surface == 'sop_scale'
                                   else 'candidates.parquet')
        candidates = pd.read_parquet(candidate_path, columns=graph.ENDS)
        raw = np.load(source / f'{model}_raw.npy', mmap_mode='r')
        calibrated = np.load(source / f'{model}_calibrated.npy', mmap_mode='r')
        assert len(raw) == len(calibrated) == len(candidates)
        ids, a, b, _ = coco.positive_edges(frame, candidates, raw, calibrated,
                                          2/3 if model == 'vl_dedup_lr' else 1.)
        frame = frame.set_index('record_id').loc[ids].rename_axis('record_id').reset_index()
    return dict(setting=setting, ids=ids, a=a, b=b, frame=frame)


def truth_for(data, partition):
    setting, frame = data['setting'], data['frame']
    if setting['setting'] <= 8:
        return graph.metadata(setting['surface'], partition)
    if partition == 'all':
        return frame
    return frame[frame.evaluation_partition.eq(partition)]


def entity_summary(data, partition):
    truth = truth_for(data, partition)
    assert (data['ids'].get_indexer(truth.record_id) >= 0).all()
    summaries, distributions = {}, []
    for convention in ['all', 'non_M']:
        subset = truth if convention == 'all' or 'original_level' not in truth else truth[truth.original_level.ne('M')]
        sizes = subset.groupby('visual_group_id').size().tolist()
        summaries[convention] = dict(records=len(subset), entities=len(sizes), **size_statistics(sizes))
        for size, count in sorted(Counter(sizes).items()):
            distributions.append(dict(setting=data['setting']['setting'], partition=partition,
                                      convention=convention, size=size, entities=count, records=size*count))
    summaries['non_M_differs'] = summaries['all'] != summaries['non_M']
    return summaries, distributions


def measure(data, partition, cap):
    begin = time.perf_counter()
    labels, _, _, _ = build_graph(len(data['ids']), data['a'], data['b'], cap)
    graph_seconds = time.perf_counter()-begin
    truth = truth_for(data, partition)
    if data['setting']['setting'] <= 8:
        metrics = graph.evaluate(labels, truth, data['ids'])
    else:
        positions = data['ids'].get_indexer(truth.record_id)
        metrics = coco.graph_metrics(labels[positions], truth.visual_group_id.to_numpy())
    structure = graph.structure(labels)
    return {**metrics, 'setting':data['setting']['setting'], 'partition':partition, 'K':cap,
            'clusters':structure['clusters'], 'max_cluster':structure['largest_cluster'],
            'U':structure['U'], 'graph_seconds':graph_seconds, 'wall_seconds':time.perf_counter()-begin}


def curve(data, partition, caps, phase):
    rows = []
    path = OUT / f'{phase}_{data["setting"]["setting"]}.json'
    for cap in caps:
        rows.append(measure(data, partition, cap))
        dump(path, rows)
    return rows


def select(rows, grid):
    eligible = [r for r in rows if r['K'] in grid]
    return min(eligible, key=lambda r: (-Fraction(r['exact_f1']),
                                       float('inf') if r['K'] == 'inf' else r['K']))['K']


def gates():
    # Only predefined historical test points are evaluated here, never used for selection.
    expected = {
        1: [(5,8696,106,5,.993658), ('inf',8701,307,0,.982664)],
        2: [(5,8687,78,14,.994733)], 3: [(5,9268,376,2417,.869052)],
        4: [(64,107201,407,0,.998105), (5,18895,106,88306,.299441)],
        5: [(5,8693,385,8,.977895), ('inf',8698,7711852,3,.002251)],
        6: [(5,5583,1105,3118,.725583)], 7: [(5,9031,1199,2654,.824184)],
        8: [(64,106426,27286,775,.883522)], 9: [(8,4852,2139,22012,.286634)],
        10: [(8,2495,7852,24369,.134100)], 11: [(5,1160359,371,2114,.998930)],
        12: [(5,1159328,1349,3145,.998066)]}
    receipt = dict(status='running', protocol_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(), rows=[])
    previous = read(OUT/'gate_counts.json')['rows'] if (OUT/'gate_counts.json').exists() else []
    begin = time.perf_counter()
    workloads = []
    for setting in SETTINGS:
        start = time.perf_counter()
        data = load_setting(setting)
        workloads.append(dict(setting=setting['setting'], nodes=len(data['ids']),
                              positive_edges=len(data['a']), load_seconds=time.perf_counter()-start))
        for cap, tp, fp, fn, f1 in expected[setting['setting']]:
            done = next((r for r in previous if r['setting']==setting['setting'] and r['K']==cap and r['matched']), None)
            if done is not None:
                receipt['rows'].append(done)
                continue
            row = measure(data, setting['test'], cap)
            ok = [row[k] for k in ['tp','fp','fn']] == [tp,fp,fn] and abs(row['f1']-f1) <= .0000005
            receipt['rows'].append(dict(**row, expected=dict(tp=tp,fp=fp,fn=fn,f1=f1), matched=ok))
            dump(OUT/'gate_counts.json', receipt)
            if not ok:
                receipt.update(status='failed', wall_seconds=time.perf_counter()-begin)
                dump(OUT/'gate_counts.json', receipt)
                raise SystemExit(f'Count gate mismatch: setting {setting["setting"]}, K={cap}')
        print('count gate passed',setting['setting'],flush=True)
    receipt.update(status='passed', wall_seconds=time.perf_counter()-begin, workloads=workloads)
    dump(OUT/'gate_counts.json', receipt)
    data = load_setting(SETTINGS[0])
    start = time.perf_counter()
    pilot = curve(data, 'validation', list(range(2,65)), 'pilot')
    seconds = time.perf_counter()-start
    # Scale all work by nodes+positive edges; allow 2x for counting/sorting variation.
    # Full dense counts include rebuilding once per validation/test partition.
    base = len(data['ids'])+len(data['a'])
    weighted_runs = sum((w['nodes']+w['positive_edges'])/base *
                        (2*(len(DENSE)+3) if w['setting']<=10 else 8) for w in workloads)
    estimate = seconds/63*weighted_runs*2 + sum(w['load_seconds'] for w in workloads)*3 + receipt['wall_seconds']
    timing = dict(status='passed' if estimate<=7200 else 'failed', pilot_K=list(range(2,65)),
                  pilot_wall_seconds=seconds, estimate_seconds=estimate, limit_seconds=7200,
                  method='2x (nodes+positive_edges)-weighted per-K pilot; all partitions plus 3 off-grid statistic caps; 3 input loads plus count-gate elapsed',
                  weighted_runs=weighted_runs, workloads=workloads)
    dump(OUT/'timing_gate.json', timing)
    print(json.dumps(timing),flush=True)
    if timing['status']!='passed':
        raise SystemExit('Timing gate exceeds two hours')


def validation():
    assert read(OUT/'gate_counts.json')['status'] == read(OUT/'timing_gate.json')['status'] == 'passed'
    frozen = dict(run_id=OUT.name, selection_partition='validation (COCO all entity statistics only)',
                  protocol_commit=read(OUT/'gate_counts.json')['protocol_commit'], settings=[], distributions=[])
    for setting in SETTINGS:
        data = load_setting(setting)
        partition = setting['validation'] or 'all'
        summary, distribution = entity_summary(data, partition)
        stats = summary['all']
        rules = {k:stats[k] for k in ['max','p95_entity','p95_record']}
        assert all(v is not None for v in rules.values())
        rules.update(prior_2=2, prior_5=5, prior_inf='inf')
        if setting['validation']:
            caps = sorted(set(x for x in DENSE if x!='inf') | set(rules[k] for k in ['max','p95_entity','p95_record']))+['inf']
            rows = curve(data, 'validation', caps, 'validation')
            rules.update(val_grid=select(rows,GRID), val_dense=select(rows,DENSE))
            expected = 64 if setting['surface']=='large_entities' else 8 if setting['surface']=='sop_scale' else 5
            if rules['val_grid'] != expected:
                dump(OUT/'validation_selection_gate_failure.json', dict(setting=setting,actual=rules,expected_val_grid=expected))
                raise SystemExit(f'Validation selection gate mismatch: {setting["setting"]}')
        else:
            rules.update(val_grid=5, val_dense=None)
            assert all(k in [2,3,4,5,6,7,8,'inf',None] for k in rules.values())
        frozen['settings'].append(dict(**setting,entity_summary={partition:summary},rules=rules))
        frozen['distributions'].extend(distribution)
        dump(OUT/'selection_in_progress.json',frozen)
        print('validation frozen',setting['setting'],rules,flush=True)
    dump(REPORT/'selection_frozen.json',frozen)
    dump(OUT/'selection_frozen.json',frozen)


def test():
    # Require the exact selection file in a Git commit, not merely on disk.
    relative = str((REPORT/'selection_frozen.json').relative_to(ROOT))
    committed = subprocess.check_output(['git','show',f'HEAD:{relative}'],cwd=ROOT)
    assert committed == (REPORT/'selection_frozen.json').read_bytes()
    selection_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    frozen = read(REPORT/'selection_frozen.json')
    results = dict(run_id=OUT.name, selection_commit=selection_commit,
                   gate_counts=read(OUT/'gate_counts.json'), timing_gate=read(OUT/'timing_gate.json'),
                   settings=[], curves=[], distributions=frozen['distributions'])
    for setting in frozen['settings']:
        data = load_setting(setting)
        rules = setting['rules'].copy()
        caps = PROTOCOL['K_rules']['COCO']['grid'] if setting['surface']=='coco_scale' else sorted(
            set(x for x in DENSE if x!='inf') | {v for v in rules.values() if isinstance(v,int)})+['inf']
        rows = curve(data,setting['test'],caps,'test')
        rules['test_best'] = select(rows,caps if setting['surface']=='coco_scale' else DENSE)
        base = next(r for r in rows if r['K']==rules['val_grid'])
        comparisons = []
        for rule,cap in rules.items():
            if cap is None:
                continue
            row = next(r for r in rows if r['K']==cap)
            comparisons.append(dict(rule=rule,**row,delta_f1=float(Fraction(row['exact_f1'])-Fraction(base['exact_f1'])),
                                    U_ratio=row['U']/base['U']))
        summary = setting['entity_summary'].copy()
        if setting['validation']:
            test_summary,distribution = entity_summary(data,setting['test'])
            summary[setting['test']] = test_summary
            results['distributions'].extend(distribution)
            results['curves'].extend(read(OUT/f'validation_{setting["setting"]}.json'))
        results['curves'].extend(rows)
        results['settings'].append(dict(**{k:v for k,v in setting.items() if k not in ['rules','entity_summary']},
                                        entity_summary=summary,rules=comparisons))
        dump(OUT/'test_results_in_progress.json',results)
        print('test complete',setting['setting'],flush=True)
    dump(REPORT/'results.json',results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['gates','validation','test'])
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    begin = time.perf_counter()
    with threadpool_limits(limits=4):
        globals()[args.phase]()
    dump(OUT/f'{args.phase}_stage.json',dict(status='completed',wall_seconds=time.perf_counter()-begin))


if __name__=='__main__':
    main()
