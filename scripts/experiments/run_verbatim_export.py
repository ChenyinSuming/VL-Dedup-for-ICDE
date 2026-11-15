#!/usr/bin/env python3
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
"""VERBATIM_EXPORT_RUN CPU export; frozen inputs are read-only, provenance is reporting only."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_flag_rep_export import RES, COMBINED_ACTIONS_RUN, COVERAGE_EXPORT_RUN, CAPTION_DOWNSTREAM_RUN, STATES, validate_arm, counts
from scripts.experiments.run_caption_downstream_sets import load, rows, dump, writel, arm_members, composition

RUN = RES / 'runs/verbatim_export'
FLAG_REP_RUN = RES / 'runs/flag_rep_export'
REPORT = ROOT / 'outputs/reports/verbatim_export'
METHODS = ('b_fr_cov_vb', 'clipscore_vb', 'random_vb', 'b_fr_vb')


def norm(text):
    return re.sub(r'\s+', ' ', text.strip().lower()).rstrip(' .')


def verbatim(ids, arm, clip, text, strict=False):
    validate_arm(ids, arm)
    groups = defaultdict(list)
    for rid in arm['retained']:
        groups[text[rid] if strict else norm(text[rid])].append(rid)
    kept = sorted(min(group, key=lambda r: (-clip[r], r)) for group in groups.values())
    removed = sorted(set(arm['retained']) - set(kept))
    result = dict(retained=kept, pending=list(arm['pending']), deleted=sorted(arm['deleted'] + removed))
    validate_arm(ids, result)
    assert {norm(text[r]) for r in arm['retained']} == {norm(text[r]) for r in kept}
    assert set(result['deleted']) - set(arm['deleted']) == set(removed)
    witnesses = {r: min((k for k in kept if (text[k] == text[r] if strict else norm(text[k]) == norm(text[r]))),
                       key=lambda k: (-clip[k], k)) for r in removed}
    return result, witnesses


def matched_controls(ids, clip, removed):
    ids = sorted(ids)
    return dict(clipscore_vb=sorted(set(ids) - set(sorted(ids, key=lambda r: (clip[r], r))[:removed])),
                random_vb=sorted(set(ids) - set(random.Random(20261107).sample(ids, removed))))


def gate(name, actual, expected):
    path = RUN / 'gates.json'
    receipt = load(path) if path.exists() else dict(checks={})
    receipt['checks'][name] = dict(actual=actual, expected=expected, passed=actual == expected)
    dump(path, receipt)
    if actual != expected:
        raise RuntimeError(f'Gate {name}: independently computed {actual!r} != {expected!r}')


def identity():
    return dict(task='verbatim_export',
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())


def benchmark():
    start = time.monotonic()
    exports = list(rows(FLAG_REP_RUN / 'benchmark_exports.jsonl'))
    text = {r['record_id']: r['text'] for r in rows(RES / 'datasets/benchmark/public_records.jsonl')}
    clip = {r: v['clipscore'] for r, v in load(COMBINED_ACTIONS_RUN / 'record_scores.json').items()}
    results, witnesses = [], []
    for e in exports:
        out = {k: e[k] for k in ('partition', 'component', 'record_ids')}
        for source, target in [('flag_rep_cov', 'b_cov'), ('flag_rep', 'b')]:
            gate_key = f"{e['partition']}/{e['component']}/{target}"
            if e[source] != e[target]:
                raise RuntimeError(f'FLAG_REP_RUN benchmark frozen strategy disagreement: {gate_key}')
            new, w = verbatim(e['record_ids'], e[source], clip, text)
            strict, sw = verbatim(e['record_ids'], e[source], clip, text, strict=True)
            out[target], out[target + '_vb'] = e[source], new
            out[target + '_strict_deleted'] = len(sw)
            out[target + '_vb_deleted'] = sorted(w)
            witnesses.extend(dict(partition=e['partition'], component=e['component'], method=target+'_vb',
                                  source_id=r, witness_id=k, raw_text_equal=text[r] == text[k]) for r, k in w.items())
        results.append(out)
    partition = {}
    for part in ('train', 'validation', 'test'):
        selected = [e for e in results if e['partition'] == part]
        partition[part] = {}
        for method in ('b_cov', 'b'):
            value = dict(clusters=len(selected), records=sum(len(e['record_ids']) for e in selected),
                         before=counts([e[method] for e in selected]), after=counts([e[method+'_vb'] for e in selected]),
                         normalized_deleted=sum(len(e[method+'_vb_deleted']) for e in selected),
                         strict_deleted=sum(e[method+'_strict_deleted'] for e in selected))
            partition[part][method] = value
            expected = load(CONFIG / 'verbatim_export.json')['gates']
            gate(f'{part}_{method}_normalized', value['normalized_deleted'], expected['benchmark_normalized'][method][part])
            if method == 'b_cov':
                gate(f'{part}_{method}_strict', value['strict_deleted'], expected['benchmark_strict_b_cov'][part])
    sample = list(rows(COVERAGE_EXPORT_RUN / 'sample_clusters.jsonl'))
    by_component = {e['component']: e for e in results}
    sampled = []
    for s in sample:
        e = by_component[s['component']]
        assert set(e['record_ids']) == set(s['record_ids'])
        sampled.append(dict(e, cluster_id=s['cluster_id']))
    gate('sample_clusters_records', [len(sampled), sum(len(e['record_ids']) for e in sampled)], [200, 953])
    audits = list(rows(COVERAGE_EXPORT_RUN / 'benchmark_audits.jsonl'))
    samples = {}
    for label, selected, denominator, strict_lost, loose_lost in [
        ('all200', sampled, 753, 266, 615),
        ('size_five', [e for e in sampled if len(e['record_ids']) == 5], 640, 227, 520)]:
        selected_by_id = {e['cluster_id']: e for e in selected}
        value = dict(clusters=len(selected), records=sum(len(e['record_ids']) for e in selected), methods={})
        lost = {kind: [(a['cluster_id'], a['source_id']) for a in audits if a['strategy'] == 'keep_one'
                        and a['cluster_id'] in selected_by_id
                        and (all(v == 'lost' for v in a['votes']) if kind == 'strict' else 'lost' in a['votes'])]
                for kind in ('strict', 'loose')}
        gate(label + '_keep_one_lost', [len(lost['strict']), len(lost['loose'])], [strict_lost, loose_lost])
        for method in ('b_cov', 'b'):
            before, after = counts([e[method] for e in selected]), counts([e[method+'_vb'] for e in selected])
            added = [(e, r) for e in selected for r in e[method+'_vb_deleted']]
            raw_equal = sum(any(text[r] == text[k] for k in e[method+'_vb']['retained']) for e, r in added)
            gate(label + '_' + method + '_new_deletions_exact_caption_witness', raw_equal, len(added))
            fates = {}
            for kind, lost_sources in lost.items():
                fate = Counter()
                for cid, rid in lost_sources:
                    e = selected_by_id[cid]
                    state = ('retained' if rid in e[method+'_vb']['retained'] else
                             'held' if rid in e[method+'_vb']['pending'] else
                             'vb_deleted' if rid in e[method+'_vb_deleted'] else 'old_deleted')
                    fate[state] += 1
                fates[kind] = {k: fate[k] for k in ('retained', 'held', 'vb_deleted', 'old_deleted')}
            old_audits=[a for a in audits if a['strategy']==method and a['cluster_id'] in selected_by_id]
            old_loss=[sum(all(v=='lost' for v in a['votes']) for a in old_audits),
                      sum('lost' in a['votes'] for a in old_audits)]
            gate(label+'_'+method+'_audited_deletion_records',len(old_audits),before['deleted'])
            if method=='b_cov' or label=='all200':
                reference=([1,11] if label=='all200' else [1,7]) if method=='b_cov' else [18,72]
                gate(label+'_'+method+'_original_loss',old_loss,reference)
            n = after['deleted']
            expectations = [Fraction(v*n, denominator) for v in (strict_lost, loose_lost)]
            value['methods'][method+'_vb'] = dict(before=before, after=after, added_deleted=len(added),
                raw_equal_witnesses=raw_equal, added_loss=0, original_deletion_loss=old_loss,
                all_added_size_five=all(len(e['record_ids']) == 5 for e, r in added),
                keep_one_expected_loss_exact=[str(v) for v in expectations],
                keep_one_expected_loss=[float(v) for v in expectations], keep_one_lost_fates=fates)
        samples[label] = value
    for method, expected in [('b_cov_vb', [714,148,91]), ('b_vb', [651,148,154])]:
        m = samples['all200']['methods'][method]
        gate(method+'_sample_states', [m['after'][s] for s in STATES], expected)
        gate(method+'_sample_added', m['added_deleted'], 36)
        gate(method+'_sample_added_size_five', m['all_added_size_five'], True)
    gate('size_five_clusters', samples['size_five']['clusters'], 160)
    gate('size_five_b_cov_vb_deleted', samples['size_five']['methods']['b_cov_vb']['after']['deleted'], 80)
    result = dict(**identity(), status='complete', partitions=partition, samples=samples,
                  normalized_caption_sets_preserved_clusters=len(results), elapsed_seconds=time.monotonic()-start,
                  original_deletion_loss={label:{m:r['original_deletion_loss'] for m,r in sample['methods'].items()} for label,sample in samples.items()},
                  interpretation='Frozen dual-AI strict/loose ranges; vb zero loss is caption-set preservation under specified norm, not a new audit.')
    writel(RUN/'benchmark_exports.jsonl', results)
    writel(RUN/'benchmark_vb_witnesses.jsonl', witnesses)
    dump(REPORT/'benchmark/results.json', result)
    dump(RUN/'benchmark_complete.json', result)


def caption_composition(frame, kept, images):
    value = composition(frame, kept, images)
    correct = frame[frame.record_id.isin(kept) & ~frame.caption_swapped.astype(bool)]
    keys = {(int(r.original_image_id), norm(r.text)) for r in correct.itertuples()}
    raw_keys = {(int(r.original_image_id), r.text) for r in correct.itertuples()}
    value.update(distinct_correct_captions=len(keys), redundant_correct_captions=len(correct)-len(keys),
                 raw_redundant_correct_captions=len(correct)-len(raw_keys))
    return value


def training_sets():
    import pandas as pd
    start = time.monotonic()
    records = {r['record_id']:r for r in rows(CAPTION_DOWNSTREAM_RUN/'records.jsonl') if r['partition']=='train'}
    gate('training_pool_records',len(records),35208)
    text={r:v['caption'] for r,v in records.items()}; clip={r:v['clipscore'] for r,v in records.items()}
    exports=list(rows(FLAG_REP_RUN/'training_exports.jsonl'))
    for e in exports:
        assert all(records[r]['component']==e['component'] for r in e['record_ids'])
    for method,n in [('b_fr_cov',26405),('b_fr',24161)]:
        old=set(arm_members(exports,method,records)['retained'])
        archived=pd.read_parquet(FLAG_REP_RUN/f'training_sets/{method}.parquet')
        gate(method+'_exact_r70_IDs',old==set(archived.record_id) and len(archived)==len(old),True)
        gate(method+'_r70_records',len(old),n)
    new=[]; witnesses=[]
    for e in exports:
        out=dict(component=e['component'],record_ids=e['record_ids'])
        for method in ('b_fr_cov','b_fr'):
            arm,w=verbatim(e['record_ids'],e[method],clip,text)
            _,strict_w=verbatim(e['record_ids'],e[method],clip,text,strict=True)
            out[method+'_vb']=arm; out[method+'_strict_deleted']=len(strict_w)
            out[method+'_vb_deleted']=sorted(w)
            witnesses.extend(dict(component=e['component'],method=method+'_vb',source_id=r,witness_id=k,raw_text_equal=text[r]==text[k]) for r,k in w.items())
        new.append(out)
    methods={m:arm_members(new,m,records)['retained'] for m in ('b_fr_cov_vb','b_fr_vb')}
    for method in ('b_fr_cov','b_fr'):
        gate(method+'_normalized_deleted',sum(len(e[method+'_vb_deleted']) for e in new),7849)
        gate(method+'_strict_deleted',sum(e[method+'_strict_deleted'] for e in new),7843)
    removed=len(records)-len(methods['b_fr_cov_vb'])
    methods.update(matched_controls(records,clip,removed))
    for method in ('clipscore_vb','random_vb'):gate(method+'_matched_removed',len(records)-len(methods[method]),removed)
    # Reporting truth is read only after all decisions are fixed.
    frame=pd.read_parquet(RES/'runs/caption_pilot/b/train/corpus.parquet')
    assert set(frame.record_id)==set(records)
    images=sorted(frame.original_image_id.unique())
    lineage=frame.set_index('record_id').original_image_id.to_dict()
    compositions={m:caption_composition(frame,set(kept),images) for m,kept in methods.items()}
    expected=load(CONFIG / 'verbatim_export.json')['gates']['training']
    for method in METHODS:
        for key,value in expected[method].items():
            gate(method+'_'+key,(len(records)-compositions[method]['records']) if key=='deleted' else compositions[method][key],value)
    all_compositions={}
    for prefix,origin,names in [('caption_downstream',CAPTION_DOWNSTREAM_RUN,('no_dedup','keep_one','semdedup','clean','b','b_cov','clipscore','random')),
                                ('flag_rep',FLAG_REP_RUN,('b_fr','b_fr_cov','clipscore_fr','random_fr'))]:
        for name in names:
            archived=pd.read_parquet(origin/f'training_sets/{name}.parquet')
            kept=set(archived.record_id)
            # Clean is an independent original-caption corpus with clean:* IDs.
            reference=archived.assign(is_copy=False,caption_swapped=False) if name=='clean' else frame
            all_compositions[prefix+'_'+name]=caption_composition(reference,kept,images)
    all_compositions.update(compositions)
    target=RUN/'training_sets';target.mkdir(exist_ok=True)
    for method,kept in methods.items():
        pd.DataFrame([dict(record_id=r,original_image_id=int(lineage[r]),image_path=records[r]['image_path'],text=text[r]) for r in sorted(kept)]).to_parquet(target/f'{method}.parquet',index=False)
    dump(RUN/'subsets.json',load(FLAG_REP_RUN/'subsets.json'))
    assert load(RUN/'subsets.json')==load(CAPTION_DOWNSTREAM_RUN/'subsets.json')
    writel(RUN/'training_exports.jsonl',new);writel(RUN/'training_vb_witnesses.jsonl',witnesses)
    value=dict(**identity(),status='complete',composition=compositions,all_compositions=all_compositions,
               states={m:counts([e[m] for e in new]) for m in ('b_fr_cov_vb','b_fr_vb')},
               pool_records=len(records),matched_removed=removed,elapsed_seconds=time.monotonic()-start)
    dump(RUN/'training_sets_complete.json',value);dump(REPORT/'downstream/training_sets.json',value)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['benchmark','training-sets']);args=parser.parse_args()
    try:
        (benchmark if args.command=='benchmark' else training_sets)()
    except Exception as exc:
        dump(RUN/'stop.json',dict(status='stopped',phase=args.command,error_type=type(exc).__name__,error=str(exc),**identity()))
        raise
