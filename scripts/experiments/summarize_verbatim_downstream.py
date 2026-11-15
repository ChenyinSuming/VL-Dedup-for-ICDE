#!/usr/bin/env python3
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
"""VERBATIM_EXPORT_RUN paired families and descriptive caption-count regressions."""
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiments.run_verbatim_export import RUN,REPORT,CAPTION_DOWNSTREAM_RUN,FLAG_REP_RUN,METHODS,load,dump,identity
from scripts.experiments.summarize_flag_rep_downstream import family
from scripts.experiments.summarize_caption_downstream import METRICS


def read_values(new_methods):
    sources=[(m,RUN,m) for m in new_methods]
    sources += [('caption_downstream_'+m,CAPTION_DOWNSTREAM_RUN,m) for m in ('no_dedup','keep_one','semdedup','clean','b','b_cov','clipscore','random')]
    sources += [('flag_rep_'+m,FLAG_REP_RUN,m) for m in ('b_fr','b_fr_cov','clipscore_fr','random_fr')]
    values={};receipts=[]
    for key,origin,method in sources:
        values[key]={m:[] for m in METRICS}
        for seed in range(20261104,20261114):
            d=origin/'training'/str(seed)/method
            t=load(d/'training_complete.json');e=load(d/'evaluation_complete.json')
            if t['steps']!=4370 or e['seed']!=seed or e['test_images']!=5000:raise RuntimeError(f'Receipt contract differs: {d}')
            for metric in METRICS:values[key][metric].append(e['metrics'][metric])
            receipts.append(dict(method=key,seed=seed,source=str(d),train_seconds=t['train_seconds'],eval_seconds=e['eval_seconds']))
    for m in ('no_dedup','keep_one','semdedup','clean'):values[m]=values['caption_downstream_'+m]
    return values,receipts


def summary(values):
    return {method:{m:dict(n=len(v),mean=statistics.mean(v),sample_sd=statistics.stdev(v)) for m,v in metrics.items()}
            for method,metrics in values.items() if method not in ('caption_downstream_no_dedup','caption_downstream_keep_one','caption_downstream_semdedup','caption_downstream_clean')}


def regressions(values):
    from scipy.stats import linregress
    comp=load(RUN/'training_sets_complete.json')['all_compositions']
    names=[k for k,v in comp.items() if v['retained_swapped']/v['records']<.05]
    def fit(names):
        x=[comp[k]['distinct_correct_captions'] for k in names]
        y=[statistics.mean(values[k]['CIDEr']) for k in names]
        r=linregress(x,y)
        return dict(methods=names,slope_per_caption=r.slope,slope_per_thousand=1000*r.slope,intercept=r.intercept,r=r.rvalue,
                    residuals={k:y0-(r.intercept+r.slope*x0) for k,x0,y0 in zip(names,x,y,strict=True)})
    allfit=fit(names)
    seven=['flag_rep_b_fr','flag_rep_b_fr_cov','flag_rep_clipscore_fr','caption_downstream_b','caption_downstream_b_cov','caption_downstream_clipscore','caption_downstream_clean']
    frozenfit=fit(seven)
    predictions={m:dict(distinct_correct_captions=comp[m]['distinct_correct_captions'],
        retained_swapped_fraction=comp[m]['retained_swapped']/comp[m]['records'],
        under_five_percent_swapped=comp[m]['retained_swapped']/comp[m]['records']<.05,
        predicted_CIDEr=frozenfit['intercept']+frozenfit['slope_per_caption']*comp[m]['distinct_correct_captions'],
        measured_CIDEr=statistics.mean(values[m]['CIDEr'])) for m in METHODS}
    for p in predictions.values():p['residual']=p['measured_CIDEr']-p['predicted_CIDEr']
    return dict(all_under_five_percent_swapped=allfit,frozen_seven_fit=frozenfit,predictions=predictions,
                interpretation='Descriptive post-hoc regression over correlated training sets; no causal or selection claim.')


def main():
    if load(RUN/'training_campaign_complete.json')['status']!='complete':raise RuntimeError('Complete 40 new runs first')
    values,receipts=read_values(METHODS)
    stat=load(CONFIG / 'verbatim_export.json')['downstream']['B3']
    result=dict(**identity(),status='complete',n=10,method_metrics=summary(values),per_seed=values,receipts=receipts,
        main_family=family(values,stat['main_contrasts'],231),additional_family=family(values,stat['additional_contrasts'],63),
        regressions=regressions(values),retraining_gate=load(RUN/'retraining_gate.json'),runtime_gate=load(RUN/'runtime_gate.json'))
    dump(REPORT/'downstream/results.json',result)


if __name__=='__main__':main()
