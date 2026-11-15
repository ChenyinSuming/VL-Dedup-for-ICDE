"""Checks of the released benchmark, artifacts and results. They need no resources folder."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'benchmark'
ART = ROOT / 'artifacts'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def records():
    return pd.read_csv(BENCH / 'records.csv.gz')


@pytest.fixture(scope='module')
def pairs():
    return pd.read_csv(BENCH / 'pairs.csv.gz')


def test_graph_metrics_recompute_from_cluster_labels():
    module = load_script('recompute_graph_metrics')
    new = module.compute().set_index('method')
    released = pd.read_csv(ROOT / 'results' / 'graph.csv').set_index('method')
    counts = [c for c in new.columns]
    assert (new[counts] == released.loc[new.index, counts]).all().all()


def test_benchmark_sizes(records, pairs):
    assert len(records) == 50000 and records.record_id.is_unique
    assert len(pairs) == 66788 and pairs.pair_id.is_unique
    assert records.partition.value_counts().to_dict() == {'train': 35820, 'validation': 7639, 'test': 6541}
    assert pairs.partition.value_counts().to_dict() == {'train': 46425, 'validation': 10326, 'test': 10037}
    assert pairs.relation.value_counts().to_dict() == {
        'duplicate': 52618, 'keep_augment_candidate': 5037, 'text_duplicate_only': 4481,
        'mismatch': 3599, 'hard_negative': 1053}


def test_pairs_stay_inside_partitions(records, pairs):
    part = records.set_index('record_id').partition
    assert (part.loc[pairs.record_id_a].to_numpy() == pairs.partition.to_numpy()).all()
    assert (part.loc[pairs.record_id_b].to_numpy() == pairs.partition.to_numpy()).all()
    key = np.where(records.source_dataset.eq('coco2017'), 'coco:' + records.coco_image_id.astype(str),
                   'sop:' + records.sop_super_class_id.astype(str))
    assert records.assign(key=key).groupby('key').partition.nunique().max() == 1


def test_visual_relation_matches_groups(records, pairs):
    group = records.set_index('record_id').visual_group_id
    same = group.loc[pairs.record_id_a].to_numpy() == group.loc[pairs.record_id_b].to_numpy()
    assert same[pairs.visual_relation.eq('same').to_numpy()].all()
    assert not same[pairs.visual_relation.eq('different').to_numpy()].any()


def test_surface_manifests():
    mv = pd.read_csv(BENCH / 'multiview' / 'records.csv.gz')
    le = pd.read_csv(BENCH / 'larger_entities' / 'records.csv.gz')
    edits = pd.read_csv(BENCH / 'edit_surfaces' / 'edits.csv.gz')
    assert len(mv) == 8550 and len(le) == 11000
    assert set(edits.level) == {'light', 'mild', 'strong'}
    assert (edits.level.value_counts() == 5709).all()
    check = pd.read_csv(ART / 'audits' / 'edited_image_check.csv')
    for level, both, either in [('mild', 0, 0), ('strong', 5, 10)]:
        c = check[check.level == level]
        lost = c.auditor_1.eq('subject_lost'), c.auditor_2.eq('subject_lost')
        assert (len(c), (lost[0] & lost[1]).sum(), (lost[0] | lost[1]).sum()) == (1362, both, either)
    human = pd.read_csv(ART / 'audits' / 'edited_image_human_check.csv')
    assert len(human) == 73 and (human.auditor_1 == 'same_subject_full').all() and (human.auditor_2 == 'same_subject_full').all()
    labels = pd.read_parquet(ART / 'clusters' / 'multiview.parquet')
    assert len(labels) == 58550 and set(mv.record_id) <= set(labels.record_id)


def test_vlm_outputs_match_paper_counts():
    verdicts = pd.read_csv(ART / 'vlm' / 'pair_verdicts.csv.gz')
    assert len(verdicts) == 71930
    assert (verdicts.partition == 'test').sum() == 10290
    assert verdicts[verdicts.partition == 'test'].mismatch_action.sum() == 1405
    t = json.loads((ART / 'vlm' / 'thresholds.json').read_text())['mismatch_threshold_t']
    assert round(t, 4) == 0.0596


def test_audit_counts():
    deletions = pd.read_csv(ART / 'audits' / 'retained_set_deletions.csv')
    for export, n, strict, loose in [('keep_one', 753, 266, 615), ('vl_dedup', 91, 1, 11), ('vl_dedup_no_check', 154, 18, 72)]:
        d = deletions[deletions.export == export]
        assert len(d) == n
        assert ((d.auditor_1 == 'lost') & (d.auditor_2 == 'lost')).sum() == strict
        assert ((d.auditor_1 == 'lost') | (d.auditor_2 == 'lost')).sum() == loose
    for name, tp, predicted, positive in [('caption_action_pairs', 163, 217, 269),
                                          ('caption_action_routing_pairs', 31, 50, 66)]:
        c = pd.read_csv(ART / 'audits' / f'{name}.csv')
        ref, pred = c.reference.eq('complementary'), c.vlm_verdict.eq('keep_augment')
        assert ((ref & pred).sum(), pred.sum(), ref.sum()) == (tp, predicted, positive)

    natural = pd.read_csv(ART / 'audits' / 'natural_errors.csv')
    wrong = lambda i, side: natural[f'auditor_{i}_{side}'].isin(['local_error', 'whole_mismatch'])
    strict = (wrong(1, 'a') & wrong(2, 'a')) | (wrong(1, 'b') & wrong(2, 'b'))
    lenient = wrong(1, 'a') | wrong(2, 'a') | wrong(1, 'b') | wrong(2, 'b')
    for arm, n, s, l in [('flagged', 59, 20, 27), ('control', 60, 5, 16)]:
        rows = natural.arm == arm
        assert (rows.sum(), (strict & rows).sum(), (lenient & rows).sum()) == (n, s, l)


def test_stage_cd_recomputes_clusters():
    pytest.importorskip('sklearn')
    module = load_script('recompute_stage_cd')
    assert module.main() == 0
