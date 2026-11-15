"""VERBATIM_EXPORT_RUN deterministic verbatim export and frozen removal-budget contracts."""
import random

import pytest

from scripts.experiments.run_verbatim_export import matched_controls, norm, verbatim


def test_normalization_is_exact_brief_rule():
    assert norm('  A\tCAT . ') == 'a cat'
    assert norm('Ａ, Cat!') == 'ａ, cat!'


def test_highest_clip_then_smallest_id_with_surviving_witnesses():
    ids=['a','b','c','d']
    arm=dict(retained=ids,pending=[],deleted=[])
    text=dict(a=' A CAT. ',b='a cat',c='a cat',d='different')
    after,w=verbatim(ids,arm,dict(a=.1,b=.9,c=.9,d=.2),text)
    assert after==dict(retained=['b','d'],pending=[],deleted=['a','c'])
    assert w==dict(a='b',c='b')
    assert arm['retained']==ids
    assert {norm(text[r]) for r in arm['retained']}=={norm(text[r]) for r in after['retained']}


def test_only_retained_changes_and_strict_is_separate():
    ids=['a','b','c','d','e']
    arm=dict(retained=['a','b','c'],pending=['d'],deleted=['e'])
    text=dict(a='cat.',b='cat',c='cat',d='cat',e='cat')
    clip=dict.fromkeys(ids,1.)
    after,w=verbatim(ids,arm,clip,text)
    strict,sw=verbatim(ids,arm,clip,text,strict=True)
    assert after==dict(retained=['a'],pending=['d'],deleted=['b','c','e'])
    assert strict==dict(retained=['a','b'],pending=['d'],deleted=['c','e'])
    assert set(w)=={'b','c'} and sw=={'c':'b'}


@pytest.mark.parametrize('state',['retained','pending','deleted'])
def test_singleton_and_state_partition(state):
    arm={s:['a'] if s==state else [] for s in ('retained','pending','deleted')}
    result,w=verbatim(['a'],arm,{'a':1},{'a':'hello'})
    assert result==arm and not w
    flattened=sum(result.values(),[])
    assert len(flattened)==len(set(flattened))==1 and set(flattened)=={'a'}


def test_empty_retained_is_unchanged():
    arm=dict(retained=[],pending=['a'],deleted=['b'])
    assert verbatim(['a','b'],arm,dict(a=1,b=2),dict(a='x',b='x'))==(arm,{})


def test_frozen_pool_controls_have_exact_removal_budget():
    ids=[f'r{i:05}' for i in range(35208)]
    kept=18556;removed=len(ids)-kept
    clip=dict.fromkeys(ids,1.)
    controls=matched_controls(list(reversed(ids)),clip,removed)
    assert all(len(v)==kept and len(set(ids)-set(v))==removed for v in controls.values())
    assert controls['clipscore_vb']==ids[removed:]
    expected=set(random.Random(20261107).sample(ids,removed))
    assert set(ids)-set(controls['random_vb'])==expected
