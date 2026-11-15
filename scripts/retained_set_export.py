"""Rebuild the export of VL-Dedup from the released scores, pair verdicts and coverage answers.

The export has three steps. It builds the export of each cluster from the pair actions, with a
representative that the record score does not flag. It deletes a record only if the VLM finds every
fact of its caption in the retained captions. It then deletes exact repeats among the retained
captions. Run it from the repository root:

    python -m scripts.retained_set_export
"""
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from scripts.retained_set_helpers import export_cluster
from scripts.run_combined_actions import pair_action

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
STATES = ('retained', 'held', 'deleted')


def normalize_caption(text):
    return re.sub(r'\s+', ' ', text.strip().lower()).rstrip(' .')


def flagged_representative_export(ids, clip, beta, decisions, threshold):
    """Export one cluster. A flagged record is never the representative and is held for review."""
    ids = sorted(ids)
    flagged = {r for r in ids if beta[r] < threshold}
    if flagged == set(ids):
        return dict(retained=[], held=ids, deleted=[])
    representative = min(set(ids) - flagged, key=lambda r: (-clip[r], r))
    actions = {pair: pair_action(pair, beta, decisions[pair], threshold) for pair in itertools.combinations(ids, 2)}
    arm = export_cluster(ids, representative, actions)['b']
    assert flagged <= set(arm['held'])
    return arm


def coverage_check(arm, clip, covered):
    """Check the deletion candidates one at a time, in decreasing order of CLIPScore.

    A candidate is deleted only when the VLM finds its caption covered by the retained captions.
    Otherwise it is retained, and later checks also compare with its caption.
    """
    result = {state: list(arm[state]) for state in STATES}
    result['deleted'] = []
    for record in sorted(arm['deleted'], key=lambda r: (-clip[r], r)):
        (result['deleted'] if covered(record, sorted(result['retained'])) is True else result['retained']).append(record)
    return {state: sorted(result[state]) for state in STATES}


def delete_exact_repeats(arm, clip, text):
    """Keep the record with the highest CLIPScore among retained records with one normalized caption."""
    groups = defaultdict(list)
    for record in arm['retained']:
        groups[normalize_caption(text[record])].append(record)
    kept = sorted(min(group, key=lambda r: (-clip[r], r)) for group in groups.values())
    removed = set(arm['retained']) - set(kept)
    return dict(retained=kept, held=list(arm['held']), deleted=sorted(set(arm['deleted']) | removed))


def load_inputs():
    scores = pd.read_csv(ART / 'vlm' / 'record_scores.csv.gz')
    clip = dict(zip(scores.record_id, scores.clipscore_alpha))
    beta = dict(zip(scores.record_id, scores.record_score_beta))
    threshold = json.loads((ART / 'vlm' / 'thresholds.json').read_text())['mismatch_threshold_t']
    records = pd.read_csv(ROOT / 'benchmark' / 'records.csv.gz', usecols=['record_id', 'caption'])
    text = dict(zip(records.record_id, records.caption))
    pairs = pd.read_csv(ART / 'vlm' / 'pair_verdicts.csv.gz', keep_default_na=False)
    checks = pd.read_csv(ART / 'vlm' / 'coverage_checks.csv.gz')
    answers = {(r, tuple(k.split())): bool(c) for r, k, c in
               zip(checks.record_id, checks.retained_records.fillna(''), checks.covered)}
    return clip, beta, threshold, text, pairs, answers


def rebuild():
    clip, beta, threshold, text, pairs, answers = load_inputs()
    groups = defaultdict(dict)
    for row in pairs.itertuples():
        pair = tuple(sorted((row.record_id_a, row.record_id_b)))
        groups[(row.partition, row.cluster)][pair] = row.verdict or None
    output = {}
    for decisions in groups.values():
        ids = sorted({record for pair in decisions for record in pair})
        arm = flagged_representative_export(ids, clip, beta, decisions, threshold)
        arm = coverage_check(arm, clip, lambda record, retained: answers.get((record, tuple(retained))))
        arm = delete_exact_repeats(arm, clip, text)
        output.update({r: 'retained' for r in arm['retained']})
        output.update({r: 'held' for r in arm['held']})
        output.update({r: 'deleted' for r in arm['deleted']})
    return output


def main():
    released = pd.read_csv(ART / 'exports' / 'retained_set_vl_dedup.csv.gz')
    rebuilt = rebuild()
    inside = released[released.record_id.isin(rebuilt)]
    differ = int((inside.record_id.map(rebuilt) != inside.status).sum())
    print(f'{len(inside)} records in clusters, {differ} differ from the released export')
    print(inside.groupby(['partition', 'status']).size().to_string())


if __name__ == '__main__':
    main()
