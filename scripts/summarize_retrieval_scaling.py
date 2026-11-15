import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
from pathlib import Path
from statistics import median
METRICS = ['stage_wall_seconds', 'stage_cpu_seconds', 'process_wall_seconds', 'process_cpu_seconds', 'peak_process_rss_gib', 'candidate_edges']

def summarize(trials):
    rows = list(csv.DictReader(Path(trials).open()))
    expected = {(n, r) for n in [10000, 20000, 30000, 50000] for r in [1, 2, 3]}
    found = [(int(x['records']), int(x['repeat'])) for x in rows]
    if len(found) != 12 or set(found) != expected:
        raise ValueError('Expected all 12 distinct completed measurements; do not fill or discard repeats')
    results = []
    for size in [10000, 20000, 30000, 50000]:
        group = [x for x in rows if int(x['records']) == size]
        record = {'records': size, 'repeats': len(group)}
        for metric in METRICS:
            values = [float(x[metric]) for x in group]
            record.update({metric + '_median': median(values), metric + '_min': min(values), metric + '_max': max(values)})
        results.append(record)
    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.trials)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
if __name__ == '__main__':
    main()
