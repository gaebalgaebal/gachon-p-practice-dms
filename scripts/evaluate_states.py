"""Score manually labeled frame indices against live JSONL, including unknowns."""

import argparse
import csv
import json
from pathlib import Path

from dms.metrics import classification_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True,
                        help='CSV: frame_index,drowsiness,attention. Blank task labels are unevaluated.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('Output already exists')
        predictions = {}
        for line in args.predictions.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['frame_index'] in predictions:
                raise ValueError('Duplicate predicted frame index')
            predictions[row['frame_index']] = row
        with args.labels.open(encoding='utf-8-sig', newline='') as handle:
            labels = list(csv.DictReader(handle))
        seen = set()
        for row in labels:
            index = int(row['frame_index'])
            if index in seen:
                raise ValueError('Duplicate labeled frame index')
            if index not in predictions:
                raise ValueError(f'Missing prediction for labeled frame {index}')
            seen.add(index)
        report = {'note': 'Frame-level metrics, not event-level false alarms or clinical 3-state drowsiness.'}
        for task, values in [('drowsiness', ['normal', 'suspected']),
                             ('attention', ['forward', 'off_forward'])]:
            selected = [row for row in labels if row.get(task, '').strip()]
            if not selected:
                continue
            truth = [row[task].strip() for row in selected]
            predicted = [predictions[int(row['frame_index'])][task] for row in selected]
            report[task] = classification_report(truth, predicted, values)
            if task == 'drowsiness':
                baseline = [predictions[int(row['frame_index'])]['baseline_drowsiness'] for row in selected]
                report['single_frame_ear_baseline'] = classification_report(truth, baseline, values)
        if len(report) == 1:
            raise ValueError('No task labels to evaluate')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Saved frame-level evaluation to {args.output}')
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f'Evaluation error: {exc}\n')


if __name__ == '__main__':
    main()
