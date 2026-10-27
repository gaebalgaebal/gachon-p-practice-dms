"""Evaluate labeled state events against frame JSONL without hiding false alarms."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class Event:
    task: str
    label: str
    start_s: float
    end_s: float


def predicted_events(rows, task, label):
    events = []
    start = None
    previous_t = None
    for row in rows:
        t = float(row["timestamp_s"])
        active = row.get(task) == label
        if active and start is None:
            start = t
        elif not active and start is not None:
            events.append(Event(task, label, start, t))
            start = None
        previous_t = t
    if start is not None and previous_t is not None:
        events.append(Event(task, label, start, previous_t))
    return events


def match_events(truth, predicted, tolerance_s=0.5):
    candidates = []
    for ti, target in enumerate(truth):
        for pi, guess in enumerate(predicted):
            overlap = min(target.end_s, guess.end_s) - max(target.start_s, guess.start_s)
            distance = max(target.start_s - guess.end_s, guess.start_s - target.end_s, 0)
            if overlap >= 0 or distance <= tolerance_s:
                candidates.append((max(overlap, 0), -distance, ti, pi))
    used_truth, used_predictions, matches = set(), set(), []
    for _, _, ti, pi in sorted(candidates, reverse=True):
        if ti in used_truth or pi in used_predictions:
            continue
        used_truth.add(ti)
        used_predictions.add(pi)
        matches.append((truth[ti], predicted[pi]))
    return matches, len(truth) - len(matches), len(predicted) - len(matches)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def evaluate(rows, truth, tolerance_s=0.5):
    if not rows:
        raise ValueError("Prediction log is empty")
    duration_s = float(rows[-1]["timestamp_s"]) - float(rows[0]["timestamp_s"])
    if duration_s <= 0:
        raise ValueError("Prediction log must span a positive duration")
    groups = sorted({(event.task, event.label) for event in truth})
    result = {"duration_s": duration_s, "match_tolerance_s": tolerance_s, "groups": {}}
    for task, label in groups:
        targets = [e for e in truth if e.task == task and e.label == label]
        guesses = predicted_events(rows, task, label)
        matches, missed, false_alarms = match_events(targets, guesses, tolerance_s)
        latencies = [guess.start_s - target.start_s for target, guess in matches]
        result["groups"][f"{task}:{label}"] = {
            "truth_events": len(targets),
            "predicted_events": len(guesses),
            "matched_events": len(matches),
            "missed_events": missed,
            "false_alarm_events": false_alarms,
            "event_recall": len(matches) / len(targets) if targets else None,
            "event_precision": len(matches) / len(guesses) if guesses else None,
            "false_alarms_per_minute": false_alarms / (duration_s / 60),
            "detection_latency_s_mean": mean(latencies) if latencies else None,
            "detection_latency_s_p95": percentile(latencies, .95),
            "latency_note": "Signed latency: negative values indicate an early prediction.",
        }
    return result


def load_truth(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"task", "label", "start_s", "end_s"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Event CSV requires task,label,start_s,end_s")
    events = []
    for row in rows:
        start, end = float(row["start_s"]), float(row["end_s"])
        if not row["task"].strip() or not row["label"].strip() or start < 0 or end <= start:
            raise ValueError(f"Invalid event row: {row}")
        events.append(Event(row["task"].strip(), row["label"].strip(), start, end))
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-tolerance", type=float, default=.5)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output already exists")
        if args.match_tolerance < 0:
            raise ValueError("Match tolerance must be nonnegative")
        rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
        report = evaluate(rows, load_truth(args.labels), args.match_tolerance)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved event-level evaluation to {args.output}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Event evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
