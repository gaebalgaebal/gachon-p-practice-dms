"""Select an eye confidence threshold from saved validation predictions."""

import argparse
import csv
import json
from pathlib import Path

from dms.metrics import classification_report
from dms.mrl_eye import label_from_scores


def evaluate_thresholds(rows, start: float = .50, stop: float = .95,
                        step: float = .01):
    if step <= 0 or start < 0 or stop > 1 or start > stop:
        raise ValueError("Require 0 <= start <= stop <= 1 and a positive step")
    truth = [row["label"] for row in rows]
    thresholds = []
    count = round((stop - start) / step)
    for index in range(count + 1):
        threshold = round(start + index * step, 10)
        predicted = [label_from_scores(float(row["open_score"]),
                                       float(row["closed_score"]), threshold)
                     for row in rows]
        report = classification_report(truth, predicted, ["open", "closed"])
        thresholds.append({"threshold": threshold, **report})
    selected = max(thresholds, key=lambda row: (
        row["macro_f1"], row["accuracy"], row["coverage"], -row["threshold"]))
    return {
        "selection_split": "validation",
        "selection_rule": "maximize macro_f1, then accuracy, coverage, lower threshold",
        "selected_threshold": selected["threshold"],
        "selected_report": selected,
        "thresholds": thresholds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--start", type=float, default=.50)
    parser.add_argument("--stop", type=float, default=.95)
    parser.add_argument("--step", type=float, default=.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.predictions.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error("Predictions CSV is empty")
    try:
        result = evaluate_thresholds(rows, args.start, args.stop, args.step)
    except ValueError as exc:
        parser.error(str(exc))
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
