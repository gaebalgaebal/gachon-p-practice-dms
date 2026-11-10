"""Evaluate video-level UTA-RLDD predictions against a DMS manifest."""

import argparse
import csv
import json
from pathlib import Path

from dms.metrics import classification_report
from dms.uta_rldd import binary_label


LABELS = ["alert", "low_vigilance", "drowsy"]


def load_truth(manifest: Path, split: str) -> list[dict]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["dataset"] == "uta_rldd" and row["task"] == "drowsiness"
                and row["split"] == split]
    if not rows:
        raise ValueError(f"No UTA-RLDD rows for split={split}")
    return rows


def write_template(manifest: Path, output: Path, split: str = "test") -> int:
    rows = load_truth(manifest, split)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "prediction"])
        writer.writeheader()
        writer.writerows({"sample_id": row["sample_id"], "prediction": "unknown"}
                         for row in rows)
    return len(rows)


def _read_predictions(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"sample_id", "prediction"}.issubset(reader.fieldnames or []):
            raise ValueError("Predictions require sample_id,prediction columns")
        result = {}
        for line, row in enumerate(reader, start=2):
            sample_id, prediction = row["sample_id"].strip(), row["prediction"].strip()
            if sample_id in result:
                raise ValueError(f"Duplicate prediction at line {line}: {sample_id}")
            if prediction not in LABELS + ["unknown"]:
                raise ValueError(f"Invalid prediction at line {line}: {prediction}")
            result[sample_id] = prediction
    return result


def evaluate(manifest: Path, predictions_path: Path, split: str = "test") -> dict:
    rows = load_truth(manifest, split)
    predictions_by_id = _read_predictions(predictions_path)
    expected = {row["sample_id"] for row in rows}
    missing = sorted(expected - predictions_by_id.keys())
    extra = sorted(predictions_by_id.keys() - expected)
    if missing or extra:
        raise ValueError(f"Prediction IDs mismatch: missing={len(missing)}, extra={len(extra)}")
    truth = [row["label"] for row in rows]
    predicted = [predictions_by_id[row["sample_id"]] for row in rows]
    three_class = classification_report(truth, predicted, LABELS)
    binary = classification_report([binary_label(value) for value in truth],
                                   [binary_label(value) for value in predicted],
                                   ["alert", "fatigued"])
    return {
        "dataset": "uta_rldd", "split": split,
        "three_class": three_class, "binary_alert_vs_fatigued": binary,
        "binary_policy": "low_vigilance and drowsy are mapped to fatigued",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("predictions", type=Path, nargs="?")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--write-template", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.write_template:
            count = write_template(args.manifest, args.write_template, args.split)
            print(f"PASS: wrote {count} prediction rows to {args.write_template}")
            return
        if not args.predictions:
            parser.error("predictions is required unless --write-template is used")
        report = evaluate(args.manifest, args.predictions, args.split)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
