"""Evaluate the OpenVINO eye classifier on an MRL Eye DMS manifest."""

import argparse
import csv
import json
from pathlib import Path

from dms.eye import OpenVINOEyeClassifier
from dms.metrics import classification_report
from dms.mrl_eye import label_from_scores


def resolve_file(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def subgroup_reports(rows, predictions, fields):
    reports = {}
    for field in fields:
        reports[field] = {}
        for value in sorted({row.get(field, "") for row in rows}):
            indices = [index for index, row in enumerate(rows) if row.get(field, "") == value]
            reports[field][value] = classification_report(
                [rows[index]["label"] for index in indices],
                [predictions[index] for index in indices], ["open", "closed"])
    return reports


def evaluate(manifest: Path, model: Path, split: str = "test", threshold: float = .70,
             device: str = "CPU", predictions_path: Path | None = None):
    import cv2

    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["dataset"] == "mrl_eye" and row["task"] == "eye_state"
                and row["split"] == split]
    if not rows:
        raise ValueError(f"No MRL Eye rows for split={split}")
    classifier = OpenVINOEyeClassifier(model, device)
    predictions, scored = [], []
    for row in rows:
        path = resolve_file(manifest, row["file_path"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        open_score, closed_score = classifier.predict_patch(image)
        predicted = label_from_scores(open_score, closed_score, threshold)
        predictions.append(predicted)
        scored.append({**row, "open_score": open_score, "closed_score": closed_score,
                       "prediction": predicted})
    report = classification_report([row["label"] for row in rows], predictions,
                                   ["open", "closed"])
    report["split"] = split
    report["threshold"] = threshold
    report["subgroups"] = subgroup_reports(
        rows, predictions, ["gender", "glasses", "reflections", "lighting", "sensor_id"])
    if predictions_path:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=scored[0].keys())
            writer.writeheader()
            writer.writerows(scored)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/open_closed_eye.onnx"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=.70)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(args.manifest, args.model, args.split, args.threshold,
                          args.device, args.predictions)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
