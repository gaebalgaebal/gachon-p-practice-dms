"""Validate generic or project-wide DMS CSV manifests."""

import argparse
import csv
import json
import math
from pathlib import Path


BASE_REQUIRED = {"sample_id", "subject_id", "session_id", "split", "label"}
DMS_REQUIRED = BASE_REQUIRED | {
    "dataset", "file_path", "frame_index", "timestamp_s", "task"
}


def load_dataset_tasks() -> dict[str, set[str]]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "datasets.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    result = {}
    for phase in ("initial", "later"):
        for dataset, details in payload[phase].items():
            result[dataset] = set(details["tasks"])
    return result


def validate(path: Path, profile: str = "base") -> int:
    if profile not in {"base", "dms"}:
        raise ValueError(f"Unknown manifest profile: {profile}")
    required = DMS_REQUIRED if profile == "dms" else BASE_REQUIRED
    dataset_tasks = load_dataset_tasks() if profile == "dms" else {}
    owners = {"subject_id": {}, "session_id": {}}
    seen_samples, splits = set(), set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not required.issubset(rows.fieldnames or []):
            raise ValueError(f"Required columns: {', '.join(sorted(required))}")
        for line, row in enumerate(rows, start=2):
            if any(not row.get(key) or not row[key].strip() for key in required):
                raise ValueError(f"Line {line}: missing value")
            row = {key: value.strip() for key, value in row.items() if key in required}
            split = row["split"]
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Line {line}: invalid split {split}")
            sample = row["sample_id"]
            if sample in seen_samples:
                raise ValueError(f"Line {line}: duplicate sample_id {sample}")
            seen_samples.add(sample)
            splits.add(split)
            if profile == "dms":
                dataset = row["dataset"]
                task = row["task"]
                if dataset not in dataset_tasks:
                    raise ValueError(f"Line {line}: unsupported dataset {dataset}")
                if task not in dataset_tasks[dataset]:
                    raise ValueError(
                        f"Line {line}: task {task} is not configured for dataset {dataset}"
                    )
                try:
                    frame_index = int(row["frame_index"])
                    timestamp_s = float(row["timestamp_s"])
                except ValueError as exc:
                    raise ValueError(f"Line {line}: invalid frame_index or timestamp_s") from exc
                if frame_index < 0 or not math.isfinite(timestamp_s) or timestamp_s < 0:
                    raise ValueError(f"Line {line}: frame_index/timestamp_s must be non-negative")
            for key, mapping in owners.items():
                identity = row[key]
                if identity in mapping and mapping[identity] != split:
                    raise ValueError(f"Line {line}: cross-split {key} leakage: {identity}")
                mapping[identity] = split
    if splits != {"train", "val", "test"}:
        raise ValueError("Manifest must contain train, val and test")
    return len(seen_samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--profile", choices=("base", "dms"), default="base")
    args = parser.parse_args()
    try:
        count = validate(args.manifest, args.profile)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"PASS: {count} {args.profile} samples; subject/session splits are disjoint.")


if __name__ == "__main__":
    main()
