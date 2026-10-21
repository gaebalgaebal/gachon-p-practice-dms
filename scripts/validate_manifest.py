"""Reject missing splits and cross-split driver/session leakage in a CSV manifest."""

import argparse
import csv
from pathlib import Path


def validate(path: Path) -> int:
    owners = {"subject_id": {}, "session_id": {}}
    seen_samples, splits = set(), set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"sample_id", "subject_id", "session_id", "split", "label"}
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
    args = parser.parse_args()
    try:
        count = validate(args.manifest)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"PASS: {count} samples; subject/session splits are disjoint.")


if __name__ == "__main__":
    main()
