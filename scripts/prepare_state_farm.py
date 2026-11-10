"""Create a driver-disjoint normal/phone/drink manifest from State Farm."""

import argparse
import csv
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

from dms.state_farm import map_source_class, split_subjects


FIELDS = ["sample_id", "dataset", "subject_id", "session_id", "file_path",
          "frame_index", "timestamp_s", "task", "label", "split",
          "source_class", "source_activity"]


def _validate(path: Path) -> int:
    validator_path = Path(__file__).with_name("validate_manifest.py")
    spec = importlib.util.spec_from_file_location("dms_manifest_validator", validator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(path, "dms")


def build_manifest(source: Path, output: Path, train_ratio: float = .70,
                   val_ratio: float = .15, seed: int = 42,
                   force: bool = False) -> dict:
    csv_path = source / "driver_imgs_list.csv"
    image_root = source / "imgs" / "train"
    if not csv_path.is_file() or not image_root.is_dir():
        raise ValueError(f"State Farm CSV/images not found below: {source}")
    if output.exists() and not force:
        raise ValueError(f"Output exists; pass --force to replace it: {output}")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    required = {"subject", "classname", "img"}
    if not source_rows or not required.issubset(source_rows[0]):
        raise ValueError("driver_imgs_list.csv requires subject,classname,img")
    assignments = split_subjects((row["subject"] for row in source_rows),
                                 train_ratio, val_ratio, seed)
    rows, excluded = [], Counter()
    for row in source_rows:
        mapped = map_source_class(row["classname"])
        if mapped.action_label is None:
            excluded[row["classname"]] += 1
            continue
        path = image_root / row["classname"] / row["img"]
        if not path.is_file():
            raise ValueError(f"Missing State Farm image: {path}")
        subject = f"state_farm:{row['subject']}"
        rows.append({
            "sample_id": f"state_farm:{row['img'].rsplit('.', 1)[0]}",
            "dataset": "state_farm", "subject_id": subject,
            "session_id": f"{subject}:controlled", "file_path": path,
            "frame_index": 0, "timestamp_s": 0, "task": "action",
            "label": mapped.action_label, "split": assignments[row["subject"]],
            "source_class": mapped.source_class,
            "source_activity": mapped.source_activity,
        })
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate State Farm image IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda value: value["sample_id"]):
            row = dict(row)
            row["file_path"] = Path(os.path.relpath(
                Path(row["file_path"]).resolve(), output.parent.resolve())).as_posix()
            writer.writerow(row)
    count = _validate(output)
    summary = {
        "samples": count, "subjects": len(assignments),
        "split_subjects": {split: sum(value == split for value in assignments.values())
                           for split in ("train", "val", "test")},
        "split_samples": dict(Counter(row["split"] for row in rows)),
        "labels": dict(Counter(row["label"] for row in rows)),
        "excluded_source_classes": dict(sorted(excluded.items())),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("data/manifests/state_farm_actions.csv"))
    parser.add_argument("--train-ratio", type=float, default=.70)
    parser.add_argument("--val-ratio", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        summary = build_manifest(args.source, args.output, args.train_ratio,
                                 args.val_ratio, args.seed, args.force)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
