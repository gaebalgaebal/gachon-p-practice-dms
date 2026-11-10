"""Convert the official MRL Eye directory into the common DMS manifest."""

import argparse
import csv
import importlib.util
import os
from pathlib import Path

from dms.mrl_eye import parse_mrl_filename, split_subjects


FIELDS = ["sample_id", "dataset", "subject_id", "session_id", "file_path",
          "frame_index", "timestamp_s", "task", "label", "split", "gender",
          "glasses", "reflections", "lighting", "sensor_id"]


def validate_dms_manifest(path: Path) -> int:
    """Load the sibling CLI validator without depending on the current directory."""
    validator_path = Path(__file__).with_name("validate_manifest.py")
    spec = importlib.util.spec_from_file_location("dms_manifest_validator", validator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(path, "dms")


def build_manifest(source: Path, output: Path, train_ratio: float = .70,
                   val_ratio: float = .15, seed: int = 42, force: bool = False) -> int:
    if not source.is_dir():
        raise ValueError(f"MRL Eye directory not found: {source}")
    if output.exists() and not force:
        raise ValueError(f"Output exists; pass --force to replace it: {output}")
    paths = sorted(path for path in source.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise ValueError(f"No eye images found below: {source}")
    parsed = [(path, parse_mrl_filename(path)) for path in paths]
    assignments = split_subjects((item.subject_id for _, item in parsed),
                                 train_ratio, val_ratio, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for path, item in parsed:
            writer.writerow({
                "sample_id": f"mrl_eye:{path.stem}", "dataset": "mrl_eye",
                "subject_id": item.subject_id,
                "session_id": f"mrl_eye:{item.subject_id}:sensor_{item.sensor_id}",
                "file_path": Path(os.path.relpath(path.resolve(), output.parent.resolve())).as_posix(),
                "frame_index": 0, "timestamp_s": 0, "task": "eye_state",
                "label": item.label, "split": assignments[item.subject_id],
                "gender": item.gender, "glasses": item.glasses,
                "reflections": item.reflections, "lighting": item.lighting,
                "sensor_id": item.sensor_id,
            })
    return validate_dms_manifest(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/mrl_eye.csv"))
    parser.add_argument("--train-ratio", type=float, default=.70)
    parser.add_argument("--val-ratio", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        count = build_manifest(args.source, args.output, args.train_ratio,
                               args.val_ratio, args.seed, args.force)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"PASS: wrote {count} MRL Eye samples to {args.output}")


if __name__ == "__main__":
    main()
