"""Create a subject-disjoint video manifest for UTA-RLDD Fold directories."""

import argparse
import csv
import importlib.util
import os
from pathlib import Path

from dms.uta_rldd import parse_uta_path, split_subjects


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}
FIELDS = ["sample_id", "dataset", "subject_id", "session_id", "file_path",
          "frame_index", "timestamp_s", "task", "label", "split", "fold",
          "part", "level", "segment", "fps", "frame_count", "duration_s",
          "width", "height"]


def _video_metadata(path: Path) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise ValueError("OpenCV is required: install the project vision extra") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open UTA-RLDD video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return {"fps": fps, "frame_count": frames, "duration_s": frames / fps,
            "width": width, "height": height}


def _validate(path: Path) -> int:
    validator_path = Path(__file__).with_name("validate_manifest.py")
    spec = importlib.util.spec_from_file_location("dms_manifest_validator", validator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(path, "dms")


def build_manifest(source: Path, output: Path, train_ratio: float = .70,
                   val_ratio: float = .15, seed: int = 42,
                   force: bool = False) -> int:
    if not source.is_dir():
        raise ValueError(f"UTA-RLDD directory not found: {source}")
    if output.exists() and not force:
        raise ValueError(f"Output exists; pass --force to replace it: {output}")
    paths = sorted(path for path in source.rglob("*")
                   if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    if not paths:
        raise ValueError(f"No UTA-RLDD videos found below: {source}")
    parsed = [(path, parse_uta_path(path)) for path in paths]
    assignments = split_subjects((item.subject_id for _, item in parsed),
                                 train_ratio, val_ratio, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for path, item in parsed:
            metadata = _video_metadata(path)
            session = f"{item.subject_id}:level_{item.level}"
            writer.writerow({
                "sample_id": f"{session}:segment_{item.segment}",
                "dataset": "uta_rldd", "subject_id": item.subject_id,
                "session_id": session,
                "file_path": Path(os.path.relpath(path.resolve(), output.parent.resolve())).as_posix(),
                "frame_index": 0, "timestamp_s": 0, "task": "drowsiness",
                "label": item.label, "split": assignments[item.subject_id],
                "fold": item.fold, "part": item.part, "level": item.level,
                "segment": item.segment,
                **{key: f"{value:.6f}" if isinstance(value, float) else value
                   for key, value in metadata.items()},
            })
    return _validate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/uta_rldd_videos.csv"))
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
    print(f"PASS: wrote {count} UTA-RLDD videos to {args.output}")


if __name__ == "__main__":
    main()
