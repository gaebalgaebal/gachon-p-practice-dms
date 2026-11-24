"""Extract clean scheduled webcam frames into a subject-disjoint action manifest."""

import argparse
import csv
import importlib.util
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from dms.action_dataset import ACTION_LABELS, sample_times, validate_sessions
from dms.live import load_label_schedule


FIELDS = ["sample_id", "dataset", "subject_id", "session_id", "file_path",
          "frame_index", "timestamp_s", "task", "label", "split", "source_video"]


def _validate(path: Path) -> int:
    script = Path(__file__).with_name("validate_manifest.py")
    spec = importlib.util.spec_from_file_location("dms_manifest_validator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(path, "dms")


def build_dataset(config: Path, output: Path, manifest: Path, sample_fps: float = 2,
                  jpeg_quality: int = 95, force: bool = False) -> dict:
    import cv2

    sessions = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(sessions, list):
        raise ValueError("Session config must be a JSON list")
    validate_sessions(sessions)
    if output.exists():
        if not force:
            raise ValueError(f"Output exists; pass --force to replace it: {output}")
        shutil.rmtree(output)
    if manifest.exists() and not force:
        raise ValueError(f"Manifest exists; pass --force to replace it: {manifest}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1,100]")

    rows = []
    config_dir = config.resolve().parent
    for item in sessions:
        video = Path(item["video"])
        schedule_path = Path(item["schedule"])
        video = video if video.is_absolute() else config_dir / video
        schedule_path = schedule_path if schedule_path.is_absolute() else config_dir / schedule_path
        if not video.is_file() or not schedule_path.is_file():
            raise ValueError(f"Missing video or schedule for {item['session_id']}")
        schedule = load_label_schedule(schedule_path)
        capture = cv2.VideoCapture(str(video))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not source_fps or source_fps <= 0:
            capture.release()
            raise ValueError(f"Unreadable video or FPS: {video}")
        try:
            for sample_index, (timestamp, label) in enumerate(sample_times(schedule, sample_fps)):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Cannot read {video} at {timestamp:.3f}s")
                stem = f"{item['session_id']}_{sample_index:06d}"
                image = output / "images" / item["split"] / label / f"{stem}.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    raise OSError(f"Cannot write {image}")
                rows.append({
                    "sample_id": f"webcam_action:{stem}", "dataset": "webcam_action",
                    "subject_id": f"webcam_action:{item['subject_id']}",
                    "session_id": f"webcam_action:{item['session_id']}",
                    "file_path": Path(os.path.relpath(image.resolve(), manifest.parent.resolve())).as_posix(),
                    "frame_index": round(timestamp * source_fps), "timestamp_s": f"{timestamp:.6f}",
                    "task": "action", "label": label, "split": item["split"],
                    "source_video": video.name,
                })
        finally:
            capture.release()
    if not rows:
        raise ValueError("No normal/phone/drink frames were selected")
    for split in ("train", "val", "test"):
        labels = {row["label"] for row in rows if row["split"] == split}
        expected = {label for label in ACTION_LABELS if any(row["label"] == label for row in rows)}
        if labels != expected:
            raise ValueError(f"Every selected class must appear in {split}: {labels} != {expected}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    count = _validate(manifest)
    summary = {
        "samples": count, "sessions": len(sessions),
        "subjects": len({item["subject_id"] for item in sessions}),
        "splits": dict(Counter(row["split"] for row in rows)),
        "labels": dict(Counter(row["label"] for row in rows)),
        "sample_fps": sample_fps,
        "privacy": "raw videos and extracted face images are local-only and Git-ignored",
    }
    manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/webcam_action"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/webcam_action.csv"))
    parser.add_argument("--sample-fps", type=float, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = build_dataset(args.config, args.output, args.manifest, args.sample_fps,
                               args.jpeg_quality, args.force)
    except (OSError, ValueError, json.JSONDecodeError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
