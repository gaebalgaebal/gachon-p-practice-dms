"""Interactively record YawDD yawn start/end intervals in a CSV file."""

import argparse
import csv
from pathlib import Path

from dms.yawdd import validate_interval


FIELDS = ["interval_id", "video_id", "file_path", "start_s", "end_s",
          "label", "annotator", "notes"]


def append_interval(output: Path, video_id: str, file_path: str, start_s: float,
                    end_s: float, duration_s: float, annotator: str = "local",
                    notes: str = "") -> dict:
    validate_interval(start_s, end_s, duration_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if output.exists():
        with output.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    sequence = 1 + sum(row.get("video_id") == video_id for row in existing)
    row = {"interval_id": f"{video_id}:yawn:{sequence:03d}", "video_id": video_id,
           "file_path": file_path, "start_s": f"{start_s:.3f}",
           "end_s": f"{end_s:.3f}", "label": "yawn",
           "annotator": annotator, "notes": notes}
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    return row


def _load_video(manifest: Path, video_id: str) -> tuple[Path, float]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    if video_id not in rows:
        raise ValueError(f"Video ID not found in manifest: {video_id}")
    row = rows[video_id]
    path = (manifest.parent / row["file_path"]).resolve()
    return path, float(row["duration_s"])


def label_video(manifest: Path, video_id: str, output: Path, annotator: str) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ValueError("OpenCV is required: install the project vision extra") from exc
    path, duration_s = _load_video(manifest, video_id)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    paused, start_s = False, None
    print("Keys: SPACE pause | [ start | ] end+save | a/d -/+1s | q quit")
    while True:
        if not paused:
            ok, frame = capture.read()
            if not ok:
                break
        position_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        display = frame.copy()
        cv2.putText(display, f"{video_id}  {position_s:.2f}/{duration_s:.2f}s",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
        marker = "unset" if start_s is None else f"{start_s:.2f}s"
        cv2.putText(display, f"yawn start: {marker}", (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
        cv2.imshow("YawDD interval labeler", display)
        key = cv2.waitKey(0 if paused else max(1, round(1000 / fps))) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("["):
            start_s = position_s
        elif key == ord("]") and start_s is not None:
            row = append_interval(output, video_id, str(path), start_s,
                                  position_s, duration_s, annotator)
            print(f"Saved {row['interval_id']}: {start_s:.3f}-{position_s:.3f}s")
            start_s = None
        elif key in (ord("a"), ord("d")):
            target = max(0.0, min(duration_s, position_s + (-1 if key == ord("a") else 1)))
            capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000)
            ok, frame = capture.read()
            paused = True
            if not ok:
                break
    capture.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("video_id")
    parser.add_argument("--output", type=Path, default=Path("data/annotations/yawdd_yawn_intervals.csv"))
    parser.add_argument("--annotator", default="local")
    args = parser.parse_args()
    try:
        label_video(args.manifest, args.video_id, args.output, args.annotator)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
