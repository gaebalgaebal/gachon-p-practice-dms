"""Standalone Windows-friendly YawDD interval labeling tool."""

import argparse
import csv
from pathlib import Path


INTERVAL_FIELDS = [
    "interval_id", "video_id", "file_name", "split", "start_s", "end_s",
    "label", "annotator",
]
REVIEW_FIELDS = [
    "video_id", "file_name", "split", "review_status", "yawn_count", "annotator",
]


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def verify_assignment(root: Path, rows):
    missing = []
    for row in rows:
        path = root / row["copied_file"]
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise ValueError("Missing assigned videos:\n" + "\n".join(missing))
    return len(rows)


def put_text(frame, lines):
    import cv2

    output = frame.copy()
    for index, (text, color) in enumerate(lines):
        cv2.putText(output, text, (16, 30 + index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .62, color, 2, cv2.LINE_AA)
    return output


def label(root: Path, assignment: Path, member: str):
    try:
        import cv2
    except ImportError as exc:
        raise ValueError("OpenCV is missing. Run setup_windows.bat first.") from exc

    assignments = read_csv(assignment)
    verify_assignment(root, assignments)
    results = root / "results"
    intervals_path = results / f"yawdd_{member}_intervals.csv"
    reviews_path = results / f"yawdd_{member}_review.csv"
    intervals = read_csv(intervals_path)
    reviews = read_csv(reviews_path)
    completed = {row["video_id"] for row in reviews if row["review_status"] == "complete"}
    pending = [row for row in assignments if row["sample_id"] not in completed]

    print(f"Assigned: {len(assignments)} | completed: {len(completed)} | pending: {len(pending)}")
    if not pending:
        print("All assigned videos are complete.")
        return

    for order, row in enumerate(pending, start=1):
        video_id = row["sample_id"]
        path = root / row["copied_file"]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps > 0 else 0
        paused, start_s, frame = False, None, None
        current_count = sum(item["video_id"] == video_id for item in intervals)

        while True:
            if not paused or frame is None:
                ok, next_frame = capture.read()
                if not ok:
                    paused = True
                    if frame is None:
                        break
                else:
                    frame = next_frame
            position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            marker = "unset" if start_s is None else f"{start_s:.2f}s"
            display = put_text(frame, [
                (f"{member}  {order}/{len(pending)}  {video_id}", (0, 255, 255)),
                (f"{position:.2f}/{duration:.2f}s  yawn start: {marker}", (0, 255, 0)),
                (f"saved yawns: {current_count}", (255, 255, 0)),
                ("SPACE play/pause | [ start | ] save end | U undo", (255, 255, 255)),
                ("A/D -/+1 sec | C complete | N no-yawn | Q quit", (255, 255, 255)),
            ])
            cv2.imshow("YawDD Labeler", display)
            key = cv2.waitKey(0 if paused else max(1, round(1000 / max(fps, 1)))) & 0xFF

            if key in (ord("q"), ord("Q")):
                capture.release()
                cv2.destroyAllWindows()
                write_csv(intervals_path, INTERVAL_FIELDS, intervals)
                write_csv(reviews_path, REVIEW_FIELDS, reviews)
                print("Progress saved. Run again to continue.")
                return
            if key == ord(" "):
                paused = not paused
            elif key == ord("["):
                start_s = position
            elif key == ord("]") and start_s is not None and position > start_s:
                current_count += 1
                intervals.append({
                    "interval_id": f"{video_id}:yawn:{current_count:03d}",
                    "video_id": video_id, "file_name": path.name, "split": row["split"],
                    "start_s": f"{start_s:.3f}", "end_s": f"{position:.3f}",
                    "label": "yawn", "annotator": member,
                })
                write_csv(intervals_path, INTERVAL_FIELDS, intervals)
                start_s = None
            elif key in (ord("u"), ord("U")):
                indices = [i for i, item in enumerate(intervals) if item["video_id"] == video_id]
                if indices:
                    intervals.pop(indices[-1])
                    current_count -= 1
                    write_csv(intervals_path, INTERVAL_FIELDS, intervals)
            elif key in (ord("a"), ord("A"), ord("d"), ord("D")):
                delta = -1 if key in (ord("a"), ord("A")) else 1
                target = max(0.0, min(duration, position + delta))
                capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000)
                ok, next_frame = capture.read()
                if ok:
                    frame = next_frame
                paused = True
            elif key in (ord("n"), ord("N")) and current_count == 0:
                reviews.append({
                    "video_id": video_id, "file_name": path.name, "split": row["split"],
                    "review_status": "complete", "yawn_count": 0, "annotator": member,
                })
                write_csv(reviews_path, REVIEW_FIELDS, reviews)
                break
            elif key in (ord("c"), ord("C")):
                reviews.append({
                    "video_id": video_id, "file_name": path.name, "split": row["split"],
                    "review_status": "complete", "yawn_count": current_count,
                    "annotator": member,
                })
                write_csv(reviews_path, REVIEW_FIELDS, reviews)
                break
        capture.release()

    cv2.destroyAllWindows()
    print("All assigned videos are complete. Send the entire results folder.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--member", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    assignment = root / "assignment.csv"
    rows = read_csv(assignment)
    if not rows:
        parser.error(f"Assignment not found or empty: {assignment}")
    try:
        count = verify_assignment(root, rows)
        if args.check:
            print(f"PASS: {count} assigned videos are present for member {args.member}.")
            return
        label(root, assignment, args.member)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
