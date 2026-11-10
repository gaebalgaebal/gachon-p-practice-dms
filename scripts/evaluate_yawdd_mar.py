"""Extract MediaPipe MAR and select/evaluate YawDD yawn event thresholds."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from dms.metrics import classification_report
from dms.vision import MediaPipeFaceAnalyzer


SAMPLE_FIELDS = ["video_id", "split", "timestamp_s", "mar", "valid", "reason"]


def load_inputs(manifest_path: Path, labels_path: Path):
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    if not labels:
        raise ValueError("YawDD interval labels are empty")
    video_ids = sorted({row["video_id"] for row in labels})
    if any(video_id not in manifest for video_id in video_ids):
        raise ValueError("Interval labels contain a video absent from the manifest")
    intervals = defaultdict(list)
    for row in labels:
        intervals[row["video_id"]].append((float(row["start_s"]), float(row["end_s"])))
    return manifest, intervals, video_ids


def extract_samples(manifest_path: Path, labels_path: Path, model_path: Path,
                    output: Path, sample_fps: float = 5.0) -> int:
    if not 0 < sample_fps <= 30:
        raise ValueError("sample_fps must be in (0, 30]")
    try:
        import cv2
    except ImportError as exc:
        raise ValueError("OpenCV is required") from exc
    manifest, _, video_ids = load_inputs(manifest_path, labels_path)
    analyzer = MediaPipeFaceAnalyzer(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
            for video_index, video_id in enumerate(video_ids, start=1):
                item = manifest[video_id]
                video_path = (manifest_path.parent / item["file_path"]).resolve()
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise ValueError(f"Cannot open video: {video_path}")
                source_fps = float(capture.get(cv2.CAP_PROP_FPS))
                step = max(1, round(source_fps / sample_fps))
                frame_index = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_index % step == 0:
                        timestamp_s = frame_index / source_fps
                        observation = analyzer.analyze(frame, count / sample_fps)
                        writer.writerow({
                            "video_id": video_id, "split": item["split"],
                            "timestamp_s": f"{timestamp_s:.3f}",
                            "mar": "" if observation.mar is None else f"{observation.mar:.6f}",
                            "valid": int(observation.valid), "reason": observation.reason,
                        })
                        count += 1
                    frame_index += 1
                capture.release()
                print(f"[{video_index}/{len(video_ids)}] {video_id}", flush=True)
    finally:
        analyzer.close()
    return count


def load_samples(path: Path) -> dict[str, list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("MAR sample CSV is empty")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append({
            "split": row["split"], "t": float(row["timestamp_s"]),
            "mar": float(row["mar"]) if row["mar"] else None,
        })
    return grouped


def predict(samples: list[dict], high: float, low: float,
            min_duration_s: float) -> tuple[list[str], list[tuple[float, float]]]:
    if not 0 < low < high < 2 or min_duration_s <= 0:
        raise ValueError("Require 0 < low < high < 2 and positive duration")
    states, events = [], []
    candidate, active_start = None, None
    previous_t = samples[0]["t"] if samples else 0.0
    for row in samples:
        t, mar = row["t"], row["mar"]
        if mar is None:
            states.append("unknown")
            if active_start is not None:
                events.append((active_start, previous_t))
            candidate, active_start = None, None
        elif active_start is not None:
            if mar <= low:
                events.append((active_start, t))
                active_start, candidate = None, None
                states.append("normal")
            else:
                states.append("yawn")
        elif mar >= high:
            candidate = t if candidate is None else candidate
            if t - candidate >= min_duration_s:
                active_start = candidate
                states.append("yawn")
            else:
                states.append("normal")
        else:
            candidate = None
            states.append("normal")
        previous_t = t
    if active_start is not None:
        events.append((active_start, previous_t))
    return states, events


def _truth_at(t: float, intervals: list[tuple[float, float]]) -> str:
    return "yawn" if any(start <= t <= end for start, end in intervals) else "normal"


def _match(truth, predicted, tolerance_s=.5):
    candidates = []
    for ti, target in enumerate(truth):
        for pi, guess in enumerate(predicted):
            overlap = min(target[1], guess[1]) - max(target[0], guess[0])
            distance = max(target[0] - guess[1], guess[0] - target[1], 0)
            if overlap >= 0 or distance <= tolerance_s:
                candidates.append((max(overlap, 0), -distance, ti, pi))
    used_t, used_p = set(), set()
    for _, _, ti, pi in sorted(candidates, reverse=True):
        if ti not in used_t and pi not in used_p:
            used_t.add(ti); used_p.add(pi)
    return len(used_t), len(truth) - len(used_t), len(predicted) - len(used_p)


def evaluate_config(grouped, intervals, split, high, low, min_duration_s):
    truth_frames, predicted_frames = [], []
    matched = missed = false_alarms = 0
    duration_s = 0.0
    selected_videos = 0
    for video_id, samples in grouped.items():
        if samples[0]["split"] != split:
            continue
        selected_videos += 1
        states, events = predict(samples, high, low, min_duration_s)
        truth = intervals[video_id]
        m, miss, false = _match(truth, events)
        matched += m; missed += miss; false_alarms += false
        duration_s += samples[-1]["t"] - samples[0]["t"]
        truth_frames.extend(_truth_at(row["t"], truth) for row in samples)
        predicted_frames.extend(states)
    if not selected_videos:
        raise ValueError(f"No MAR samples for split={split}")
    frame = classification_report(truth_frames, predicted_frames, ["normal", "yawn"])
    precision = matched / (matched + false_alarms) if matched + false_alarms else 0.0
    recall = matched / (matched + missed) if matched + missed else 0.0
    event_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "split": split, "videos": selected_videos, "high_threshold": high,
        "low_threshold": low, "min_duration_s": min_duration_s,
        "frame": frame,
        "events": {"matched": matched, "missed": missed,
                   "false_alarms": false_alarms, "precision": precision,
                   "recall": recall, "f1": event_f1,
                   "false_alarms_per_minute": false_alarms / (duration_s / 60)},
    }


def select_and_evaluate(samples_path: Path, labels_path: Path) -> dict:
    grouped = load_samples(samples_path)
    intervals = defaultdict(list)
    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            intervals[row["video_id"]].append((float(row["start_s"]), float(row["end_s"])))
    candidates = []
    for high_i in range(35, 76, 5):
        high = high_i / 100
        low = max(.20, high - .10)
        for duration in (.2, .4, .6, .8, 1.0):
            candidates.append(evaluate_config(grouped, intervals, "val", high, low, duration))
    selected = max(candidates, key=lambda row: (
        row["events"]["f1"], row["frame"]["macro_f1"],
        -row["events"]["false_alarms_per_minute"], -row["high_threshold"],
    ))
    test = evaluate_config(grouped, intervals, "test", selected["high_threshold"],
                           selected["low_threshold"], selected["min_duration_s"])
    valid = sum(row["mar"] is not None for rows in grouped.values() for row in rows)
    total = sum(len(rows) for rows in grouped.values())
    return {"selection_policy": "maximize validation event F1, then frame macro F1",
            "samples": total, "mar_valid_samples": valid,
            "mar_extraction_coverage": valid / total,
            "validation_candidates": len(candidates), "validation_selected": selected,
            "test_final": test}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--samples", type=Path, default=Path("artifacts/yawdd_mar_samples.csv"))
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--reuse-samples", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/yawdd_mar_report.json"))
    args = parser.parse_args()
    try:
        if not args.reuse_samples:
            count = extract_samples(args.manifest, args.labels, args.model,
                                    args.samples, args.sample_fps)
            print(f"Extracted {count} MAR samples", flush=True)
        report = select_and_evaluate(args.samples, args.labels)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
