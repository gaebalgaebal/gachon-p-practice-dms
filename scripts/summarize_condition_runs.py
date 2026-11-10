"""Aggregate live DMS runs into environment-condition failure and coverage metrics."""

import argparse
import json
from pathlib import Path
from statistics import mean


def summarize_run(path):
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            (path / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"Empty frame log: {path}")
    frames = len(rows)
    reasons = {}
    for row in rows:
        reason = row.get("reason", "missing")
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "run": str(path),
        "condition": summary.get("condition", "unspecified"),
        "frames": frames,
        "face_valid_fraction": sum(bool(r.get("face_valid")) for r in rows) / frames,
        "calibrated_fraction": sum(bool(r.get("calibrated")) for r in rows) / frames,
        "drowsiness_unknown_fraction": sum(
            r.get("drowsiness") == "unknown" for r in rows) / frames,
        "attention_unknown_fraction": sum(
            r.get("attention") == "unknown" for r in rows) / frames,
        "any_unknown_fraction": sum(
            r.get("drowsiness") == "unknown" or r.get("attention") == "unknown"
            for r in rows) / frames,
        "brightness_mean": mean(float(r["brightness_mean"]) for r in rows
                                if r.get("brightness_mean") is not None),
        "processing_fps": summary.get("processing_fps"),
        "steady_state_fps": summary.get("steady_state_fps"),
        "camera_frames_skipped": summary.get("camera_frames_skipped"),
        "face_failure_reasons": reasons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output already exists")
        report = {"runs": [summarize_run(path) for path in args.runs]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved {len(report['runs'])} condition runs to {args.output}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Condition summary error: {exc}\n")


if __name__ == "__main__":
    main()
