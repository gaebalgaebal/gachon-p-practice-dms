"""Probe a local webcam without recording or saving image frames."""

import argparse
import json
import math
from pathlib import Path
import platform
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.camera < 0 or args.frames <= 0:
        parser.error("camera must be nonnegative and frames must be positive")
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        parser.exit(2, f"Missing camera dependency: {exc}. Install with: pip install -e '.[vision]'\n")

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    capture = cv2.VideoCapture(args.camera, backend)
    if not capture.isOpened() and platform.system() == "Windows":
        capture.release()
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        parser.exit(2, f"Cannot open camera {args.camera}. Close other camera apps and check permission.\n")
    started = time.perf_counter()
    brightness, shapes, failures = [], [], 0
    reported_source_fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        for _ in range(args.frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                failures += 1
                continue
            shapes.append(list(frame.shape[:2]))
            brightness.append(float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    report = {
        "camera": args.camera,
        "requested_frames": args.frames,
        "captured_frames": len(shapes),
        "failed_reads": failures,
        "capture_fps": len(shapes) / elapsed if elapsed else None,
        "resolution_hw": shapes[-1] if shapes else None,
        "mean_brightness_0_255": sum(brightness) / len(brightness) if brightness else None,
        "reported_source_fps": reported_source_fps,
        "privacy": "No frame images were saved.",
    }
    if not shapes or not math.isfinite(report["capture_fps"]):
        parser.exit(2, "Camera opened but no usable frames were captured.\n")
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        if args.output.exists():
            parser.exit(2, f"Output already exists: {args.output}\n")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
