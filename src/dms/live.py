"""Real RGB demo: OpenCV capture -> MediaPipe -> calibrated rules -> UI/logs."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

from .calibration import Calibrator, EyeHistory, angle_difference
from .contracts import DriverState, Features
from .state import Config, StateEngine


def parse_roi(value):
    try:
        roi = tuple(float(x) for x in value.split(","))
        if len(roi) != 4:
            raise ValueError()
        x, y, w, h = roi
        if not all(math.isfinite(v) for v in roi) or min(x, y) < 0 or min(w, h) <= 0:
            raise ValueError()
        if x + w > 1 or y + h > 1:
            raise ValueError()
        return roi
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h in [0,1], within the frame") from exc


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]


def json_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def render(frame, observation, state, info, roi_box, fps):
    import cv2
    from .vision import LEFT_EYE, RIGHT_EYE

    x, y, w, h = roi_box
    cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (100, 100, 100), 1)
    if observation.points is not None:
        for index in LEFT_EYE + RIGHT_EYE:
            px, py = observation.points[index]
            cv2.circle(frame, (round(px) + x, round(py) + y), 2, (70, 230, 190), -1)
    canvas = cv2.copyMakeBorder(frame, 0, 0, 0, 390, cv2.BORDER_CONSTANT, value=(24, 28, 36))
    active = state.drowsiness == "suspected" or state.attention == "off_forward"
    color = (70, 70, 255) if active else (70, 220, 170)
    tx = frame.shape[1] + 16
    def label(text, line, tint=(220, 225, 235), scale=.53):
        cv2.putText(canvas, text, (tx, 30 + line * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, tint, 1, cv2.LINE_AA)
    label("GACHON P-PRACTICE / DMS", 0, (70, 220, 170), .62)
    label("LOCAL RGB CAMERA DEMO", 1)
    label(f"Drowsiness: {state.drowsiness}", 3, color)
    label(f"Head attention: {state.attention}", 4, color)
    label(f"Action: {state.action}", 5)
    label(f"Face: {observation.reason}", 7)
    label(f"Calibration: {info['calibration']}", 8)
    ear = f"{observation.ear:.3f}" if observation.ear is not None else "--"
    yaw = f"{info['yaw']:.1f}" if info['yaw'] is not None else "--"
    pitch = f"{info['pitch']:.1f}" if info['pitch'] is not None else "--"
    label(f"EAR {ear} / Yaw {yaw} / Pitch {pitch}", 9)
    perclos = info['closed_fraction']
    label(f"Closed fraction: {perclos:.2f}" if perclos is not None else "Closed fraction: --", 10)
    label(f"Window coverage: {info['eye_coverage']:.0%} / FPS: {fps:.1f}", 11)
    label("C: recalibrate  Q / Esc: exit", 13)
    label("Look forward, eyes OPEN to calibrate", 14, (80, 200, 255), .48)
    label("Head direction proxy; not eye gaze", 15, (160, 165, 175), .48)
    if active:
        cv2.rectangle(canvas, (2, 2), (frame.shape[1] - 3, frame.shape[0] - 3), color, 5)
        cv2.putText(canvas, "CHECK DRIVER STATE", (14, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, color, 2, cv2.LINE_AA)
    return canvas


def run(args):
    import cv2
    from .vision import MediaPipeFaceAnalyzer

    config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output}; choose a new directory")
    if args.video and not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    analyzer = MediaPipeFaceAnalyzer(args.model)
    cap = None
    writer = None
    frames_log = events_log = None
    latencies, unknown, frame_count = [], 0, 0
    started = time.perf_counter()
    exit_reason = "not_started"
    calibration = Calibrator(max_gap=config.max_gap_seconds)
    eye_history = EyeHistory(max_gap=config.max_gap_seconds)
    engine = StateEngine(config)
    last_state = None
    last_warning = -math.inf
    source_origin = None
    previous_t = -1.0
    applied_neutral = None
    action_model = None
    try:
        if args.action_checkpoint:
            from .action import MobileNetAction
            action_model = MobileNetAction(args.action_checkpoint, args.action_threshold)
        if args.video:
            cap = cv2.VideoCapture(str(args.video.resolve()))
        elif platform.system() == "Windows":
            cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(args.camera)
        else:
            cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError("Cannot open input. Check camera permission/index or video codec.")
        if not args.video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        source_fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            if args.video:
                raise ValueError("Video FPS unavailable; supply --fps explicitly")
            source_fps = 30.0
        args.output.mkdir(parents=True, exist_ok=True)
        frames_log = (args.output / "frames.jsonl").open("w", encoding="utf-8")
        events_log = (args.output / "events.jsonl").open("w", encoding="utf-8")
        started = time.perf_counter()
        while True:
            read_started = time.perf_counter()
            ok, frame = cap.read()
            captured = time.perf_counter()
            if not ok:
                if args.video and frame_count:
                    exit_reason = "video_end"
                    break
                raise RuntimeError("Camera read failed" if not args.video else "Video has no readable frames")
            if args.video:
                reported = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                t = reported if math.isfinite(reported) and reported > previous_t else frame_count / source_fps
                if t <= previous_t:
                    t = previous_t + 1 / source_fps
            else:
                if source_origin is None:
                    source_origin = captured
                t = captured - source_origin
            previous_t = t
            h, w = frame.shape[:2]
            scale = min(640 / w, 480 / h)
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
            h, w = frame.shape[:2]
            # Letterbox to 480px so the panel remains legible for widescreen input.
            if h < 480:
                frame = cv2.copyMakeBorder(frame, 0, 480 - h, 0, 0, cv2.BORDER_CONSTANT)
            rx, ry, rw, rh = args.roi
            x, y, cw, ch = round(rx * w), round(ry * h), round(rw * w), round(rh * h)
            if min(cw, ch) < 70:
                raise ValueError("Selected ROI is too small; use at least 70x70 pixels")
            crop = frame[y:y + ch, x:x + cw]
            observation = analyzer.analyze(crop, t)
            valid = observation.valid
            neutral = calibration.update(
                t, observation.ear if valid else None,
                observation.yaw if valid else None, observation.pitch if valid else None,
            )
            if neutral is not None and applied_neutral is None:
                engine = StateEngine(neutral.config(config))
                applied_neutral = neutral
                (args.output / "calibration.json").write_text(
                    json.dumps({"neutral": asdict(neutral), "rules": asdict(engine.config)}, indent=2),
                    encoding="utf-8",
                )
            yaw = angle_difference(observation.yaw, neutral.yaw) if valid and neutral else None
            pitch = angle_difference(observation.pitch, neutral.pitch) if valid and neutral else None
            usable = valid and neutral is not None
            features = Features(t, observation.ear, yaw, usable, pitch)
            state = engine.update(features)
            state = DriverState(t, state.drowsiness, state.attention, "unknown", "mediapipe_ear_pose_v1")
            action_scores = {}
            if action_model and usable:
                # Always compute a fresh prediction; never keep a stale action after loss.
                action, action_scores = action_model.predict(crop)
                state = DriverState(t, state.drowsiness, state.attention, action, state.source)
            closed = observation.ear < engine.config.ear_close if usable else None
            fraction, coverage = eye_history.update(t, closed)
            unknown += state.drowsiness == "unknown" or state.attention == "unknown"
            info = {"calibration": "ready" if neutral else "hold forward / eyes open",
                    "yaw": yaw, "pitch": pitch, "closed_fraction": fraction, "eye_coverage": coverage}
            frame_count += 1
            elapsed = time.perf_counter() - started
            fps = frame_count / max(elapsed, 1e-6)
            canvas = render(frame, observation, state, info, (x, y, cw, ch), fps)
            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output / "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), source_fps,
                                             (canvas.shape[1], canvas.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create annotated video; try without --record")
                writer.write(canvas)
            row = {**asdict(state), "frame_index": frame_count - 1,
                   "face_valid": valid, "reason": observation.reason, "face_count": observation.face_count,
                   "ear": observation.ear, "left_ear": observation.left_ear,
                   "right_ear": observation.right_ear, "raw_yaw": observation.yaw,
                   "raw_pitch": observation.pitch, "relative_yaw": yaw, "relative_pitch": pitch,
                   "iris_x_diagnostic": observation.iris_x, "closed_fraction": fraction,
                   "eye_window_coverage": coverage, "calibrated": neutral is not None,
                   "baseline_drowsiness": "suspected" if usable and closed else (
                       "normal" if usable else "unknown"), "action_scores": action_scores,
                   "processing_ms": (time.perf_counter() - captured) * 1000}
            json_line(frames_log, row)
            signature = (state.drowsiness, state.attention, state.action)
            if signature != last_state:
                json_line(events_log, {"type": "state_change", **asdict(state)})
                last_state = signature
            warning = state.drowsiness == "suspected" or state.attention == "off_forward"
            if warning and t - last_warning >= 3:
                last_warning = t
                json_line(events_log, {"type": "warning", **asdict(state)})
                if args.sound and platform.system() == "Windows":
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            key = -1
            if not args.headless:
                cv2.imshow("Gachon DMS - Q: exit / C: recalibrate", canvas)
                key = cv2.waitKey(1) & 0xff
            latencies.append((time.perf_counter() - read_started) * 1000)
            if key in (ord('q'), 27):
                exit_reason = "user_exit"
                break
            if key == ord('c'):
                calibration.reset()
                applied_neutral = None
                engine = StateEngine(config)
                eye_history = EyeHistory(max_gap=config.max_gap_seconds)
                json_line(events_log, {"type": "recalibration", "timestamp_s": t})
            if args.max_frames and frame_count >= args.max_frames:
                exit_reason = "frame_limit"
                break
        return 0
    except BaseException:
        exit_reason = "error"
        raise
    finally:
        elapsed = time.perf_counter() - started
        analyzer.close()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if frames_log:
            frames_log.close()
            events_log.close()
            versions = {name: importlib.metadata.version(name) for name in (
                "mediapipe", "opencv-contrib-python", "numpy")}
            summary = {"frames": frame_count, "wall_seconds": elapsed,
                       "processing_fps": frame_count / elapsed if elapsed else None,
                       "frame_loop_ms_mean": sum(latencies) / len(latencies) if latencies else None,
                       "frame_loop_ms_p95": percentile(latencies, .95),
                       "unknown_frame_fraction": unknown / frame_count if frame_count else None,
                       "exit_reason": exit_reason, "source_kind": "video" if args.video else "camera",
                       "video_source_fps": source_fps, "source_duration_s": previous_t,
                       "python": platform.python_version(), "platform": platform.platform(),
                       "versions": versions, "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                       "config": asdict(config), "effective_config": asdict(engine.config),
                       "calibrated": calibration.neutral is not None,
                       "roi": args.roi, "recorded": args.record,
                       "action_checkpoint": args.action_checkpoint.name if args.action_checkpoint else None,
                       "action_checkpoint_sha256": hashlib.sha256(args.action_checkpoint.read_bytes()).hexdigest()
                           if args.action_checkpoint else None,
                       "action_threshold": args.action_threshold,
                       "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                       "note": "Technical demo, not accuracy evaluation; closed fraction is not clinical PERCLOS."}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False),
                                                     encoding="utf-8")
            print(f"{frame_count} frames, {summary['processing_fps']:.1f} processing FPS; logs: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0)
    source.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--roi", type=parse_roi, default=(0., 0., 1., 1.))
    parser.add_argument("--fps", type=float, help="Override file FPS when metadata is missing")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--record", action="store_true", help="Opt in to local annotated video storage")
    parser.add_argument("--sound", action="store_true", help="Opt in to Windows warning sound")
    parser.add_argument("--action-checkpoint", type=Path)
    parser.add_argument("--action-threshold", type=float, default=.7)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.fps is not None and (not math.isfinite(args.fps) or not 0 < args.fps <= 1000):
        parser.error("--fps must be finite and in (0,1000]")
    if args.headless and not args.video and args.max_frames is None:
        parser.error("Headless camera requires --max-frames")
    if not 0 < args.action_threshold <= 1:
        parser.error("--action-threshold must be in (0,1]")
    try:
        return run(args)
    except ImportError as exc:
        parser.exit(2, f"Missing demo dependency: {exc}. Run: python -m pip install -e .[vision]\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Demo error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Stopped. Partial logs and summary retained.\n")


if __name__ == "__main__":
    raise SystemExit(main())
