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
import threading
import time

from .calibration import Calibrator, EyeHistory, EyeSignalHistory, angle_difference
from .contracts import DriverState, Features
from .state import Config, StateEngine


MANUAL_LABEL_KEYS = {
    ord("0"): "normal", ord("1"): "eyes_closed", ord("2"): "away",
    ord("3"): "yawn", ord("4"): "phone", ord("u"): "unknown",
    ord("U"): "unknown",
}
GROUND_TRUTH_LABELS = {"normal", "eyes_closed", "away", "yawn", "phone", "unknown"}


def load_label_schedule(path):
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Label schedule must be a JSON list")
    schedule = []
    previous_end = 0.0
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each label schedule item must be an object")
        try:
            start, end, label = float(item["start_s"]), float(item["end_s"]), item["label"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Schedule items require start_s, end_s and label") from exc
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start
                or label not in GROUND_TRUTH_LABELS or start < previous_end):
            raise ValueError(f"Invalid or overlapping label schedule item: {item}")
        schedule.append((start, end, label))
        previous_end = end
    return schedule


def label_at_time(schedule, timestamp_s):
    for start, end, label in schedule:
        if start <= timestamp_s < end:
            return label
    return "unknown"


class LatestFrameCapture:
    """Continuously capture camera frames so inference does not serialize with device I/O."""

    def __init__(self, capture):
        self.capture = capture
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.consumed = 0
        self.total_dropped = 0
        self.failed = False
        self.stopped = False
        self.thread = threading.Thread(target=self._loop, name="dms-camera-capture", daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stopped:
            ok, frame = self.capture.read()
            with self.condition:
                if not ok:
                    self.failed = True
                    self.condition.notify_all()
                    return
                self.frame = frame
                self.sequence += 1
                self.condition.notify_all()

    def read(self):
        with self.condition:
            self.condition.wait_for(lambda: self.sequence > self.consumed or self.failed,
                                    timeout=2.)
            if self.sequence <= self.consumed:
                return False, None
            self.total_dropped += max(0, self.sequence - self.consumed - 1)
            self.consumed = self.sequence
            return True, self.frame.copy()

    @property
    def dropped_frames(self):
        return self.total_dropped

    def release(self):
        self.stopped = True
        self.capture.release()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2.)


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


def draw_eye_graph(canvas, samples, threshold, origin=(12, 48), size=(260, 112)):
    """Draw a fixed-scale 10-second left/right EAR graph over the camera image."""
    import cv2

    x, y = origin
    width, height = size
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (18, 22, 28), -1)
    cv2.addWeighted(overlay, .78, canvas, .22, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (115, 125, 140), 1)
    cv2.putText(canvas, "EYE OPENING / 10s", (x + 8, y + 17),
                cv2.FONT_HERSHEY_SIMPLEX, .42, (225, 230, 235), 1, cv2.LINE_AA)
    plot_top, plot_bottom = y + 24, y + height - 9
    if threshold is not None:
        ty = round(plot_bottom - min(max(threshold / .5, 0), 1) * (plot_bottom - plot_top))
        cv2.line(canvas, (x + 1, ty), (x + width - 1, ty), (80, 90, 230), 1)
    if len(samples) >= 2:
        latest = samples[-1][0]
        for value_index, color in ((1, (70, 230, 190)), (2, (250, 190, 80))):
            points = []
            for timestamp, left, right in samples:
                value = (left, right)[value_index - 1]
                px = round(x + width - min(max((latest - timestamp) / 10, 0), 1) * width)
                py = round(plot_bottom - min(max(value / .5, 0), 1) *
                           (plot_bottom - plot_top))
                points.append((px, py))
            for start, end in zip(points, points[1:]):
                cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "L", (x + width - 38, y + 17), cv2.FONT_HERSHEY_SIMPLEX,
                .42, (70, 230, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, "R", (x + width - 20, y + 17), cv2.FONT_HERSHEY_SIMPLEX,
                .42, (250, 190, 80), 1, cv2.LINE_AA)


def render(frame, observation, state, info, roi_box, fps, eye_trace):
    import cv2
    from .vision import LEFT_EYE, MOUTH, RIGHT_EYE

    x, y, w, h = roi_box
    cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (100, 100, 100), 1)
    for detection in info.get("action_detections", []):
        x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
        cv2.rectangle(frame, (x1 + x, y1 + y), (x2 + x, y2 + y), (40, 80, 255), 2)
        cv2.putText(frame, f"{detection.label} {detection.confidence:.2f}",
                    (x1 + x, max(18, y1 + y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (40, 80, 255), 2, cv2.LINE_AA)
    if observation.points is not None:
        for index in LEFT_EYE + RIGHT_EYE:
            px, py = observation.points[index]
            cv2.circle(frame, (round(px) + x, round(py) + y), 2, (70, 230, 190), -1)
        for index in MOUTH:
            px, py = observation.points[index]
            cv2.circle(frame, (round(px) + x, round(py) + y), 2, (80, 190, 250), -1)
    draw_eye_graph(frame, eye_trace, info.get("ear_close"))
    canvas = cv2.copyMakeBorder(frame, 0, 0, 0, 390, cv2.BORDER_CONSTANT, value=(24, 28, 36))
    active = (state.drowsiness == "suspected" or state.attention == "off_forward"
              or state.yawn == "yawning" or state.action in {"phone", "drink", "smoke"})
    color = (70, 70, 255) if active else (70, 220, 170)
    tx = frame.shape[1] + 16
    def label(text, line, tint=(220, 225, 235), scale=.53):
        cv2.putText(canvas, text, (tx, 30 + line * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, tint, 1, cv2.LINE_AA)
    label("GACHON P-PRACTICE / DMS", 0, (70, 220, 170), .62)
    label("LOCAL RGB CAMERA DEMO", 1)
    label(f"Drowsiness: {state.drowsiness}", 3, color)
    label(f"Eye: {state.eye}", 4)
    label(f"Head attention: {state.attention}", 5, color)
    label(f"Yawn: {state.yawn}", 6, color if state.yawn == "yawning" else (220, 225, 235))
    label(f"Action: {state.action}", 7)
    label(f"Face: {observation.reason}", 9)
    label(f"Calibration: {info['calibration']}", 10)
    ear = f"{observation.ear:.3f}" if observation.ear is not None else "--"
    mar = f"{observation.mar:.3f}" if observation.mar is not None else "--"
    yaw = f"{info['yaw']:.1f}" if info['yaw'] is not None else "--"
    pitch = f"{info['pitch']:.1f}" if info['pitch'] is not None else "--"
    label(f"EAR {ear} / MAR {mar}", 11)
    eye_score = info.get("eye_closed_score")
    label(f"Eye model closed: {eye_score:.2f}" if eye_score is not None
          else "Eye model closed: --", 12)
    label(f"Yaw {yaw} / Pitch {pitch}", 13)
    gaze_x = f"{info['gaze_x']:+.2f}" if info['gaze_x'] is not None else "--"
    gaze_y = f"{info['gaze_y']:+.2f}" if info['gaze_y'] is not None else "--"
    label(f"Calibrated gaze X {gaze_x} / Y {gaze_y}", 14)
    perclos = info['closed_fraction']
    label(f"Closed fraction: {perclos:.2f}" if perclos is not None else "Closed fraction: --", 15)
    label(f"Coverage: {info['eye_coverage']:.0%} / FPS: {fps:.1f}", 16)
    label(f"Ground truth: {info['ground_truth']}", 17, (80, 200, 255))
    label("0 normal  1 eyes  2 away  3 yawn  4 phone", 18, scale=.42)
    label("U unknown  C recalibrate  Q / Esc exit", 19, scale=.42)
    if active:
        cv2.rectangle(canvas, (2, 2), (frame.shape[1] - 3, frame.shape[0] - 3), color, 5)
        cv2.putText(canvas, "CHECK DRIVER STATE", (14, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, color, 2, cv2.LINE_AA)
    return canvas


def run(args):
    import cv2
    from .vision import MediaPipeFaceAnalyzer

    process = None
    cpu_started = None
    rss_started = None
    rss_peak = None
    try:
        import psutil
        process = psutil.Process()
        cpu_started = process.cpu_times()
        rss_started = process.memory_info().rss
        rss_peak = rss_started
    except ImportError:
        pass

    config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    label_schedule = load_label_schedule(args.label_schedule)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output}; choose a new directory")
    if args.video and not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    analyzer = MediaPipeFaceAnalyzer(args.model)
    cap = None
    writer = None
    frames_log = events_log = labels_log = None
    latencies, completion_times, unknown, frame_count = [], [], 0, 0
    started = time.perf_counter()
    exit_reason = "not_started"
    calibration = Calibrator(max_gap=config.max_gap_seconds)
    eye_history = EyeHistory(max_gap=config.max_gap_seconds)
    eye_signal_history = EyeSignalHistory(max_gap=config.max_gap_seconds)
    engine = StateEngine(config)
    last_state = None
    last_warning = -math.inf
    source_origin = None
    previous_t = -1.0
    applied_neutral = None
    action_model = None
    action_smoother = None
    eye_model = None
    torch_cuda = None
    ground_truth = "unknown"
    manual_label_changes = 0
    try:
        if args.action_checkpoint:
            from .action import ActionSmoother, load_action_model
            action_model = load_action_model(args.action_checkpoint, args.action_backend,
                                             args.action_threshold, args.action_device)
            action_smoother = ActionSmoother(
                args.action_window, args.action_consensus, args.action_min_samples
            )
            if type(action_model).__name__ == "YOLOAction":
                try:
                    import torch
                    if torch.cuda.is_available() and str(args.action_device).lower() != "cpu":
                        torch.cuda.reset_peak_memory_stats()
                        torch_cuda = torch.cuda
                except (ImportError, RuntimeError):
                    pass
        if args.eye_model:
            from .eye import OpenVINOEyeClassifier
            eye_model = OpenVINOEyeClassifier(args.eye_model, args.eye_device)
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
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if args.camera_exposure is not None:
                # V4L2 uses 1 for manual exposure on this target camera.
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                cap.set(cv2.CAP_PROP_EXPOSURE, args.camera_exposure)
        source_fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            if args.video:
                raise ValueError("Video FPS unavailable; supply --fps explicitly")
            source_fps = 30.0
        if not args.video:
            cap = LatestFrameCapture(cap)
        args.output.mkdir(parents=True, exist_ok=True)
        frames_log = (args.output / "frames.jsonl").open("w", encoding="utf-8")
        events_log = (args.output / "events.jsonl").open("w", encoding="utf-8")
        labels_log = (args.output / "manual_labels.jsonl").open("w", encoding="utf-8")
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
                observation.iris_x if valid else None,
                observation.iris_y if valid else None,
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
            gaze_x = (observation.iris_x - neutral.iris_x
                      if valid and neutral and observation.iris_x is not None
                      and neutral.iris_x is not None else None)
            gaze_y = (observation.iris_y - neutral.iris_y
                      if valid and neutral and observation.iris_y is not None
                      and neutral.iris_y is not None else None)
            usable = valid and neutral is not None
            eye_scores = eye_model.predict(crop, observation.points) if eye_model and usable else None
            eye_closed_score = eye_scores.closed_score if eye_scores is not None else None
            features = Features(t, observation.ear, yaw, usable, pitch, gaze_x, gaze_y,
                                observation.mar, eye_closed_score, eye_model is not None)
            state = engine.update(features)
            source = "mediapipe_ear_pose_iris_v1" if gaze_x is not None else "mediapipe_ear_pose_v1"
            state = DriverState(t, state.drowsiness, state.attention, action="unknown",
                                source=source, yawn=state.yawn, eye=state.eye)
            action_scores = {}
            action_input_usable = action_model and (usable or not action_model.requires_face)
            if action_input_usable:
                if frame_count % args.action_interval == 0:
                    raw_action, raw_scores = action_model.predict(crop)
                    action, action_scores = action_smoother.update(raw_action, raw_scores)
                else:
                    action, action_scores = action_smoother.current()
                state = DriverState(t, state.drowsiness, state.attention, action=action,
                                    source=state.source, yawn=state.yawn, eye=state.eye)
            elif action_smoother:
                # Never retain an action after the face/ROI becomes unusable.
                action_smoother.reset()
            closed = observation.ear < engine.config.ear_close if usable else None
            fraction, coverage = eye_history.update(t, closed)
            eye_trace = eye_signal_history.update(
                t, observation.left_ear if usable else None,
                observation.right_ear if usable else None,
            )
            if label_schedule:
                next_label = label_at_time(label_schedule, t)
                if next_label != ground_truth:
                    ground_truth = next_label
                    manual_label_changes += 1
                    json_line(labels_log, {"timestamp_s": t, "frame_index": frame_count,
                                           "label": ground_truth, "source": "schedule"})
            unknown += state.drowsiness == "unknown" or state.attention == "unknown"
            info = {"calibration": "ready" if neutral else "hold forward / eyes open",
                    "yaw": yaw, "pitch": pitch, "gaze_x": gaze_x, "gaze_y": gaze_y,
                    "closed_fraction": fraction, "eye_coverage": coverage,
                    "ear_close": engine.config.ear_close,
                    "eye_closed_score": eye_closed_score,
                    "ground_truth": ground_truth,
                    "action_detections": getattr(action_model, "last_detections", [])}
            frame_count += 1
            elapsed = time.perf_counter() - started
            fps = frame_count / max(elapsed, 1e-6)
            canvas = render(frame, observation, state, info, (x, y, cw, ch), fps, eye_trace)
            if args.record:
                if writer is None:
                    record_fps = args.record_fps or source_fps
                    writer = cv2.VideoWriter(str(args.output / "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), record_fps,
                                             (canvas.shape[1], canvas.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create annotated video; try without --record")
                writer.write(canvas)
            row = {**asdict(state), "frame_index": frame_count - 1,
                   "brightness_mean": float(frame.mean()),
                   "face_valid": valid, "reason": observation.reason, "face_count": observation.face_count,
                   "ear": observation.ear, "left_ear": observation.left_ear,
                   "right_ear": observation.right_ear, "mar": observation.mar,
                   "eye_left_open_score": eye_scores.left_open if eye_scores else None,
                   "eye_left_closed_score": eye_scores.left_closed if eye_scores else None,
                   "eye_right_open_score": eye_scores.right_open if eye_scores else None,
                   "eye_right_closed_score": eye_scores.right_closed if eye_scores else None,
                   "eye_closed_score": eye_closed_score,
                   "raw_yaw": observation.yaw,
                   "raw_pitch": observation.pitch, "relative_yaw": yaw, "relative_pitch": pitch,
                   "iris_x": observation.iris_x, "iris_y": observation.iris_y,
                   "relative_gaze_x": gaze_x, "relative_gaze_y": gaze_y,
                   "closed_fraction": fraction,
                   "eye_window_coverage": coverage, "calibrated": neutral is not None,
                   "baseline_drowsiness": "suspected" if usable and closed else (
                       "normal" if usable else "unknown"), "action_scores": action_scores,
                   "ground_truth": ground_truth,
                   "action_detections": [asdict(item) for item in
                                         getattr(action_model, "last_detections", [])],
                   "processing_ms": (time.perf_counter() - captured) * 1000}
            json_line(frames_log, row)
            signature = (state.drowsiness, state.attention, state.action)
            if signature != last_state:
                json_line(events_log, {"type": "state_change", **asdict(state)})
                last_state = signature
            warning = (state.drowsiness == "suspected" or state.attention == "off_forward"
                       or state.yawn == "yawning" or state.action in {"phone", "drink", "smoke"})
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
            if not label_schedule and key in MANUAL_LABEL_KEYS:
                next_label = MANUAL_LABEL_KEYS[key]
                if next_label != ground_truth:
                    ground_truth = next_label
                    manual_label_changes += 1
                    json_line(labels_log, {"timestamp_s": t, "frame_index": frame_count - 1,
                                           "label": ground_truth})
            latencies.append((time.perf_counter() - read_started) * 1000)
            completion_times.append(time.perf_counter())
            if process is not None:
                try:
                    rss_peak = max(rss_peak, process.memory_info().rss)
                except OSError:
                    pass
            if key in (ord('q'), 27):
                exit_reason = "user_exit"
                break
            if key == ord('c'):
                calibration.reset()
                applied_neutral = None
                engine = StateEngine(config)
                eye_history = EyeHistory(max_gap=config.max_gap_seconds)
                eye_signal_history.reset()
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
            labels_log.close()
            versions = {name: importlib.metadata.version(name) for name in (
                "mediapipe", "opencv-contrib-python", "numpy")}
            if eye_model:
                versions["openvino"] = importlib.metadata.version("openvino")
            if action_model and type(action_model).__name__ == "YOLOAction":
                versions["ultralytics"] = importlib.metadata.version("ultralytics")
            if process is not None:
                versions["psutil"] = importlib.metadata.version("psutil")
            resource_metrics = {
                "process_cpu_user_seconds": None,
                "process_cpu_system_seconds": None,
                "process_cpu_percent_one_core": None,
                "process_rss_start_mb": rss_started / 1024 ** 2 if rss_started else None,
                "process_rss_end_mb": None,
                "process_rss_peak_sampled_mb": rss_peak / 1024 ** 2 if rss_peak else None,
                "gpu_memory_allocated_mb": None,
                "gpu_memory_reserved_mb": None,
                "gpu_peak_memory_allocated_mb": None,
                "gpu_peak_memory_reserved_mb": None,
            }
            if process is not None and cpu_started is not None:
                try:
                    cpu_ended = process.cpu_times()
                    cpu_user = cpu_ended.user - cpu_started.user
                    cpu_system = cpu_ended.system - cpu_started.system
                    rss_end = process.memory_info().rss
                    resource_metrics.update({
                        "process_cpu_user_seconds": cpu_user,
                        "process_cpu_system_seconds": cpu_system,
                        "process_cpu_percent_one_core": (
                            (cpu_user + cpu_system) / elapsed * 100 if elapsed else None),
                        "process_rss_end_mb": rss_end / 1024 ** 2,
                    })
                except OSError:
                    pass
            if torch_cuda is not None:
                try:
                    resource_metrics.update({
                        "gpu_memory_allocated_mb": torch_cuda.memory_allocated() / 1024 ** 2,
                        "gpu_memory_reserved_mb": torch_cuda.memory_reserved() / 1024 ** 2,
                        "gpu_peak_memory_allocated_mb": (
                            torch_cuda.max_memory_allocated() / 1024 ** 2),
                        "gpu_peak_memory_reserved_mb": (
                            torch_cuda.max_memory_reserved() / 1024 ** 2),
                    })
                except RuntimeError:
                    pass
            summary = {"frames": frame_count, "wall_seconds": elapsed,
                       "condition": args.condition,
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
                       "record_fps": (args.record_fps or source_fps) if args.record else None,
                       "camera_fps_requested": args.camera_fps if not args.video else None,
                       "camera_exposure_requested": args.camera_exposure if not args.video else None,
                       "camera_frames_skipped": cap.dropped_frames
                           if isinstance(cap, LatestFrameCapture) else 0,
                       "action_checkpoint": args.action_checkpoint.name if args.action_checkpoint else None,
                       "action_checkpoint_sha256": hashlib.sha256(args.action_checkpoint.read_bytes()).hexdigest()
                           if args.action_checkpoint else None,
                       "action_threshold": action_model.threshold if action_model else args.action_threshold,
                       "action_threshold_requested": args.action_threshold,
                       "action_backend": type(action_model).__name__ if action_model else None,
                       "action_device": args.action_device if action_model else None,
                       "action_interval": args.action_interval,
                       "action_window": args.action_window,
                       "action_consensus": args.action_consensus,
                       "action_min_samples": args.action_min_samples,
                       "eye_model": args.eye_model.name if args.eye_model else None,
                       "eye_model_sha256": hashlib.sha256(args.eye_model.read_bytes()).hexdigest()
                           if args.eye_model else None,
                       "eye_device": args.eye_device if args.eye_model else None,
                       "manual_label_changes": manual_label_changes,
                       "label_schedule": args.label_schedule.name if args.label_schedule else None,
                       "label_schedule_sha256": hashlib.sha256(
                           args.label_schedule.read_bytes()).hexdigest()
                           if args.label_schedule else None,
                       "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                       "note": "Technical demo, not accuracy evaluation; closed fraction is not clinical PERCLOS."}
            summary.update(resource_metrics)
            warmup = min(30, max(0, len(completion_times) - 2))
            steady_latencies = latencies[warmup:]
            steady_elapsed = (completion_times[-1] - completion_times[warmup]
                              if len(completion_times) - warmup >= 2 else None)
            summary.update({
                "steady_state_warmup_frames": warmup,
                "steady_state_fps": ((len(completion_times) - warmup - 1) / steady_elapsed
                                     if steady_elapsed and steady_elapsed > 0 else None),
                "steady_state_frame_loop_ms_mean": (sum(steady_latencies) / len(steady_latencies)
                                                     if steady_latencies else None),
                "steady_state_frame_loop_ms_p95": percentile(steady_latencies, .95),
            })
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False),
                                                     encoding="utf-8")
            print(f"{frame_count} frames, {summary['processing_fps']:.1f} processing FPS; logs: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0)
    source.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--eye-model", type=Path,
                        help="Open Model Zoo open-closed-eye-0001 ONNX model")
    parser.add_argument("--eye-device", default="CPU",
                        help="OpenVINO device such as CPU, GPU or AUTO")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--roi", type=parse_roi, default=(0., 0., 1., 1.))
    parser.add_argument("--fps", type=float, help="Override file FPS when metadata is missing")
    parser.add_argument("--camera-fps", type=float, default=30.,
                        help="Requested camera FPS; backend support varies")
    parser.add_argument("--camera-exposure", type=float,
                        help="Optional manual camera exposure value; device-specific")
    parser.add_argument("--condition", default="unspecified",
                        help="Environment label stored in summary, e.g. baseline or glasses")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--record", action="store_true", help="Opt in to local annotated video storage")
    parser.add_argument("--record-fps", type=float,
                        help="Output video FPS; use measured processing FPS for real-time playback")
    parser.add_argument("--label-schedule", type=Path,
                        help="JSON intervals for focus-independent ground-truth labels")
    parser.add_argument("--sound", action="store_true", help="Opt in to Windows warning sound")
    parser.add_argument("--action-checkpoint", type=Path)
    parser.add_argument("--action-backend", choices=("auto", "mobilenet", "yolo11"), default="auto")
    parser.add_argument("--action-device", default="auto", help="YOLO device: auto, cpu, or CUDA index")
    parser.add_argument("--action-threshold", type=float,
                        help="Override confidence; defaults: MobileNet 0.70, YOLO11 0.25")
    parser.add_argument("--action-interval", type=int, default=3,
                        help="Run the action CNN every N usable frames")
    parser.add_argument("--action-window", type=int, default=5,
                        help="Number of fresh CNN outputs used for temporal voting")
    parser.add_argument("--action-consensus", type=float, default=.6)
    parser.add_argument("--action-min-samples", type=int, default=3)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.fps is not None and (not math.isfinite(args.fps) or not 0 < args.fps <= 1000):
        parser.error("--fps must be finite and in (0,1000]")
    if args.record_fps is not None and (
            not math.isfinite(args.record_fps) or not 0 < args.record_fps <= 1000):
        parser.error("--record-fps must be finite and in (0,1000]")
    if (not math.isfinite(args.camera_fps) or not 0 < args.camera_fps <= 1000
            or args.camera_exposure is not None and not math.isfinite(args.camera_exposure)):
        parser.error("Camera FPS must be in (0,1000] and exposure finite")
    if args.headless and not args.video and args.max_frames is None:
        parser.error("Headless camera requires --max-frames")
    if args.action_threshold is not None and not 0 < args.action_threshold <= 1:
        parser.error("--action-threshold must be in (0,1]")
    if (args.action_interval <= 0 or args.action_window <= 0
            or not 0 < args.action_min_samples <= args.action_window):
        parser.error("Action interval/window must be positive and min-samples within the window")
    if not 0 < args.action_consensus <= 1:
        parser.error("--action-consensus must be in (0,1]")
    try:
        return run(args)
    except ImportError as exc:
        parser.exit(2, f"Missing demo dependency: {exc}. Run: python -m pip install -e .[vision,eye]\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Demo error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Stopped. Partial logs and summary retained.\n")


if __name__ == "__main__":
    raise SystemExit(main())
