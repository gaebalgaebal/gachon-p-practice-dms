"""Reusable product engine independent from CLI and desktop UI."""

from dataclasses import dataclass, field
from pathlib import Path

from .calibration import Calibrator, EyeHistory, angle_difference
from .contracts import DriverState, Features
from .state import Config, StateEngine
from .vision import Observation


@dataclass(frozen=True)
class EngineResult:
    timestamp_s: float
    state: DriverState
    observation: Observation
    calibrated: bool
    calibration_progress: float
    calibration_message: str
    warning: bool
    unknown: bool
    closed_fraction: float | None
    eye_coverage: float
    eye_closed_score: float | None
    relative_yaw: float | None = None
    relative_pitch: float | None = None
    relative_gaze_x: float | None = None
    relative_gaze_y: float | None = None
    usable: bool = False
    closed: bool | None = None
    eye_scores: object | None = None
    action_scores: dict[str, float] = field(default_factory=dict)
    action_detections: tuple = ()


class CalibrationService:
    """Expose neutral calibration as product progress and user-facing reason codes."""

    def __init__(self, config: Config):
        self.config = config
        self.calibrator = Calibrator(max_gap=config.max_gap_seconds)

    @property
    def neutral(self):
        return self.calibrator.neutral

    def reset(self):
        self.calibrator.reset()

    def update(self, observation: Observation, timestamp_s: float):
        valid = observation.valid
        return self.calibrator.update(
            timestamp_s, observation.ear if valid else None,
            observation.yaw if valid else None, observation.pitch if valid else None,
            observation.iris_x if valid else None, observation.iris_y if valid else None,
        )

    def progress(self) -> float:
        if self.neutral is not None:
            return 1.0
        samples = self.calibrator.samples
        if not samples:
            return 0.0
        sample_progress = len(samples) / self.calibrator.min_samples
        time_progress = ((samples[-1][0] - samples[0][0]) / self.calibrator.seconds
                         if len(samples) > 1 else 0.0)
        return max(0.0, min(0.99, min(sample_progress, time_progress)))

    @staticmethod
    def message(observation: Observation, calibrated: bool) -> str:
        if calibrated:
            return "ready"
        return {"no_face": "face_not_found", "multiple_faces": "multiple_faces",
                "face_too_small": "move_closer", "pose_out_of_range": "look_forward",
                "invalid_geometry": "look_forward"}.get(observation.reason, "hold_forward")


class DmsEngine:
    """Own model state and convert one BGR frame into a stable product result."""

    def __init__(self, analyzer, config: Config | None = None, eye_model=None,
                 action_model=None, action_smoother=None, action_interval: int = 5,
                 experimental_action: bool = False):
        if action_interval <= 0:
            raise ValueError("action_interval must be positive")
        if action_model is not None and not experimental_action:
            raise ValueError("Action model requires experimental_action=True")
        self.analyzer = analyzer
        self.base_config = config or Config()
        self.eye_model = eye_model
        self.action_model = action_model
        self.action_smoother = action_smoother
        self.action_interval = action_interval
        self.experimental_action = experimental_action
        self.frame_index = 0
        self.reset_calibration()

    @classmethod
    def from_models(cls, face_model: Path, config: Config | None = None,
                    eye_model: Path | None = None, eye_device: str = "CPU",
                    action_checkpoint: Path | None = None, action_backend: str = "auto",
                    action_device: str = "auto", action_threshold: float | None = None,
                    experimental_action: bool = False, action_interval: int = 5):
        from .vision import MediaPipeFaceAnalyzer

        analyzer = MediaPipeFaceAnalyzer(face_model)
        eye = None
        action = smoother = None
        if eye_model:
            from .eye import OpenVINOEyeClassifier
            eye = OpenVINOEyeClassifier(eye_model, eye_device)
        if action_checkpoint:
            if not experimental_action:
                analyzer.close()
                raise ValueError("Action checkpoint is allowed only in experimental mode")
            from .action import ActionSmoother, load_action_model
            action = load_action_model(action_checkpoint, action_backend,
                                       action_threshold, action_device)
            smoother = ActionSmoother()
        return cls(analyzer, config, eye, action, smoother, action_interval,
                   experimental_action)

    def reset_calibration(self):
        self.calibration = CalibrationService(self.base_config)
        self.engine = StateEngine(self.base_config)
        self.eye_history = EyeHistory(max_gap=self.base_config.max_gap_seconds)
        if getattr(self, "action_smoother", None):
            self.action_smoother.reset()

    @property
    def calibrator(self):
        return self.calibration.calibrator

    def process(self, frame, timestamp_s: float) -> EngineResult:
        observation = self.analyzer.analyze(frame, timestamp_s)
        valid = observation.valid
        neutral = self.calibration.update(observation, timestamp_s)
        if neutral is not None and self.engine.config == self.base_config:
            self.engine = StateEngine(neutral.config(self.base_config))
        yaw = angle_difference(observation.yaw, neutral.yaw) if valid and neutral else None
        pitch = angle_difference(observation.pitch, neutral.pitch) if valid and neutral else None
        gaze_x = (observation.iris_x - neutral.iris_x
                  if valid and neutral and observation.iris_x is not None
                  and neutral.iris_x is not None else None)
        gaze_y = (observation.iris_y - neutral.iris_y
                  if valid and neutral and observation.iris_y is not None
                  and neutral.iris_y is not None else None)
        usable = valid and neutral is not None
        eye_scores = self.eye_model.predict(frame, observation.points) \
            if self.eye_model and usable else None
        eye_closed_score = eye_scores.closed_score if eye_scores else None
        features = Features(timestamp_s, observation.ear, yaw, usable, pitch, gaze_x, gaze_y,
                            observation.mar, eye_closed_score, self.eye_model is not None)
        base_state = self.engine.update(features)
        source = "mediapipe_ear_pose_iris_v1" if gaze_x is not None else "mediapipe_ear_pose_v1"
        state = DriverState(timestamp_s, base_state.drowsiness, base_state.attention,
                            action="unknown", source=source,
                            yawn=base_state.yawn, eye=base_state.eye)
        action_scores = {}
        if self.action_model and (usable or not self.action_model.requires_face):
            if self.frame_index % self.action_interval == 0:
                raw, scores = self.action_model.predict(frame)
                action, action_scores = self.action_smoother.update(raw, scores)
            else:
                action, action_scores = self.action_smoother.current()
            state = DriverState(timestamp_s, state.drowsiness, state.attention, action,
                                state.source, state.yawn, state.eye)
        elif self.action_smoother:
            self.action_smoother.reset()
        closed = observation.ear < self.engine.config.ear_close if usable else None
        fraction, coverage = self.eye_history.update(timestamp_s, closed)
        self.frame_index += 1
        warning = (state.drowsiness == "suspected" or state.attention == "off_forward"
                   or state.yawn == "yawning")
        unknown = (state.drowsiness == "unknown" or state.attention == "unknown")
        return EngineResult(
            timestamp_s, state, observation, neutral is not None,
            self.calibration.progress(), self.calibration.message(observation, neutral is not None),
            warning, unknown, fraction, coverage, eye_closed_score,
            yaw, pitch, gaze_x, gaze_y, usable, closed, eye_scores, action_scores,
            tuple(getattr(self.action_model, "last_detections", ())),
        )

    def close(self):
        self.analyzer.close()
