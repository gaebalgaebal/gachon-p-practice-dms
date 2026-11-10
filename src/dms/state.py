"""Timestamp-based DMS rules with dataset-validated baseline thresholds."""

from dataclasses import dataclass
from math import isfinite

from .contracts import DriverState, Features


@dataclass(frozen=True)
class Config:
    ear_close: float = 0.20
    ear_open: float = 0.24
    closed_seconds: float = 1.5
    yaw_limit_degrees: float = 25.0
    away_seconds: float = 1.5
    max_gap_seconds: float = 0.75
    face_loss_grace_seconds: float = 2.0
    pitch_limit_degrees: float = 20.0
    gaze_x_limit_ratio: float = 0.12
    gaze_y_limit_ratio: float = 0.18
    mar_yawn: float = 0.35
    mar_normal: float = 0.25
    yawn_seconds: float = 0.8
    eye_model_threshold: float = 0.70

    def __post_init__(self):
        if not all(isfinite(v) for v in vars(self).values()):
            raise ValueError("Configuration values must be finite")
        if not 0 < self.ear_close < self.ear_open < 1:
            raise ValueError("Require 0 < ear_close < ear_open < 1")
        if min(self.closed_seconds, self.away_seconds, self.max_gap_seconds,
               self.face_loss_grace_seconds, self.yawn_seconds) <= 0:
            raise ValueError("Durations must be positive")
        if not 0 < self.yaw_limit_degrees < 180:
            raise ValueError("Yaw limit must be between 0 and 180")
        if not 0 < self.pitch_limit_degrees < 180:
            raise ValueError("Pitch limit must be between 0 and 180")
        if not 0 < self.gaze_x_limit_ratio <= 1:
            raise ValueError("Horizontal gaze limit must be in (0, 1]")
        if not 0 < self.gaze_y_limit_ratio <= 1:
            raise ValueError("Vertical gaze limit must be in (0, 1]")
        if not 0 < self.mar_normal < self.mar_yawn < 2:
            raise ValueError("Require 0 < mar_normal < mar_yawn < 2")
        if not .5 < self.eye_model_threshold <= 1:
            raise ValueError("Eye model threshold must be in (.5, 1]")


class StateEngine:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._last: float | None = None
        self._reset()

    def _reset(self):
        self._closed_at: float | None = None
        self._away_at: float | None = None
        self._eye_closed: bool | None = None
        self._yawn_at: float | None = None
        self._mouth_open: bool | None = None
        self._last_face_at: float | None = None

    def update(self, f: Features) -> DriverState:
        t, c = f.timestamp_s, self.config
        if not isfinite(t) or t < 0:
            raise ValueError("Timestamp must be finite and nonnegative")
        if self._last is not None:
            if t <= self._last:
                raise ValueError("Timestamps must be strictly increasing")
            if t - self._last > c.max_gap_seconds:
                self._reset()
        self._last = t
        if not f.face_valid:
            # A sharp head turn can briefly defeat the landmark detector. Preserve only
            # an already-started attention-away timer; never infer away from face loss
            # alone. Eye and mouth timers are cleared so occlusion cannot count as an
            # eye closure or yawn.
            self._closed_at = None
            self._eye_closed = None
            self._yawn_at = None
            self._mouth_open = None
            recent_away = (self._away_at is not None and self._last_face_at is not None
                           and t - self._last_face_at <= c.face_loss_grace_seconds)
            if recent_away:
                attention = ("off_forward" if t - self._away_at >= c.away_seconds
                             else "unknown")
                return DriverState(t, "unknown", attention)
            self._reset()
            return DriverState(t, "unknown", "unknown")
        self._last_face_at = t

        drowsiness = "unknown"
        eye = "unknown"
        if f.ear is None or not isfinite(f.ear) or not 0 <= f.ear <= 1:
            self._closed_at = None
            self._eye_closed = None
        else:
            if f.ear < c.ear_close:
                self._eye_closed = True
            elif f.ear >= c.ear_open:
                self._eye_closed = False
            combined_closed = self._eye_closed
            if f.eye_model_active:
                if (f.eye_closed_score is None or not isfinite(f.eye_closed_score)
                        or not 0 <= f.eye_closed_score <= 1):
                    combined_closed = None
                elif f.eye_closed_score >= c.eye_model_threshold:
                    combined_closed = True if self._eye_closed is True else None
                elif f.eye_closed_score <= 1 - c.eye_model_threshold:
                    combined_closed = False if self._eye_closed is False else None
                else:
                    combined_closed = None
            eye = ("closed" if combined_closed is True else
                   "open" if combined_closed is False else "uncertain")
            if combined_closed is True:
                if self._closed_at is None:
                    self._closed_at = t
                drowsiness = (
                    "suspected" if t - self._closed_at >= c.closed_seconds else "normal"
                )
            elif combined_closed is False:
                self._closed_at = None
                drowsiness = "normal"
            else:
                self._closed_at = None

        yawn = "unknown"
        if f.mar is None or not isfinite(f.mar) or not 0 <= f.mar <= 2:
            self._yawn_at = None
            self._mouth_open = None
        else:
            if f.mar >= c.mar_yawn:
                self._mouth_open = True
            elif f.mar <= c.mar_normal:
                self._mouth_open = False
            if self._mouth_open is True:
                if self._yawn_at is None:
                    self._yawn_at = t
                yawn = "yawning" if t - self._yawn_at >= c.yawn_seconds else "normal"
            elif self._mouth_open is False:
                self._yawn_at = None
                yawn = "normal"

        attention = "unknown"
        if f.yaw_degrees is None or not isfinite(f.yaw_degrees) or abs(f.yaw_degrees) > 180:
            self._away_at = None
        elif f.pitch_degrees is not None and (
            not isfinite(f.pitch_degrees) or abs(f.pitch_degrees) > 180
        ):
            self._away_at = None
        elif ((f.gaze_x_ratio is not None and (
                not isfinite(f.gaze_x_ratio) or abs(f.gaze_x_ratio) > 1))
              or (f.gaze_y_ratio is not None and (
                  not isfinite(f.gaze_y_ratio) or abs(f.gaze_y_ratio) > 1))):
            self._away_at = None
        elif (abs(f.yaw_degrees) > c.yaw_limit_degrees or (
                f.pitch_degrees is not None and isfinite(f.pitch_degrees)
                and abs(f.pitch_degrees) > c.pitch_limit_degrees
              ) or (
                f.gaze_x_ratio is not None
                and abs(f.gaze_x_ratio) > c.gaze_x_limit_ratio
              ) or (
                f.gaze_y_ratio is not None
                and abs(f.gaze_y_ratio) > c.gaze_y_limit_ratio
              )):
            if self._away_at is None:
                self._away_at = t
            # A turned head is pending until sustained; do not call it forward.
            attention = "off_forward" if t - self._away_at >= c.away_seconds else "unknown"
        else:
            self._away_at = None
            attention = "forward"
        return DriverState(t, drowsiness, attention, yawn=yawn, eye=eye)
