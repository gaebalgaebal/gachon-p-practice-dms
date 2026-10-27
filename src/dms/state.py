"""Timestamp-based baseline rules; thresholds are unvalidated demo settings."""

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
    pitch_limit_degrees: float = 20.0
    gaze_x_limit_ratio: float = 0.12
    gaze_y_limit_ratio: float = 0.18

    def __post_init__(self):
        if not all(isfinite(v) for v in vars(self).values()):
            raise ValueError("Configuration values must be finite")
        if not 0 < self.ear_close < self.ear_open < 1:
            raise ValueError("Require 0 < ear_close < ear_open < 1")
        if min(self.closed_seconds, self.away_seconds, self.max_gap_seconds) <= 0:
            raise ValueError("Durations must be positive")
        if not 0 < self.yaw_limit_degrees < 180:
            raise ValueError("Yaw limit must be between 0 and 180")
        if not 0 < self.pitch_limit_degrees < 180:
            raise ValueError("Pitch limit must be between 0 and 180")
        if not 0 < self.gaze_x_limit_ratio <= 1:
            raise ValueError("Horizontal gaze limit must be in (0, 1]")
        if not 0 < self.gaze_y_limit_ratio <= 1:
            raise ValueError("Vertical gaze limit must be in (0, 1]")


class StateEngine:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._last: float | None = None
        self._reset()

    def _reset(self):
        self._closed_at: float | None = None
        self._away_at: float | None = None
        self._eye_closed: bool | None = None

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
            self._reset()
            return DriverState(t, "unknown", "unknown")

        drowsiness = "unknown"
        if f.ear is None or not isfinite(f.ear) or not 0 <= f.ear <= 1:
            self._closed_at = None
            self._eye_closed = None
        else:
            if f.ear < c.ear_close:
                self._eye_closed = True
            elif f.ear >= c.ear_open:
                self._eye_closed = False
            if self._eye_closed is True:
                if self._closed_at is None:
                    self._closed_at = t
                drowsiness = (
                    "suspected" if t - self._closed_at >= c.closed_seconds else "normal"
                )
            elif self._eye_closed is False:
                self._closed_at = None
                drowsiness = "normal"

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
        return DriverState(t, drowsiness, attention)
