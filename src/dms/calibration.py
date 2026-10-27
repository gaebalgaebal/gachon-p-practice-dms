"""Session-local neutral pose / open-eye calibration and weighted eye history."""

from collections import deque
from dataclasses import dataclass, replace
from math import isfinite
from statistics import median, pstdev

from .state import Config


def angle_difference(value: float, reference: float) -> float:
    return (value - reference + 180) % 360 - 180


@dataclass(frozen=True)
class Neutral:
    ear_open_median: float
    yaw: float
    pitch: float
    iris_x: float | None = None
    iris_y: float | None = None

    def config(self, original: Config) -> Config:
        return replace(original, ear_close=self.ear_open_median * .65,
                       ear_open=self.ear_open_median * .80)


class Calibrator:
    def __init__(self, seconds: float = 2.0, min_samples: int = 15, max_gap: float = .75):
        self.seconds, self.min_samples, self.max_gap = seconds, min_samples, max_gap
        self.samples = []
        self.neutral: Neutral | None = None

    def reset(self):
        self.samples.clear()
        self.neutral = None

    def update(self, timestamp: float, ear: float | None, yaw: float | None,
               pitch: float | None, iris_x: float | None = None,
               iris_y: float | None = None) -> Neutral | None:
        if self.neutral:
            return self.neutral
        if any(v is None or not isfinite(v) for v in (timestamp, ear, yaw, pitch)):
            self.samples.clear()
            return None
        if not .1 <= ear <= .6:
            self.samples.clear()
            return None
        if self.samples and (timestamp - self.samples[-1][0] > self.max_gap
                             or timestamp <= self.samples[-1][0]):
            self.samples.clear()
        self.samples.append((timestamp, ear, yaw, pitch, iris_x, iris_y))
        # Retain a rolling window so an early head turn cannot block calibration forever.
        while len(self.samples) > 1 and timestamp - self.samples[1][0] >= self.seconds:
            self.samples.pop(0)
        if len(self.samples) >= self.min_samples and timestamp - self.samples[0][0] >= self.seconds:
            values = list(zip(*self.samples))
            if pstdev(values[2]) < 5 and pstdev(values[3]) < 5 and pstdev(values[1]) < .04:
                iris_seen = any(v is not None and isfinite(v)
                                for axis in values[4:6] for v in axis)
                complete_x = all(v is not None and isfinite(v) for v in values[4])
                complete_y = all(v is not None and isfinite(v) for v in values[5])
                stable_x = complete_x and pstdev(values[4]) < .08
                stable_y = complete_y and pstdev(values[5]) < .10
                # If iris landmarks exist, require a steady forward gaze instead of
                # silently locking a head-only baseline after eye movement.
                if iris_seen and not (stable_x and stable_y):
                    return None
                self.neutral = Neutral(
                    median(values[1]), median(values[2]), median(values[3]),
                    median(values[4]) if stable_x else None,
                    median(values[5]) if stable_y else None,
                )
        return self.neutral


class EyeHistory:
    """Time-weighted closed-eye fraction; not a validated clinical PERCLOS."""
    def __init__(self, window_seconds: float = 30, max_gap: float = .75):
        self.window_seconds, self.max_gap = window_seconds, max_gap
        self.intervals = deque()
        self.previous = None

    def update(self, t: float, closed: bool | None) -> tuple[float | None, float]:
        if closed is None:
            self.intervals.clear()
            self.previous = None
            return None, 0.0
        if self.previous:
            previous_t, previous_closed = self.previous
            dt = t - previous_t
            if 0 < dt <= self.max_gap:
                self.intervals.append((previous_t, t, previous_closed))
            else:
                self.intervals.clear()
        self.previous = (t, closed)
        cutoff = t - self.window_seconds
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        observed = sum(end - max(start, cutoff) for start, end, _ in self.intervals)
        shut = sum(end - max(start, cutoff) for start, end, value in self.intervals if value)
        return (shut / observed if observed else None), observed / self.window_seconds
