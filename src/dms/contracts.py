"""Contracts for future camera, face and action adapters."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .vision import Observation


@dataclass(frozen=True)
class Features:
    timestamp_s: float
    ear: float | None
    yaw_degrees: float | None
    face_valid: bool = True
    pitch_degrees: float | None = None
    gaze_x_ratio: float | None = None
    gaze_y_ratio: float | None = None


@dataclass(frozen=True)
class DriverState:
    timestamp_s: float
    drowsiness: str
    attention: str
    action: str = "unknown"
    source: str = "feature_rules_v0"


class FaceAnalyzer(Protocol):
    def analyze(self, frame: object, timestamp_s: float) -> "Observation":
        """Return a fresh observation; calibration converts it to Features."""
        ...


class ActionClassifier(Protocol):
    def predict(self, frame: object) -> tuple[str, dict[str, float]]:
        """Return label and scores; softmax is not calibrated confidence."""
        ...
