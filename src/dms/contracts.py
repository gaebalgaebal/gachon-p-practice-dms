"""Contracts for future camera, face and action adapters."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Features:
    timestamp_s: float
    ear: float | None
    yaw_degrees: float | None
    face_valid: bool = True


@dataclass(frozen=True)
class DriverState:
    timestamp_s: float
    drowsiness: str
    attention: str
    action: str = "unknown"
    source: str = "feature_rules_v0"


class FaceAnalyzer(Protocol):
    def analyze(self, frame: object, timestamp_s: float) -> Features:
        """Return missing measurements as None; never reuse stale features."""
        ...


class ActionClassifier(Protocol):
    def predict(self, frame: object, timestamp_s: float) -> dict[str, float]:
        """Return calibrated per-class scores once an actual model exists."""
        ...
