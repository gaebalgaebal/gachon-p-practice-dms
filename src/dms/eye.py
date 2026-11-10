"""OpenVINO open-closed-eye-0001 adapter and MediaPipe eye cropping."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class EyeScores:
    left_open: float
    left_closed: float
    right_open: float
    right_closed: float

    @property
    def closed_score(self) -> float:
        return (self.left_closed + self.right_closed) / 2


def extract_eye_patch(frame, points, indices, padding: float = 1.6):
    """Return a square BGR eye crop around the supplied landmark indices."""
    import numpy as np

    if padding <= 1:
        raise ValueError("Eye crop padding must be greater than 1")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or max(indices) >= len(points):
        return None
    eye = points[list(indices)]
    if not np.isfinite(eye).all():
        return None
    center = eye.mean(axis=0)
    width, height = np.ptp(eye, axis=0)
    side = max(width, height * 2, 6) * padding
    x0, y0 = np.floor(center - side / 2).astype(int)
    x1, y1 = np.ceil(center + side / 2).astype(int)
    frame_height, frame_width = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame_width, x1), min(frame_height, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return frame[y0:y1, x0:x1]


class OpenVINOEyeClassifier:
    """Run the official 32x32 BGR eye-state model for both eyes."""

    def __init__(self, model_path: Path, device: str = "CPU"):
        if not model_path.is_file():
            raise FileNotFoundError(f"Eye model not found: {model_path}")
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(model_path.resolve()))
        if list(model.input(0).shape) != [1, 3, 32, 32]:
            raise ValueError(f"Unsupported eye model input shape: {list(model.input(0).shape)}")
        self.compiled = core.compile_model(model, device)
        self.output = self.compiled.output(0)

    def predict_patch(self, patch) -> tuple[float, float]:
        """Return ``(open_score, closed_score)`` for an already cropped eye."""
        import cv2
        import numpy as np

        image = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR)
        # The distributed ONNX expects the Model Zoo conversion preprocessing:
        # (BGR - 127) / 255. Converted OpenVINO IR would embed these operations.
        blob = (np.transpose(image, (2, 0, 1))[None].astype(np.float32) - 127.0) / 255.0
        scores = np.asarray(self.compiled([blob])[self.output], dtype=float).reshape(-1)
        if scores.size != 2 or not np.isfinite(scores).all():
            raise RuntimeError("Eye model returned invalid scores")
        if min(scores) < 0 or max(scores) > 1 or abs(float(scores.sum()) - 1) > .05:
            raise RuntimeError("Eye model output is not a two-class softmax")
        # The distributed ONNX was verified with all three official OMZ example
        # images: index 0 behaves as closed and index 1 as open. Return the
        # adapter's stable public order (open, closed).
        return float(scores[1]), float(scores[0])

    def predict(self, frame, points) -> EyeScores | None:
        from .vision import LEFT_EYE, RIGHT_EYE

        left = extract_eye_patch(frame, points, LEFT_EYE)
        right = extract_eye_patch(frame, points, RIGHT_EYE)
        if left is None or right is None:
            return None
        left_open, left_closed = self.predict_patch(left)
        right_open, right_closed = self.predict_patch(right)
        values = (left_open, left_closed, right_open, right_closed)
        if not all(isfinite(value) for value in values):
            return None
        return EyeScores(*values)
