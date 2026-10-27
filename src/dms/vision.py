"""Local MediaPipe Tasks adapter. No image is sent to a remote service."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


def eye_aspect_ratio(points: np.ndarray) -> float | None:
    """Six pixel-space points: horizontal corners and two vertical pairs."""
    points = np.asarray(points, dtype=float)
    if points.shape != (6, 2) or not np.isfinite(points).all():
        return None
    width = np.linalg.norm(points[0] - points[3])
    if width < 2:
        return None
    return float((np.linalg.norm(points[1] - points[5])
                  + np.linalg.norm(points[2] - points[4])) / (2 * width))


def head_angles(matrix: np.ndarray) -> tuple[float, float, float] | None:
    """ZYX Euler angles (yaw, pitch, roll) from the canonical face transform."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    u, singular, vt = np.linalg.svd(matrix[:3, :3])
    if min(singular) < 1e-6:
        return None
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        return None
    yaw = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
    roll = np.arctan2(rotation[1, 0], rotation[0, 0])
    return tuple(float(np.degrees(a)) for a in (yaw, pitch, roll))


def iris_position(points: np.ndarray) -> tuple[float, float] | None:
    """Mean screen-relative iris position, normalized inside both eye boxes."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 478 or points.shape[1] != 2:
        return None
    if not np.isfinite(points[[33, 133, 159, 145, 362, 263, 386, 374, 468, 473]]).all():
        return None
    ratios = []
    for iris, left, right, top, bottom in (
        (468, 33, 133, 159, 145),
        (473, 362, 263, 386, 374),
    ):
        x0, x1 = sorted((points[left, 0], points[right, 0]))
        y0, y1 = sorted((points[top, 1], points[bottom, 1]))
        if x1 - x0 < 2 or y1 - y0 < 1:
            continue
        ratios.append(((points[iris, 0] - x0) / (x1 - x0),
                       (points[iris, 1] - y0) / (y1 - y0)))
    if not ratios:
        return None
    return tuple(float(v) for v in np.mean(ratios, axis=0))


@dataclass
class Observation:
    valid: bool = False
    reason: str = "no_face"
    face_count: int = 0
    ear: float | None = None
    left_ear: float | None = None
    right_ear: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    iris_x: float | None = None
    iris_y: float | None = None
    points: np.ndarray | None = None


class MediaPipeFaceAnalyzer:
    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face model not found: {model_path}. Run: python scripts/download_model.py"
            )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=2,  # Detect ambiguity instead of silently choosing another person.
            min_face_detection_confidence=.6,
            min_face_presence_confidence=.6,
            min_tracking_confidence=.6,
            output_facial_transformation_matrixes=True,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def analyze(self, frame: np.ndarray, timestamp_s: float) -> Observation:
        timestamp_ms = max(self._last_ms + 1, round(timestamp_s * 1000))
        self._last_ms = timestamp_ms
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        count = len(result.face_landmarks)
        if count != 1:
            return Observation(reason="multiple_faces" if count > 1 else "no_face", face_count=count)
        h, w = frame.shape[:2]
        landmarks = result.face_landmarks[0]
        points = np.array([(p.x * w, p.y * h) for p in landmarks], dtype=float)
        box = np.ptp(points[:468], axis=0)
        if box[0] < 70 or box[1] < 70:
            return Observation(reason="face_too_small", face_count=count, points=points)
        left = eye_aspect_ratio(points[list(LEFT_EYE)])
        right = eye_aspect_ratio(points[list(RIGHT_EYE)])
        angles = head_angles(result.facial_transformation_matrixes[0]) if (
            result.facial_transformation_matrixes
        ) else None
        if left is None or right is None or angles is None:
            return Observation(reason="invalid_geometry", face_count=count, points=points)
        # Fail closed on severe pose; a 2D EAR is unreliable at large rotations.
        yaw, pitch, roll = angles
        if abs(yaw) > 65 or abs(pitch) > 55 or abs(roll) > 45:
            return Observation(reason="pose_out_of_range", face_count=count, points=points)
        iris = iris_position(points)
        iris_x, iris_y = iris if iris is not None else (None, None)
        return Observation(True, "ok", count, (left + right) / 2, left, right,
                           yaw, pitch, roll, iris_x, iris_y, points)

    def close(self):
        self.detector.close()
