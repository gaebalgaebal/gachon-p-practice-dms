"""MRL Eye filename parsing, subject splitting, and evaluation helpers."""

from dataclasses import dataclass
from pathlib import Path
import random


@dataclass(frozen=True)
class MRLAnnotation:
    subject_id: str
    image_id: str
    gender: int
    glasses: int
    eye_state: int
    reflections: int
    lighting: int
    sensor_id: str

    @property
    def label(self) -> str:
        return "open" if self.eye_state == 1 else "closed"


def parse_mrl_filename(path: Path | str) -> MRLAnnotation:
    """Parse the eight underscore-separated attributes defined by MRL."""
    stem = Path(path).stem
    fields = stem.split("_")
    if len(fields) != 8:
        raise ValueError(f"Invalid MRL filename (expected 8 fields): {stem}")
    subject, image_id, *values, sensor = fields
    if not subject.startswith("s") or not subject[1:].isdigit() or not image_id.isdigit():
        raise ValueError(f"Invalid MRL subject/image ID: {stem}")
    try:
        gender, glasses, eye_state, reflections, lighting = map(int, values)
    except ValueError as exc:
        raise ValueError(f"Non-numeric MRL attributes: {stem}") from exc
    allowed = ((gender, {0, 1}, "gender"), (glasses, {0, 1}, "glasses"),
               (eye_state, {0, 1}, "eye_state"),
               (reflections, {0, 1, 2}, "reflections"),
               (lighting, {0, 1}, "lighting"))
    for value, choices, name in allowed:
        if value not in choices:
            raise ValueError(f"Invalid MRL {name}={value}: {stem}")
    if sensor not in {"01", "02", "03"}:
        raise ValueError(f"Invalid MRL sensor_id={sensor}: {stem}")
    return MRLAnnotation(subject, image_id, gender, glasses, eye_state,
                         reflections, lighting, sensor)


def split_subjects(subjects, train_ratio: float = .70, val_ratio: float = .15,
                   seed: int = 42) -> dict[str, str]:
    """Assign each subject to exactly one deterministic split."""
    unique = sorted(set(subjects))
    if len(unique) < 3:
        raise ValueError("At least three subjects are required for train/val/test")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must leave positive train, val, and test portions")
    random.Random(seed).shuffle(unique)
    train_count = max(1, round(len(unique) * train_ratio))
    val_count = max(1, round(len(unique) * val_ratio))
    if train_count + val_count >= len(unique):
        train_count = len(unique) - val_count - 1
    return {subject: ("train" if index < train_count else
                      "val" if index < train_count + val_count else "test")
            for index, subject in enumerate(unique)}


def label_from_scores(open_score: float, closed_score: float,
                      threshold: float = .70) -> str:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if max(open_score, closed_score) < threshold:
        return "unknown"
    return "open" if open_score >= closed_score else "closed"
