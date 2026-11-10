"""YawDD filename parsing, subject splitting, and interval validation."""

from dataclasses import dataclass
from pathlib import Path
import random
import re


@dataclass(frozen=True)
class YawDDVideo:
    subject_id: str
    subject_number: int
    gender: str
    camera: str
    appearance: str
    activity: str
    coarse_label: str
    annotation_status: str


def _strip_avi_suffixes(name: str) -> str:
    while name.lower().endswith(".avi"):
        name = name[:-4]
    return name.strip()


def parse_yawdd_path(path: Path | str) -> YawDDVideo:
    """Parse the camera, identity, appearance, and filename-level activity."""
    path = Path(path)
    parts_lower = {part.lower() for part in path.parts}
    camera = "mirror" if "mirror" in parts_lower else "dash" if "dash" in parts_lower else None
    if camera is None:
        raise ValueError(f"YawDD path must be below Mirror or Dash: {path}")

    stem = _strip_avi_suffixes(path.name)
    match = re.match(r"^(\d+)-(Male|Female)(.+?)(?:-(.+))?$", stem, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid YawDD filename: {path.name}")
    number = int(match.group(1))
    gender = match.group(2).lower()
    appearance = match.group(3).strip().lower()
    raw_activity = (match.group(4) or "sequence").strip().lower()
    activity = raw_activity.replace("&", "_").replace(" ", "_")
    if activity not in {"normal", "talking", "yawning", "talking_yawning", "sequence"}:
        raise ValueError(f"Unknown YawDD activity '{raw_activity}': {path.name}")

    # Mirror female/male numbering overlaps, so gender is part of the identity.
    subject_id = f"yawdd:{gender}:{number:02d}"
    if activity in {"yawning", "talking_yawning"}:
        coarse_label, status = "yawn", "interval_required"
    elif activity in {"normal", "talking"}:
        coarse_label, status = "normal", "filename_label"
    else:
        coarse_label, status = "unknown", "interval_required"
    return YawDDVideo(subject_id, number, gender, camera, appearance, activity,
                      coarse_label, status)


def split_subjects(subjects, train_ratio: float = .70, val_ratio: float = .15,
                   seed: int = 42) -> dict[str, str]:
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


def validate_interval(start_s: float, end_s: float, duration_s: float) -> None:
    if start_s < 0 or end_s <= start_s:
        raise ValueError("Interval must satisfy 0 <= start_s < end_s")
    if duration_s <= 0 or end_s > duration_s + 1e-3:
        raise ValueError("Interval end exceeds video duration")
