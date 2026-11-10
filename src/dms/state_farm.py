"""State Farm class mapping and deterministic driver-level splitting."""

from dataclasses import dataclass
import random


SOURCE_CLASSES = {
    "c0": ("normal_driving", "normal"),
    "c1": ("texting_right", "phone"),
    "c2": ("talking_phone_right", "phone"),
    "c3": ("texting_left", "phone"),
    "c4": ("talking_phone_left", "phone"),
    "c5": ("operating_radio", None),
    "c6": ("drinking", "drink"),
    "c7": ("reaching_behind", None),
    "c8": ("hair_makeup", None),
    "c9": ("talking_passenger", None),
}


@dataclass(frozen=True)
class StateFarmLabel:
    source_class: str
    source_activity: str
    action_label: str | None


def map_source_class(source_class: str) -> StateFarmLabel:
    if source_class not in SOURCE_CLASSES:
        raise ValueError(f"Unknown State Farm class: {source_class}")
    activity, action = SOURCE_CLASSES[source_class]
    return StateFarmLabel(source_class, activity, action)


def split_subjects(subjects, train_ratio: float = .70, val_ratio: float = .15,
                   seed: int = 42) -> dict[str, str]:
    unique = sorted(set(subjects))
    if len(unique) < 3:
        raise ValueError("At least three drivers are required for train/val/test")
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
