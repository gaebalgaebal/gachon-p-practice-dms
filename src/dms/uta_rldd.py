"""UTA-RLDD path parsing, subject splitting, and label helpers."""

from dataclasses import dataclass
from pathlib import Path
import random
import re


LEVELS = {
    "0": (0, "alert"),
    "5": (5, "low_vigilance"),
    "10": (10, "drowsy"),
}


@dataclass(frozen=True)
class UTARLDDVideo:
    subject_id: str
    subject_number: int
    fold: int
    part: int
    level: int
    label: str
    segment: int


def parse_uta_path(path: Path | str) -> UTARLDDVideo:
    path = Path(path)
    if len(path.parts) < 3:
        raise ValueError(f"Invalid UTA-RLDD path: {path}")
    fold_part = path.parts[-3]
    match = re.fullmatch(r"Fold([1-5])_part([12])", fold_part, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid UTA-RLDD fold directory: {fold_part}")
    subject = path.parts[-2]
    if not subject.isdigit():
        raise ValueError(f"Invalid UTA-RLDD subject: {subject}")
    # Some archives contain a split recording such as 10_1.mp4 and 10_2.mp4.
    stem = path.stem
    label_match = re.fullmatch(r"(0|5|10)(?:_(\d+))?", stem)
    if not label_match:
        raise ValueError(f"Invalid UTA-RLDD level filename: {path.name}")
    level, label = LEVELS[label_match.group(1)]
    segment = int(label_match.group(2) or 1)
    number = int(subject)
    return UTARLDDVideo(f"uta_rldd:{number:02d}", number, int(match.group(1)),
                        int(match.group(2)), level, label, segment)


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


def binary_label(label: str) -> str:
    if label == "alert":
        return "alert"
    if label in {"low_vigilance", "drowsy"}:
        return "fatigued"
    if label == "unknown":
        return "unknown"
    raise ValueError(f"Unsupported UTA-RLDD label: {label}")
