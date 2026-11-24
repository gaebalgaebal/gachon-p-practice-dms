"""Pure helpers for subject-disjoint webcam action dataset preparation."""

import math


ACTION_LABELS = {"normal", "phone", "drink"}
SPLITS = {"train", "val", "test"}


def validate_sessions(sessions: list[dict]) -> None:
    if not sessions:
        raise ValueError("At least one session is required")
    subject_splits, session_ids = {}, set()
    for item in sessions:
        required = {"subject_id", "session_id", "split", "video", "schedule"}
        if not isinstance(item, dict) or any(not str(item.get(key, "")).strip() for key in required):
            raise ValueError(f"Session requires {sorted(required)}: {item}")
        split = item["split"]
        if split not in SPLITS:
            raise ValueError(f"Invalid split: {split}")
        subject = item["subject_id"]
        if subject in subject_splits and subject_splits[subject] != split:
            raise ValueError(f"Subject crosses splits: {subject}")
        subject_splits[subject] = split
        if item["session_id"] in session_ids:
            raise ValueError(f"Duplicate session_id: {item['session_id']}")
        session_ids.add(item["session_id"])
    if set(subject_splits.values()) != SPLITS:
        raise ValueError("Sessions must include train, val and test subjects")


def sample_times(schedule, sample_fps: float):
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive and finite")
    step = 1 / sample_fps
    for start, end, label in schedule:
        if label not in ACTION_LABELS:
            continue
        index = math.ceil((start - 1e-9) / step)
        while index * step < end - 1e-9:
            yield index * step, label
            index += 1
