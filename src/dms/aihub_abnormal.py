"""Helpers for AI Hub in-vehicle abnormal-action labels and YOLO boxes."""

import random


CATEGORY_MAP = {
    "차량제어": ("normal", None, None),
    "휴대폰조작": ("phone", "phone_use", 0),
    "통화": ("phone", "phone_call", 1),
}


def category_mapping(category_name: str) -> tuple[str, str | None, int | None]:
    if category_name not in CATEGORY_MAP:
        raise ValueError(f"Unsupported AI Hub abnormal category: {category_name}")
    return CATEGORY_MAP[category_name]


def split_validation_videos(video_ids, seed: int = 42) -> dict[str, str]:
    unique = sorted(set(video_ids))
    if len(unique) < 2:
        raise ValueError("At least two validation video IDs are required")
    random.Random(seed).shuffle(unique)
    val_count = max(1, len(unique) // 2)
    return {video_id: ("val" if index < val_count else "test")
            for index, video_id in enumerate(unique)}


def xywh_to_yolo(box, width: int, height: int) -> tuple[float, float, float, float]:
    if len(box) != 4 or width <= 0 or height <= 0:
        raise ValueError("Require an xywh box and positive image dimensions")
    x, y, w, h = map(float, box)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Body box is outside the image")
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height)
