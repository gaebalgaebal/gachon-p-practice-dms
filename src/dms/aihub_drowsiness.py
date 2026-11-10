"""Helpers for the AI Hub driver drowsiness image/JSON archives."""

from pathlib import PurePosixPath


ENVIRONMENTS = {
    "bbox(실제도로환경)": "real_road",
    "bbox(통제환경)": "controlled",
    "keypoint(준통제환경)": "semi_controlled",
}


def environment_from_archive(name: str) -> str:
    for token, environment in ENVIRONMENTS.items():
        if token in name:
            return environment
    raise ValueError(f"Unknown AI Hub drowsiness archive: {name}")


def subject_from_member(member: str, environment: str) -> str:
    parts = PurePosixPath(member).parts
    if len(parts) < 2:
        raise ValueError(f"Invalid archive member: {member}")
    folder = parts[-2]
    if environment == "real_road":
        fields = folder.split("_")
        if len(fields) < 2 or fields[0] != "R" or not fields[1].isdigit():
            raise ValueError(f"Invalid real-road subject folder: {folder}")
        identity = fields[1]
    elif environment in {"controlled", "semi_controlled"}:
        identity = folder.split("_")[1 if environment == "semi_controlled" else 0]
        if not identity.isdigit():
            raise ValueError(f"Invalid subject folder: {folder}")
    else:
        raise ValueError(f"Unknown environment: {environment}")
    return f"aihub_drowsiness:{environment}:{int(identity):03d}"


def controlled_labels(filename: str) -> tuple[str | None, str | None]:
    """Return drowsiness and direction labels encoded in controlled filenames."""
    stem = PurePosixPath(filename).stem
    fields = stem.split("_")
    if len(fields) < 9:
        return None, None
    state = fields[-4]
    direction = fields[-5]
    drowsiness = {"정상주시": "normal", "졸음재현": "drowsy"}.get(state)
    return drowsiness, direction if drowsiness else None


def eye_label(payload: dict) -> str | None:
    try:
        boxes = payload["ObjectInfo"]["BoundingBox"]
        eyes = (boxes["Leye"], boxes["Reye"])
        if not all(eye["isVisible"] for eye in eyes):
            return None
        states = [bool(eye["Opened"]) for eye in eyes]
    except (KeyError, TypeError):
        return None
    if states[0] != states[1]:
        return None
    return "open" if states[0] else "closed"


def session_from_filename(subject: str, filename: str) -> str:
    stem = PurePosixPath(filename).stem
    fields = stem.split("_")
    base = "_".join(fields[:-1]) if len(fields) > 1 else stem
    return f"{subject}:{base}"
