"""Reject private media, local paths and common secret material in a release tree."""

import argparse
import re
from pathlib import Path


BLOCKED_SUFFIXES = {".mp4", ".avi", ".mov", ".csv", ".jsonl", ".pptx", ".docx"}
TEXT_SUFFIXES = {".txt", ".md", ".json", ".yaml", ".yml", ".ini", ".cfg", ".py"}
PATTERNS = {
    "local Linux path": re.compile(r"/home/[^/\s]+/"),
    "local Windows profile": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credential assignment": re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]+"),
}


def audit(root: Path) -> list[str]:
    if not root.is_dir():
        raise ValueError(f"Release directory not found: {root}")
    errors = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            errors.append(f"blocked file type: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size < 5_000_000:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label}: {relative}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        errors = audit(args.root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if errors:
        parser.exit(2, "Release audit failed:\n" + "\n".join(errors) + "\n")
    print("Release audit passed")


if __name__ == "__main__":
    main()
