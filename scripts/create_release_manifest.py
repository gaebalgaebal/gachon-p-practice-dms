"""Create a deterministic SHA-256 manifest for release files."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def create(root: Path, output: Path | None = None):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Release directory not found: {root}")
    output = output or root / "release-manifest.json"
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.resolve() != output.resolve():
            files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                          "sha256": digest(path)})
    payload = {"product": "GachonDMS", "version": "0.4.0",
               "created_at": datetime.now(timezone.utc).isoformat(), "files": files}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        payload = create(args.root, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Manifested {len(payload['files'])} files")


if __name__ == "__main__":
    main()
