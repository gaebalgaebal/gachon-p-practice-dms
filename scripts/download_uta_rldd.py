#!/usr/bin/env python3
"""Download large public Google Drive files with bounded range requests.

Google Drive can return a quota page for a whole-file request while still
serving explicit byte ranges. This downloader validates every Content-Range,
keeps a resumable .part file, and only publishes a ZIP after integrity checks.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
DEFAULT_CHUNK_SIZE = 128 * 1024 * 1024


def probe_size(file_id: str) -> int:
    request = urllib.request.Request(
        download_url(file_id),
        headers={
            "Accept-Encoding": "identity",
            "Range": "bytes=0-0",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
        if response.status != 206 or match is None:
            raise RuntimeError("server did not return a valid size probe")
        return int(match.group(3))


def download_url(file_id: str) -> str:
    return (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )


def download_range(
    file_id: str,
    target: Path,
    start: int,
    end: int,
    total: int,
    retries: int,
) -> None:
    expected = end - start + 1
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                download_url(file_id),
                headers={
                    "Accept-Encoding": "identity",
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "Mozilla/5.0",
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                header = response.headers.get("Content-Range", "")
                match = CONTENT_RANGE.fullmatch(header)
                actual_range = tuple(map(int, match.groups())) if match else None
                if response.status != 206 or actual_range != (start, end, total):
                    raise RuntimeError(
                        f"unexpected response: status={response.status}, "
                        f"Content-Range={header!r}"
                    )

                written = 0
                with target.open("r+b") as output:
                    output.seek(start)
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        written += len(block)
                    output.flush()
                    os.fsync(output.fileno())

                if written != expected:
                    raise RuntimeError(
                        f"short range: expected={expected}, received={written}"
                    )
                return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            with target.open("r+b") as output:
                output.truncate(start)
            if attempt == retries:
                raise RuntimeError(
                    f"range {start}-{end} failed after {retries} attempts"
                ) from exc
            wait_seconds = min(60, 5 * attempt)
            print(
                f"retry {attempt}/{retries} for {start}-{end}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait_seconds)


def validate_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_member}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Google Drive file ID")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()

    final_path = args.output.resolve()
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    total = probe_size(args.id)

    if final_path.exists():
        if final_path.stat().st_size != total:
            raise RuntimeError(f"existing final file has the wrong size: {final_path}")
        validate_zip(final_path)
        print(f"ALREADY_DONE {final_path.name} {total}", flush=True)
        return 0

    part_path.touch(exist_ok=True)
    current = part_path.stat().st_size
    if current > total:
        raise RuntimeError(f"partial file is larger than the source: {part_path}")

    chunk_size = args.chunk_mib * 1024 * 1024
    print(f"START {final_path.name} {current}/{total}", flush=True)
    while current < total:
        end = min(total - 1, current + chunk_size - 1)
        download_range(args.id, part_path, current, end, total, args.retries)
        current = end + 1
        print(
            f"PROGRESS {final_path.name} {current}/{total} "
            f"({current / total:.1%})",
            flush=True,
        )

    validate_zip(part_path)
    part_path.replace(final_path)
    print(f"DONE {final_path.name} {total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
