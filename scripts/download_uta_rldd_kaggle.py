#!/usr/bin/env python3
"""Selectively extract UTA-RLDD parts from the public Kaggle mirror.

The Kaggle bundle is a large ZIP. This script reads its central directory with
HTTP range requests, downloads only selected members, validates their size and
CRC32, and writes them directly into the local raw-data layout.
"""

from __future__ import annotations

import argparse
import binascii
import os
import re
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


KAGGLE_API = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "rishab260/uta-reallife-drowsiness-dataset?datasetVersionNumber=1"
)
CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


@dataclass(frozen=True)
class Entry:
    name: str
    compressed_size: int
    uncompressed_size: int
    local_offset: int
    crc32: int
    method: int
    flags: int


def signed_archive_url() -> str:
    result = subprocess.run(
        [
            "curl",
            "-L",
            "-sS",
            "-o",
            "/dev/null",
            "--range",
            "0-0",
            "-w",
            "%{url_effective}",
            KAGGLE_API,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    url = result.stdout.strip()
    if not url.startswith("https://storage.googleapis.com"):
        raise RuntimeError("Kaggle did not return a signed archive URL")
    return url


def open_range(url: str, start: int, end: int, timeout: int = 180):
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
            "User-Agent": "Mozilla/5.0",
        },
    )
    response = urllib.request.urlopen(request, timeout=timeout)
    match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
    if response.status != 206 or match is None:
        response.close()
        raise RuntimeError("archive server did not honor the byte range")
    actual_start, actual_end, total = map(int, match.groups())
    if (actual_start, actual_end) != (start, end):
        response.close()
        raise RuntimeError("archive server returned an unexpected byte range")
    return response, total


def read_range(url: str, start: int, end: int) -> tuple[bytes, int]:
    response, total = open_range(url, start, end)
    with response:
        data = response.read()
    if len(data) != end - start + 1:
        raise RuntimeError("short archive range")
    return data, total


def zip_entries(url: str) -> tuple[list[Entry], int]:
    _, total = read_range(url, 0, 0)
    tail_start = max(0, total - 131_072)
    tail, _ = read_range(url, tail_start, total - 1)
    eocd_position = tail.rfind(b"PK\x05\x06")
    if eocd_position < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")

    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_position)
    entry_count, central_size, central_offset = eocd[4], eocd[5], eocd[6]
    if (
        entry_count == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    ):
        locator_position = tail.rfind(b"PK\x06\x07", 0, eocd_position)
        if locator_position < 0:
            raise RuntimeError("ZIP64 locator not found")
        _, _, zip64_offset, _ = struct.unpack_from(
            "<4sLQL", tail, locator_position
        )
        record, _ = read_range(url, zip64_offset, zip64_offset + 55)
        values = struct.unpack_from("<4sQ2H2L4Q", record)
        entry_count, central_size, central_offset = values[7], values[8], values[9]

    central, _ = read_range(
        url, central_offset, central_offset + central_size - 1
    )
    entries: list[Entry] = []
    position = 0
    while position < len(central):
        fields = struct.unpack_from("<4s6H3L5H2L", central, position)
        if fields[0] != b"PK\x01\x02":
            raise RuntimeError(f"bad central-directory signature at {position}")

        flags, method = fields[3], fields[4]
        crc32, compressed_size, uncompressed_size = fields[7:10]
        name_length, extra_length, comment_length = fields[10:13]
        disk_start, local_offset = fields[13], fields[16]
        variable_start = position + 46
        name_bytes = central[variable_start : variable_start + name_length]
        name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
        extra_start = variable_start + name_length
        extra = central[extra_start : extra_start + extra_length]

        extra_position = 0
        while extra_position + 4 <= len(extra):
            tag, size = struct.unpack_from("<HH", extra, extra_position)
            payload = extra[extra_position + 4 : extra_position + 4 + size]
            if tag == 0x0001:
                zip64_position = 0
                if uncompressed_size == 0xFFFFFFFF:
                    uncompressed_size = struct.unpack_from(
                        "<Q", payload, zip64_position
                    )[0]
                    zip64_position += 8
                if compressed_size == 0xFFFFFFFF:
                    compressed_size = struct.unpack_from(
                        "<Q", payload, zip64_position
                    )[0]
                    zip64_position += 8
                if local_offset == 0xFFFFFFFF:
                    local_offset = struct.unpack_from(
                        "<Q", payload, zip64_position
                    )[0]
                    zip64_position += 8
                if disk_start == 0xFFFF:
                    disk_start = struct.unpack_from(
                        "<L", payload, zip64_position
                    )[0]
            extra_position += 4 + size

        if disk_start != 0:
            raise RuntimeError("multi-disk ZIP archives are not supported")
        entries.append(
            Entry(
                name=name,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_offset=local_offset,
                crc32=crc32,
                method=method,
                flags=flags,
            )
        )
        position = extra_start + extra_length + comment_length

    if len(entries) != entry_count:
        raise RuntimeError(
            f"central-directory count mismatch: {len(entries)} != {entry_count}"
        )
    return entries, total


def output_path(root: Path, entry: Entry) -> Path:
    source = PurePosixPath(entry.name)
    if source.is_absolute() or ".." in source.parts or len(source.parts) < 3:
        raise RuntimeError(f"unsafe archive path: {entry.name}")
    # The Kaggle bundle repeats the part name, for example:
    # Fold1_part1/Fold1_part1/01/0.mov
    if source.parts[0] != source.parts[1]:
        raise RuntimeError(f"unexpected archive layout: {entry.name}")
    return root.joinpath(*source.parts[1:])


def extract_entry(url: str, entry: Entry, target: Path, retries: int) -> None:
    if entry.flags & 0x1:
        raise RuntimeError(f"encrypted ZIP member is unsupported: {entry.name}")
    if entry.method not in (0, 8):
        raise RuntimeError(f"unsupported compression method {entry.method}")
    if target.exists() and target.stat().st_size == entry.uncompressed_size:
        print(f"SKIP {entry.name}", flush=True)
        return

    header, _ = read_range(url, entry.local_offset, entry.local_offset + 29)
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header: {entry.name}")
    name_length, extra_length = fields[9], fields[10]
    data_start = entry.local_offset + 30 + name_length + extra_length
    data_end = data_start + entry.compressed_size - 1

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".download")
    for attempt in range(1, retries + 1):
        try:
            response, _ = open_range(url, data_start, data_end, timeout=300)
            decompressor = zlib.decompressobj(-15) if entry.method == 8 else None
            crc32 = 0
            written = 0
            compressed_read = 0
            with response, temporary.open("wb") as output:
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    compressed_read += len(block)
                    decoded = decompressor.decompress(block) if decompressor else block
                    output.write(decoded)
                    written += len(decoded)
                    crc32 = binascii.crc32(decoded, crc32)
                if decompressor:
                    decoded = decompressor.flush()
                    output.write(decoded)
                    written += len(decoded)
                    crc32 = binascii.crc32(decoded, crc32)
                output.flush()
                os.fsync(output.fileno())

            if compressed_read != entry.compressed_size:
                raise RuntimeError("short compressed member")
            if written != entry.uncompressed_size:
                raise RuntimeError("uncompressed size mismatch")
            if crc32 & 0xFFFFFFFF != entry.crc32:
                raise RuntimeError("CRC32 mismatch")
            temporary.replace(target)
            print(f"DONE {entry.name} {written}", flush=True)
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"failed to extract {entry.name} after {retries} attempts"
                ) from exc
            print(
                f"RETRY {entry.name} {attempt}/{retries}: {exc}",
                flush=True,
            )
            time.sleep(min(60, 5 * attempt))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--part", required=True)
    parser.add_argument(
        "--subjects",
        help="optional comma-separated subject IDs within the selected part",
    )
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    url = signed_archive_url()
    entries, archive_size = zip_entries(url)
    subjects = set(args.subjects.split(",")) if args.subjects else None
    selected = [
        entry
        for entry in entries
        if PurePosixPath(entry.name).parts[0] == args.part
        and (
            subjects is None
            or PurePosixPath(entry.name).parts[2] in subjects
        )
        and not entry.name.endswith("/")
    ]
    expected = 18 if subjects is None else 3 * len(subjects)
    if len(selected) != expected:
        raise RuntimeError(
            f"expected {expected} videos for {args.part}, found {len(selected)}"
        )
    selected.sort(key=lambda entry: entry.name)
    print(
        f"START {args.part} files={len(selected)} "
        f"compressed={sum(entry.compressed_size for entry in selected)} "
        f"archive={archive_size}",
        flush=True,
    )
    for index, entry in enumerate(selected, start=1):
        extract_entry(
            url,
            entry,
            output_path(args.output_root.resolve(), entry),
            args.retries,
        )
        print(f"PROGRESS {args.part} {index}/{len(selected)}", flush=True)
    print(f"PART_DONE {args.part}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
