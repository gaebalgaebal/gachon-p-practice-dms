"""Merge and validate YawDD interval annotations delivered as A CSV and B JSON."""

import argparse
import csv
import json
from pathlib import Path

from dms.yawdd import validate_interval


FIELDS = ["interval_id", "video_id", "file_path", "start_s", "end_s",
          "label", "split", "annotator", "source_file", "quality_flag",
          "source_interval_ids"]


def _manifest_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError("YawDD manifest is empty")
    return rows


def _plain_name(value: str) -> str:
    name = Path(value).name
    if "__" in name and name.startswith("video_"):
        name = name.split("__", 1)[1]
    return name


def _validate_row(raw: dict, manifest: dict[str, dict], source: Path,
                  default_annotator: str) -> dict:
    video_id = str(raw.get("video_id", "")).strip()
    if video_id not in manifest:
        raise ValueError(f"Unknown video_id in {source.name}: {video_id}")
    item = manifest[video_id]
    supplied_name = _plain_name(str(raw.get("file_name") or raw.get("video_path") or ""))
    expected_name = Path(item["file_path"]).name
    if supplied_name != expected_name:
        raise ValueError(
            f"Filename mismatch for {video_id}: {supplied_name!r} != {expected_name!r}"
        )
    supplied_split = str(raw.get("split", "")).strip().lower()
    supplied_split = "val" if supplied_split == "validation" else supplied_split
    if supplied_split != item["split"]:
        raise ValueError(
            f"Split mismatch for {video_id}: {supplied_split!r} != {item['split']!r}"
        )
    start_s, end_s = float(raw["start_s"]), float(raw["end_s"])
    validate_interval(start_s, end_s, float(item["duration_s"]))
    if str(raw.get("label", "yawn")).strip().lower() != "yawn":
        raise ValueError(f"Unsupported interval label for {video_id}")
    interval_id = str(raw.get("interval_id", "")).strip()
    if not interval_id:
        raise ValueError(f"Missing interval_id for {video_id}")
    return {
        "interval_id": interval_id, "video_id": video_id,
        "file_path": item["file_path"], "start_s": f"{start_s:.3f}",
        "end_s": f"{end_s:.3f}", "label": "yawn", "split": item["split"],
        "annotator": str(raw.get("annotator") or default_annotator).strip(),
        "source_file": source.name, "quality_flag": "ok",
        "source_interval_ids": interval_id,
    }


def load_a_csv(path: Path, manifest: dict[str, dict]) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [_validate_row(row, manifest, path, "A") for row in csv.DictReader(handle)]


def load_b_json(path: Path, manifest: dict[str, dict]) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("dataset") != "YawDD" or payload.get("time_unit") != "seconds":
        raise ValueError("B JSON must declare dataset=YawDD and time_unit=seconds")
    result = []
    for video in payload.get("videos", []):
        if video.get("review_status") != "complete":
            raise ValueError(f"Incomplete B review: {video.get('video_id')}")
        annotations = video.get("annotations", [])
        if int(video.get("yawn_count", -1)) != len(annotations):
            raise ValueError(f"B yawn_count mismatch: {video.get('video_id')}")
        for annotation in annotations:
            result.append(_validate_row(
                {**video, **annotation, "annotator": payload.get("annotator", "B")},
                manifest, path, "B",
            ))
    summary = payload.get("summary", {})
    if int(summary.get("total_yawn_intervals", -1)) != len(result):
        raise ValueError("B summary interval count mismatch")
    return result


def merge_labels(manifest_path: Path, a_csv: Path, b_json: Path,
                 output: Path, force: bool = False) -> dict:
    if output.exists() and not force:
        raise ValueError(f"Output exists; pass --force to replace it: {output}")
    manifest = _manifest_rows(manifest_path)
    rows = load_a_csv(a_csv, manifest) + load_b_json(b_json, manifest)
    ids = [row["interval_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate interval_id across annotation sources")
    per_video = {}
    for row in sorted(rows, key=lambda value: (value["video_id"], float(value["start_s"]))):
        intervals = per_video.setdefault(row["video_id"], [])
        if intervals and float(row["start_s"]) < float(intervals[-1]["end_s"]):
            previous = intervals[-1]
            overlap = min(float(previous["end_s"]), float(row["end_s"])) - max(
                float(previous["start_s"]), float(row["start_s"])
            )
            union = max(float(previous["end_s"]), float(row["end_s"])) - min(
                float(previous["start_s"]), float(row["start_s"])
            )
            if (overlap / union >= .80 and previous["annotator"] == row["annotator"]
                    and previous["source_file"] == row["source_file"]):
                previous["start_s"] = f"{min(float(previous['start_s']), float(row['start_s'])):.3f}"
                previous["end_s"] = f"{max(float(previous['end_s']), float(row['end_s'])):.3f}"
                previous["quality_flag"] = "merged_overlap"
                previous["source_interval_ids"] += f"|{row['interval_id']}"
                continue
            raise ValueError(f"Ambiguous overlapping intervals for {row['video_id']}")
        intervals.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(row for video in sorted(per_video) for row in per_video[video])
    split_counts = {split: sum(row["split"] == split for row in rows)
                    for split in ("val", "test")}
    normalized_count = sum(len(value) for value in per_video.values())
    return {"source_intervals": len(rows), "normalized_intervals": normalized_count,
            "merged_overlaps": len(rows) - normalized_count,
            "videos": len(per_video), "source_splits": split_counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("a_csv", type=Path)
    parser.add_argument("b_json", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("data/annotations/yawdd_yawn_intervals.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        summary = merge_labels(args.manifest, args.a_csv, args.b_json,
                               args.output, args.force)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
